r"""
Compare boundary quality (HD95 / surface distance) between a baseline and a CUL model,
case-wise and structure-wise; writes sample_metrics.csv used by the statistical scripts
in script/segmentation/uncertainty_project/.

Usage:
    python compare_boundaries.py --baseline_dir <run>/42 --uncertainty_dir <run>/42 \
        --uncertainty_tag cucl_no_calibration --pretrain ckpt.pth --root /path/to/data \
        --output_dir outputs/comparison --sliding_window
"""
import argparse
import os
import torch
import torch.nn.functional as F
from monai.inferers import sliding_window_inference
from models.build_segment import buidl_model
from dataset.dataset_seg import build_Anatomy_loader, build_Promis_loader, build_UCL_loader
from engine.segment import test_with_comparison


def _forward_logits(model, img, args, sliding_window, mc_samples, is_uncertainty_model):
    if sliding_window:
        def predictor(x):
            if is_uncertainty_model:
                output = model(
                    x,
                    compute_uncertainty=True,
                    mc_samples=mc_samples,
                    use_uncertainty_modulation=False,
                )
                logits = output[0] if isinstance(output, tuple) else output
            else:
                logits = model(x)

            if isinstance(logits, list):
                logits = logits[0]
            return logits

        return sliding_window_inference(img, args.crop_spatial_size, 4, predictor, overlap=0.5)

    if is_uncertainty_model:
        output = model(
            img,
            compute_uncertainty=True,
            mc_samples=mc_samples,
            use_uncertainty_modulation=False,
        )
        logits = output[0] if isinstance(output, tuple) else output
    else:
        logits = model(img)

    if isinstance(logits, list):
        logits = logits[0]
    return logits


def train_ensemble_weight(model_baseline, model_uncertainty, val_loader, args, sliding_window, mc_samples):
    if val_loader is None:
        print("Validation loader unavailable. Using equal-weight ensemble (0.5).")
        return 0.5

    model_baseline.eval()
    model_uncertainty.eval()

    raw_weight = torch.nn.Parameter(torch.tensor(0.0, device=args.device))
    optimizer = torch.optim.Adam([raw_weight], lr=args.ensemble_lr)
    max_batches = args.ensemble_max_batches if args.ensemble_max_batches > 0 else None

    print(
        f"\nTraining ensemble blender on validation set "
        f"(epochs={args.ensemble_epochs}, lr={args.ensemble_lr}, max_batches={max_batches})"
    )

    for epoch in range(args.ensemble_epochs):
        epoch_loss = 0.0
        steps = 0

        for batch_idx, (img, gt, _) in enumerate(val_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            img = img.to(args.device)
            gt = gt.to(args.device)

            with torch.no_grad():
                logits_baseline = _forward_logits(
                    model_baseline, img, args, sliding_window, mc_samples, is_uncertainty_model=False
                )
                logits_uncertainty = _forward_logits(
                    model_uncertainty, img, args, sliding_window, mc_samples, is_uncertainty_model=True
                )

            weight_unc = torch.sigmoid(raw_weight)
            if args.out_channels == 1:
                p_baseline = torch.sigmoid(logits_baseline)
                p_uncertainty = torch.sigmoid(logits_uncertainty)
                p_ensemble = (1.0 - weight_unc) * p_baseline + weight_unc * p_uncertainty
                loss = F.binary_cross_entropy(p_ensemble.clamp(min=1e-6, max=1.0 - 1e-6), gt.float())
            else:
                p_baseline = torch.softmax(logits_baseline, dim=1)
                p_uncertainty = torch.softmax(logits_uncertainty, dim=1)
                p_ensemble = (1.0 - weight_unc) * p_baseline + weight_unc * p_uncertainty
                loss = F.nll_loss(torch.log(p_ensemble.clamp(min=1e-8)), gt.squeeze(1).long())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            steps += 1

        print(
            f"  [ensemble epoch {epoch + 1}/{args.ensemble_epochs}] "
            f"loss={epoch_loss / max(1, steps):.6f}, "
            f"weight_uncertainty={torch.sigmoid(raw_weight).item():.4f}"
        )

    return torch.sigmoid(raw_weight).item()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline_dir', type=str, required=True)
    parser.add_argument('--uncertainty_dir', type=str, required=True)
    parser.add_argument('--uncertainty_tag', type=str, default='cucl_no_calibration')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--pretrain', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='anatomy')
    parser.add_argument('--sliding_window', action="store_true",
                    help='Use sliding window inference for test')
    parser.add_argument('--threshold_hd95', type=float, default=0.5)
    parser.add_argument('--threshold_sd', type=float, default=0.1)
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--ensemble_mode', type=str, default='trained', choices=['trained', 'fixed'])
    parser.add_argument('--ensemble_fixed_weight', type=float, default=0.5)
    parser.add_argument('--ensemble_epochs', type=int, default=5)
    parser.add_argument('--ensemble_lr', type=float, default=0.1)
    parser.add_argument('--ensemble_max_batches', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--root', type=str, default='/SAN/medic/PosPretrain')
    parser.add_argument('--data20', action='store_true', default=False)
    parser.add_argument('--data_num', type=int, default=-1)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--layer_decay', type=float, default=0.75)
    parser.add_argument("--prompt", action="store_true", help="Use visual prompt tuning")
    parser.add_argument("--lr", type=float, default=1e-4, metavar="LR", help="learning rate (absolute lr)")
    parser.add_argument("--in_channels", type=int, default=1, help="Number of input channels")
    parser.add_argument("--out_channels", type=int, default=8, help="Number of output channels")
    parser.add_argument("--mc_samples", type=int, default=5, help="Number of MC samples")
    parser.add_argument("--model", type=str, default='vit_upernet_uncer', help="Model architecture")
    parser.add_argument("--train", type=str, default='fintune', help="Training strategy")
    parser.add_argument("--state", type=str, default='test', help="Execution state for model builder")
    parser.add_argument('--spacing', type=float, nargs=3, default=(1.0, 0.5, 0.5),
                        help='Voxel spacing used for HD95/SurfaceDistance metrics')
    parser.set_defaults(prompt=False)
    
    args = parser.parse_args()
    
    # Set device
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Dataset-specific settings
    if args.dataset == 'anatomy':
        args.crop_spatial_size = (64, 256, 256)
        args.out_channels = 8
        args.in_channels = 1
        _, val_loader, test_loader = build_Anatomy_loader(args)
    elif args.dataset == 'promis':
        args.crop_spatial_size = (64, 256, 256)
        args.out_channels = 1
        args.in_channels = 1
        _, val_loader, test_loader = build_Promis_loader(args)
    elif args.dataset == 'ucl':
        args.crop_spatial_size = (64, 256, 256)
        args.out_channels = 1
        args.in_channels = 1
        _, val_loader, test_loader = build_UCL_loader(args)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Build baseline model (trained WITHOUT CUCL based on your shell script)
    print("Loading baseline model...")
    args_baseline = argparse.Namespace(**vars(args))
    args_baseline.output_dir = args.baseline_dir
    args_baseline.use_mc_dropout = False
    args_baseline.use_cucl = False  # False for baseline
    args_baseline.model = 'vit_upernet_uncer'
    args_baseline.case = 'test'
    args_baseline.train = 'fintune'
    args_baseline.state = 'test'
    model_baseline, _ = buidl_model(args_baseline, args.device)
    baseline_ckpt = os.path.join(args.baseline_dir, 'best.pth.tar')
    print(f"Loading baseline checkpoint from {baseline_ckpt}")
    model_baseline.load_state_dict(torch.load(baseline_ckpt)['model'])
    print(f"Loaded baseline model (use_cucl={False})")
    
    # Build uncertainty model with flags aligned to selected tag
    print("\nLoading uncertainty model...")
    args_uncertainty = argparse.Namespace(**vars(args))
    args_uncertainty.output_dir = args.uncertainty_dir
    args_uncertainty.use_mc_dropout = True

    uncertainty_tag = args.uncertainty_tag
    if uncertainty_tag in ('full_um_cucl', 'cucl_only', 'cucl_no_calibration', 'cucl_class_based'):
        args_uncertainty.use_cucl = True
    else:
        args_uncertainty.use_cucl = False

    if uncertainty_tag == 'cucl_class_based':
        args_uncertainty.cucl_class_based = True
    else:
        args_uncertainty.cucl_class_based = False

    if uncertainty_tag in ('full_um_cucl', 'modulation_only', 'boundary_only', 'calibration_only'):
        args_uncertainty.use_uncertainty_modulation = True
    else:
        args_uncertainty.use_uncertainty_modulation = False

    args_uncertainty.model = 'vit_upernet_uncer'
    args_uncertainty.case = 'test'
    args_uncertainty.train = 'fintune'
    args_uncertainty.state = 'test'
    model_uncertainty, _ = buidl_model(args_uncertainty, args.device)
    uncertainty_ckpt = os.path.join(args.uncertainty_dir, 'best.pth.tar')
    print(f"Loading uncertainty checkpoint from {uncertainty_ckpt}")
    model_uncertainty.load_state_dict(torch.load(uncertainty_ckpt)['model'])
    print(
        f"Loaded uncertainty model (tag={uncertainty_tag}, "
        f"use_cucl={args_uncertainty.use_cucl}, "
        f"use_uncertainty_modulation={args_uncertainty.use_uncertainty_modulation})"
    )

    if args.ensemble_mode == 'trained':
        ensemble_weight = train_ensemble_weight(
            model_baseline=model_baseline,
            model_uncertainty=model_uncertainty,
            val_loader=val_loader,
            args=args,
            sliding_window=args.sliding_window,
            mc_samples=args.mc_samples,
        )
    else:
        ensemble_weight = args.ensemble_fixed_weight

    ensemble_weight = max(0.0, min(1.0, float(ensemble_weight)))
    print(f"Using ensemble uncertainty weight: {ensemble_weight:.4f} (baseline weight={1.0 - ensemble_weight:.4f})")

    with open(os.path.join(args.output_dir, 'ensemble_model.txt'), 'w', encoding='utf-8') as f:
        f.write(f"ensemble_mode={args.ensemble_mode}\n")
        f.write(f"ensemble_uncertainty_weight={ensemble_weight:.6f}\n")
        f.write(f"ensemble_baseline_weight={1.0 - ensemble_weight:.6f}\n")
        f.write(f"uncertainty_tag={uncertainty_tag}\n")
    
    # Run comparison
    print("\nRunning boundary comparison...")
    sample_metrics = test_with_comparison(
        model_baseline=model_baseline,
        model_uncertainty=model_uncertainty,
        test_loader=test_loader,
        args=args,
        use_mc_dropout=True,
        mc_samples=args.mc_samples,
        sliding_window=args.sliding_window,
        save_dir=args.output_dir,
        threshold_hd95=args.threshold_hd95,
        threshold_sd=args.threshold_sd,
        top_k=args.top_k,
        ensemble_weight_uncertainty=ensemble_weight,
    )
    
    print("\nComparison complete!")
    print(f"Results saved to: {args.output_dir}")

if __name__ == '__main__':
    main()