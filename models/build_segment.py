"""
Builds the ViT-B (ProFound MAE weights) + ViTAdapter + UperNet model and the AdamW
optimizer with layer-wise LR decay. CUL options are read from args (--use_cucl, --cucl_*).
"""
from models.vit import vit_base_patch16_dec512d8b
from models.upernet_module import UperNet, ViTAdapter
import torch
import os
from util.lr_decay import param_groups_lrd


def buidl_model(args, device):
    if args.model != "vit_upernet_uncer":
        raise NotImplementedError(
            f"CUL_project only supports model='vit_upernet_uncer', got '{args.model}'"
        )

    vit = vit_base_patch16_dec512d8b(in_chans=args.in_channels)
    if args.pretrain is None:
        raise NotImplementedError(f"No pretrained weight")
    if not os.path.exists(args.pretrain):
        raise FileExistsError(f"{args.pretrain} Not exists")

    if args.state != "test":
        msg = vit.load_state_dict(
            torch.load(args.pretrain, map_location="cpu")["model"], strict=False
        )
        print(msg.missing_keys)
    else:
        pass

    adapter = ViTAdapter(
        img_size=args.crop_spatial_size,
        patch_size=(16, 32, 32),
        embed_dim=768,
    )

    # Check if CUCL is enabled
    use_cucl = getattr(args, 'use_cucl', False)

    model = UperNet(
        encoder=vit,
        in_channels=[768] * 4,
        out_channels=args.out_channels,
        adapter=adapter,
        out_indices=[3, 5, 7, 11],
        dropout_rate=0.1,
        use_cucl=use_cucl,
        cucl_feature_dim=512,  # Match hidden_size in UperNetHead
        cucl_proj_dim=128,
        cucl_temperature=getattr(args, 'cucl_temperature', 0.1),
        cucl_uncertainty_threshold=getattr(args, 'cucl_uncertainty_threshold', 0.5),
        cucl_num_samples=getattr(args, 'cucl_num_samples', 256),
        cucl_class_based=getattr(args, 'cucl_class_based', False),
    )
    model = model.to(device)

    backbone_param_groups = param_groups_lrd(model.encoder, args.weight_decay)
    decoder_param_groups = (
        decoder_parameters(model.adapter, args.weight_decay, lr=args.lr)
        + decoder_parameters(model.decode_head, args.weight_decay, lr=args.lr)
        + decoder_parameters(model.auxiliary_head, args.weight_decay, lr=args.lr)
    )

    # ========== ADD ALL UNCERTAINTY PARAMETERS ==========
    uncertainty_param_groups = [
        # Scalar parameters with higher LR
        {
            "params": [model.unc_threshold],
            "weight_decay": 0.0,
            "lr": args.lr * 10,
            "name": "unc_threshold"
        },

        # Boundary attention modules with higher LR
        {
            "params": list(model.boundary_attention.parameters()),
            "weight_decay": args.weight_decay,
            "lr": args.lr * 2,
            "name": "boundary_attention"
        },
    ]

    # Add CUCL parameters if enabled
    cucl_param_groups = []
    if use_cucl and model.cucl_module is not None:
        cucl_param_groups = [
            {
                "params": list(model.cucl_module.parameters()),
                "weight_decay": args.weight_decay,
                "lr": args.lr * 2,
                "name": "cucl_module"
            },
        ]

    optimizer = torch.optim.AdamW(
        backbone_param_groups + decoder_param_groups + uncertainty_param_groups + cucl_param_groups,
        lr=args.lr,
    )

    # Print parameter groups for verification
    print("\n=== Optimizer Parameter Groups ===")
    for i, pg in enumerate(optimizer.param_groups):
        name = pg.get('name', f'group_{i}')
        num_params = sum(p.numel() for p in pg['params'])
        print(f"  {name}: {num_params} params, lr={pg.get('lr', args.lr)}")
    print("===================================\n")

    return model, optimizer


def decoder_parameters(
    model: torch.nn.Module, weight_decay=1e-5, no_weight_decay_list=(), lr=1e-3
):
    no_weight_decay_list = set(no_weight_decay_list)
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param.ndim <= 1 or name.endswith(".bias") or name in no_weight_decay_list:
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {"params": no_decay, "weight_decay": 0.0, "lr": lr},
        {"params": decay, "weight_decay": weight_decay, "lr": lr},
    ]
