# CUL: Contrastive Uncertainty Learning for Robust Multi-Structure Segmentation in Lower-Pelvic MR

[![Paper](https://img.shields.io/badge/Paper-UNSURE%202026%20(MICCAI%20Satellite)-blue)](https://papers.miccai.org/miccai-2026-sat/UNSURE2026_069.html)
[![PDF](https://img.shields.io/badge/PDF-open%20access-green)](https://papers.miccai.org/miccai-2026-sat/paper/UNSURE2026_069.pdf)

Official PyTorch implementation of **Contrastive Uncertainty Learning (CUL)**, accepted at **UNSURE 2026** (Uncertainty for Safe Utilization of Machine Learning in Medical Imaging), a MICCAI 2026 satellite event.

> **Contrastive uncertainty learning for robust multi-structure segmentation in lower-pelvic MR**
> Bilal Sidiqi, Wen Yan, Yipei Wang, Shaheer U. Saeed, Daniel C. Alexander, Yipeng Hu
> *MICCAI 2026 Workshops and Challenges, LNCS 17260*
> [[Paper page]](https://papers.miccai.org/miccai-2026-sat/UNSURE2026_069.html) [[PDF]](https://papers.miccai.org/miccai-2026-sat/paper/UNSURE2026_069.pdf)

## Overview

CUL is a supervised training framework that uses predictive uncertainty to guide representation learning directly, rather than treating uncertainty only as a post-hoc quality indicator.

During training, Monte Carlo (MC) Dropout in the decoder (M = 5 stochastic passes) produces a voxel-wise predictive-entropy map. Decoder features, taken before the final classifier, are projected into a contrastive embedding space where:

- **Anchors and positives** are sampled from voxels that have low uncertainty (below τ = 0.5) and are predicted correctly. A positive is a same-class voxel from the anchor pool.
- **Negatives** are sampled from voxels that have high uncertainty or are misclassified.

An InfoNCE-style loss pulls confident, correct features together and pushes uncertain or misclassified features away from them. This prevents feature collapse at ambiguous boundaries. The contrastive term is added to the standard DiceCE segmentation loss.

The backbone is a 3D ViT-B encoder pretrained with masked autoencoding ([ProFound](https://arxiv.org/abs/2603.03961)), followed by a UperNet decoder.

<!-- Optional: add a method figure, e.g. ![CUL overview](assets/cul_overview.png) -->

## Results

These results come from 117 held-out test cases of a 589-scan multi-institute T2W prostate MRI dataset with 8 anatomical structures (mean ± std):

| Metric | Baseline | **CUL** | Class-based SupCon | Classic boundary loss |
|---|---|---|---|---|
| Dice ↑ | 0.8335 ± 0.0647 | **0.8338 ± 0.0651** | 0.8331 ± 0.0637 | **0.8338 ± 0.0634** |
| HD95 (mm) ↓ | 5.93 ± 8.14 | **5.41 ± 4.88** | 5.37 ± 4.66 | 5.63 ± 5.23 |
| SD (mm) ↓ | 1.61 ± 2.51 | **1.42 ± 1.81** | 1.44 ± 1.89 | 1.46 ± 1.92 |
| Structures with significant SD improvement (Wilcoxon) | – | **6 / 8** | 5 / 8 | 4 / 8 |

The paper also reports per-structure results and a hard-case analysis.

## Contents

- `main_segmentation.py`: entry point for training and testing.
- `engine/segment.py`: training, validation and test loops, MC-Dropout uncertainty, and boundary-comparison utilities.
- `models/upernet_module.py`: UperNet decoder, ViT adapter, and the CUL loss (`ContrastiveUncertaintyCalibrationLoss`).
- `models/vit.py`, `models/build_segment.py`: the 3D ViT-B encoder and the model and optimizer builder.
- `dataset/dataset_seg.py`: NIfTI dataset, MONAI transforms, and dataloaders.
- `compare_boundaries.py`: paired baseline-vs-CUL boundary comparison (HD95 / SD per case and per structure).
- `script/segmentation/uncertainty_project/`: launch scripts for training, ablations and inference, plus the statistical-analysis scripts (Wilcoxon, structure-wise report, HD95 variance tests).
- `spilt/anatomy/`: CSV files for the train, validation and test splits.
- `scripts/train_cul.sh`, `scripts/test_cul.sh`: launchers that don't need a scheduler (`baseline`, `cul`, `class_based`, `classic_boundary`).
- `examples/sample_split.csv`: example split CSV.
- `util/`: distributed/misc helpers, LR schedule, and layer-wise LR decay.

## Installation

```bash
git clone https://github.com/BAS-22/Contrastive_uncertainty_learning.git
cd Contrastive_uncertainty_learning
conda create -n cul python=3.10 -y
conda activate cul
pip install -r requirements.txt
```

Tested with CUDA 11.8. Change the PyTorch build in `requirements.txt` to match your CUDA version.

## Data Format

Each split is a CSV file with two columns. The paths are relative to `--root`:

```csv
t2w,mask
anatomy_align/case_0001_img.nii,anatomy_align/case_0001_mask.nii
anatomy_align/case_0002_img.nii,anatomy_align/case_0002_mask.nii
```

- `t2w`: 3D T2-weighted volume (NIfTI).
- `mask`: integer label map with the same geometry. `0` is background and `1–8` are the structures: bladder, bone, obturator internus, central gland, transition zone, rectum, seminal vesicle, and neurovascular bundle.
- The paper uses volumes of size `64 × 256 × 256` (D × H × W). Intensity normalisation and augmentation (rotation, zoom, flips) happen inside the loader.
- The loaders read `spilt/anatomy/{train,val,test}.csv`. To use your own data, replace these files or edit `build_BpAnatomy_loader` in `dataset/dataset_seg.py`.
- Use de-identified filenames only.
- Only `--dataset anatomy` is supported in this release. The other loaders in `dataset/dataset_seg.py` (UCL / Promis*) are kept for reference and need split files that aren't distributed.

The dataset used in the paper cannot be redistributed. See `examples/sample_split.csv` for the expected format.

## Pretrained Encoder

CUL fine-tunes a ViT-B encoder that was pretrained with MAE on prostate MRI ([ProFound](https://arxiv.org/abs/2603.03961)). Download the checkpoint and pass its path with `--pretrain`:

```text
pretrained/
  profound_mae_checkpoint-799.pth
```

<!-- TODO: add the download link for the ProFound weights -->

## Training

**CUL** (the configuration reported in the paper):

```bash
python main_segmentation.py \
  --state train \
  --model vit_upernet_uncer \
  --train fintune \
  --case prenc \
  --dataset anatomy \
  --root /path/to/data_root \
  --pretrain pretrained/profound_mae_checkpoint-799.pth \
  --output_dir outputs/cul --log_dir outputs/cul \
  --batch_size 1 --epochs 100 --warmup_epochs 10 --lr 1e-4 --seed 42 \
  --sliding_window \
  --use_mc_dropout --mc_samples 5 \
  --use_cucl \
  --cucl_contrastive_weight 0.05 --cucl_calibration_weight 0.0 \
  --cucl_temperature 0.1 --cucl_uncertainty_threshold 0.5 --cucl_num_samples 256 \
  --boundary_loss_weight 0.0 --calibration_loss_weight 0.0 \
  --aux_loss_warmup_epochs 20
```

**Baseline** (DiceCE only): use the same command without the `--use_mc_dropout` and `--use_cucl*` flags.

**Ablations from the paper:**

| Variant | Extra flags (in addition to the CUL command) |
|---|---|
| Class-based supervised contrastive (no uncertainty guidance) | `--cucl_class_based` |
| Classic boundary loss (Kervadec et al.) | Drop the MC-Dropout and CUL flags, then add `--use_classic_boundary_loss --classic_boundary_loss_weight 0.01` |

`scripts/train_cul.sh` wraps these as `bash scripts/train_cul.sh {baseline|cul|class_based|classic_boundary}`.

Checkpoints are saved to `<output_dir>/anatomy/vit_upernet_uncer/fintune/<seed>/{best,last}.pth.tar`. To resume, pass `--resume <path>/last.pth.tar`.

## Evaluation

```bash
python main_segmentation.py \
  --state test \
  --model vit_upernet_uncer \
  --train fintune \
  --case test \
  --dataset anatomy \
  --root /path/to/data_root \
  --pretrain pretrained/profound_mae_checkpoint-799.pth \
  --output_dir outputs/cul --log_dir outputs/cul \
  --sliding_window --use_mc_dropout --mc_samples 5 --uncertainty \
  --use_cucl
```

This loads `best.pth.tar` from the same `--output_dir` that was used for training, and reports Dice, HD95 and surface distance on the test split. For a case-wise and structure-wise comparison between two trained models (as in the paper):

```bash
python compare_boundaries.py \
  --baseline_dir outputs/baseline/anatomy/vit_upernet_uncer/fintune/42 \
  --uncertainty_dir outputs/cul/anatomy/vit_upernet_uncer/fintune/42 \
  --uncertainty_tag cucl_no_calibration \
  --pretrain pretrained/profound_mae_checkpoint-799.pth \
  --root /path/to/data_root \
  --output_dir outputs/comparison \
  --sliding_window
```

The statistical tests in the paper (paired Wilcoxon, per-structure SD, HD95 variance) can be reproduced from the resulting `sample_metrics.csv` with the scripts in `script/segmentation/uncertainty_project/`.

## Notes

- Requires Python 3.10+ and a GPU with about 57 GB of memory for `64×256×256` volumes with M = 5 MC passes. An A40 or A100 80GB was used in the paper.
- The launch scripts use `--batch_size 1` (no gradient accumulation) and the AdamW optimizer with layer-wise LR decay (`models/build_segment.py`).
- With M = 5, CUL training takes about 0.9 s/iteration (about 10 h for 100 epochs), compared with about 0.4 s/iteration for the baseline.
- The `script/segmentation/uncertainty_project/*.sh` files are SGE (`qsub`) job scripts from our cluster. Edit the paths and scheduler headers to fit your environment.
- Do not commit private data, checkpoints, pretrained weights, or output folders.

## Citation

If you find this work useful, please cite:

```bibtex
@InProceedings{SidBil_Contrastive_MICCAISAT2026,
  author    = {Sidiqi, Bilal and Yan, Wen and Wang, Yipei and Saeed, Shaheer U. and Alexander, Daniel C. and Hu, Yipeng},
  title     = {Contrastive uncertainty learning for robust multi-structure segmentation in lower-pelvic MR},
  booktitle = {Medical Image Computing and Computer Assisted Intervention -- MICCAI 2026 Workshops and Challenges},
  year      = {2026},
  publisher = {Springer Nature Switzerland},
  volume    = {LNCS 17260}
}
```

## Acknowledgements

This codebase builds on [MAE](https://github.com/facebookresearch/mae), [DeiT](https://github.com/facebookresearch/deit), [BEiT](https://github.com/microsoft/unilm/tree/master/beit), [timm](https://github.com/huggingface/pytorch-image-models) and [MONAI](https://github.com/Project-MONAI/MONAI), and on the ProFound pretrained encoder. We thank the authors for releasing their code.

## License

<!-- TODO: choose a license. Parts of this code are derived from MAE (CC BY-NC 4.0); see CHANGES_TO_APPLY.md, item 1. -->
See [LICENSE](LICENSE) for details.
