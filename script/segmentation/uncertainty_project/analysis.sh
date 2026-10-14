#$ -wd /home/bisidiqi/profound/uncer_module_logs/seg_job
#$ -l gpu=true,gpu_type=a6000|a40|a100|a100_80|a100_dgx
#$ -l tmem=32G
#$ -R y 
#$ -l h_rt=01:0:0
#$ -N boundary_comparison
source /share/apps/source_files/cuda/cuda-11.8.source
export PATH=/share/apps/miniconda3/bin:$PATH
cd /home/bisidiqi/profound/ProFound-downstream-yp/CUL_project
source activate proFM
nvidia-smi
hostname
EPOCH=799

python -u compare_boundaries.py \
    --baseline_dir /SAN/medic/PosPretrain/project1/outputs/seg/promis_profound_vit_zone_baseline_${EPOCH}epoch/anatomy/vit_upernet_uncer/fintune/42 \
    --uncertainty_dir /SAN/medic/PosPretrain/project1/outputs/seg/promis_profound_vit_zone_um_ft_6${EPOCH}epoch/anatomy/vit_upernet_uncer/fintune/42 \
    --output_dir /SAN/medic/PosPretrain/project1/outputs/seg/boundary_comparison \
    --pretrain /SAN/medic/profound/output_mae/output_mae_wo_norm_r8000/checkpoint-${EPOCH}.pth \
    --root /SAN/medic/foundation/downstream_data \
    --dataset anatomy \
    --sliding_window \
    --lr 1e-4 \
    --mc_samples 5 \
    --threshold_hd95 0.5 \
    --threshold_sd 0.1 \
    --top_k 10 \


