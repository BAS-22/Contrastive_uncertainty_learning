#$ -wd /home/bisidiqi/profound/uncer_module_logs/seg_job
#$ -l gpu=true,gpu_type=a6000|a40|a100|a100_80|a100_dgx
#$ -l tmem=32G
#$ -R y 
#$ -l h_rt=01:0:0
#$ -N promis_profound_vit_zone_baseline_ts
source /share/apps/source_files/cuda/cuda-11.8.source
export PATH=/share/apps/miniconda3/bin:$PATH
cd /home/bisidiqi/profound/ProFound-downstream-yp/CUL_project
source activate proFM
nvidia-smi
hostname
EPOCH=799
python -u main_segmentation.py \
    --batch_size 1 \
    --model vit_upernet_uncer \
    --output_dir /SAN/medic/PosPretrain/project1/outputs/seg/promis_profound_vit_zone_baseline_${EPOCH}epoch \
    --log_dir /SAN/medic/PosPretrain/project1/outputs/seg/promis_profound_vit_zone_baseline_${EPOCH}epoch \
    --train fintune \
    --case test \
    --state test \
    --pretrain /SAN/medic/profound/output_mae/output_mae_wo_norm_r8000/checkpoint-${EPOCH}.pth \
    --mc_samples 1 \
    --sliding_window \
    --epochs 100 \
    --dataset anatomy \
    --warmup_epochs 10 \
    --lr 1e-4 \
    --seed 42