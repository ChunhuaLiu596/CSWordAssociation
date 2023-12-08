#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=100:00:00
#SBATCH --mem=32G
#SBATCH --partition=deeplearn
#SBATCH -A punim0478
#SBATCH --gres=gpu:v100:1
#SBATCH -q gpgpudeeplearn


source ./config/path_generate.config

# nohup 
python -u calc_path_embedding.py \
    --data_dir $data_dir \
    --generator_type $generator_type \
    --batch_size $batch_size \
    --output_len $output_len \
    --context_len $context_len \
    --gpu_device $gpu_device \
    --pretrain_generator_ckpt $pretrain_generator_ckpt\
    --kg_name $kg_name\
    > ./saved_models/debug_save_emb.log 
    # 2>&1 &

echo ./saved_models/debug_save_emb.log 
