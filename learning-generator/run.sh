#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=100:00:00
#SBATCH --mem=32G
#SBATCH --partition=deeplearn
#SBATCH -A punim0478
#SBATCH --gres=gpu:v100:1
#SBATCH -q gpgpudeeplearn

source ./config/params.config

LOG_FOLDER="./checkpoints/"

mkdir -p $LOG_FOLDER

#nohup 
python -u main.py \
    --data_dir $data_dir \
    --save_dir $save_dir \
    --model $model \
    --learning_rate $learning_rate \
    --warmup_steps $warmup_steps \
    --weight_decay $weight_decay \
    --num_epoch $num_epoch \
    --batch_size $batch_size \
    --gpu_device 0 \
    > $LOG_FOLDER/$params.log 
#2>&1 &

echo $LOG_FOLDER/$params.log
