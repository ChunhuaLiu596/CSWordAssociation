#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=100:00:00
#SBATCH --mem=32G
#SBATCH --partition=deeplearn
#SBATCH -A punim0478
#SBATCH --gres=gpu:v100:1
#SBATCH -q gpgpudeeplearn
#SBATCH -o log/log.log


python -u utils/kgsrc/pos_entities.py --debug $1 --reload $2
# python -u utils/kgsrc/pos_swow_cue.py

# python -u utils/kgsrc/rel_num_plot.py 
