#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=100:00:00
#SBATCH --mem=32G
#SBATCH --partition=deeplearn
#SBATCH -A punim0478
#SBATCH --gres=gpu:v100:1
#SBATCH -q gpgpudeeplearn
#SBATCH -o log.out

#step1: generate graph
# python utils/conceptnet_4_path.py
# python utils/cpnet40rel_sw2rel_4_path.py

# #step2: sample path
# #python sample_path_rw.py --data_dir data/cpnet7rel/ --output_dir data/sample_path_cpnet7rel

# python sample_path_rw.py --data_dir data/cpnet_swow/ --output_dir data/sample_path_cpnet_swow

# #step3: shuffle and split path to train/dev/test
# # sh ./split_path_files.sh data/sample_path_cpnet7rel

# sh ./split_path_files.sh data/sample_path_cpnet_swow


#####
python utils/conceptnet_47rel.py
