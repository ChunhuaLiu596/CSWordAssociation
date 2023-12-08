#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=100:00:00
#SBATCH --mem=32G
#SBATCH --partition=deeplearn
#SBATCH -A punim0478
#SBATCH --gres=gpu:v100:1
#SBATCH -q gpgpudeeplearn

ikg_names=('cpnet' 'swow')
okg_names=('cpnet1rel' 'swow1rel')
iemb_paths=('./data/transe/glove.transe.sgd.rel.npy' './data/swow/glove.TransE.SGD.rel.npy')
oemb_paths=('data/cpnet1rel/glove.transe.sgd.rel.npy' './data/swow1rel/glove.TransE.SGD.rel.npy')

for ((i=0; i<${#ikg_names[@]}; i++)); do
    echo ${ikg_names[$i]}, ${okg_names[$i]}
    python utils/extract_sub_rel_emb.py --input_kg_name ${ikg_names[$i]}\
    --output_kg_name ${okg_names[$i]}\
    --input_emb_path ${iemb_paths[$i]}\
    --output_emb_path ${oemb_paths[$i]}
done