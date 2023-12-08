#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=100:00:00
#SBATCH --mem=32G
#SBATCH --partition=deeplearn
#SBATCH -A punim0478
#SBATCH --gres=gpu:v100:1
#SBATCH -q gpgpudeeplearn
#SBATCH -o log/csqa-test-pvalue.log
####SBATCH -o log/obqa-test-pvalue.log


# python -u ./analysis/recall_relatedto.py
# python -u ./analysis/negation.py $1
# python -u ./analysis/node_pair_distance.py
#python -u ./local_graph_overlap.py --dataset csqa
#python -u ./local_graph_overlap.py --dataset obqa
python -u ./utils/conceptnet_source.py False


#python -u ./utils/kgsrc/pos_swow_cue.py
# python -u ./utils/p-value.py csqa log/csqa_pvalue.csv
# python -u ./utils/p-value.py obqa log/obqa_pvalue.csv

# python -u ./utils/csqa_question_types.py


