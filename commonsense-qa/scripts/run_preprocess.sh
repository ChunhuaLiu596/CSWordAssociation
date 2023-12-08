#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=100:00:00
#SBATCH --mem=64G
#SBATCH --partition=deeplearn
#SBATCH -A punim0478
#SBATCH --gres=gpu:v100sxm2:1
#SBATCH --constraint=dlg3
#SBATCH -q gpgpudeeplearn
#SBATCH -o log.out
###SBATCH --gres=gpu:v100:1


#only run kg preprocess
# python preprocess.py cpnet --run common
# python preprocess.py cpnet7rel --run common
# python preprocess.py cpnet1rel --run common
# python preprocess.py swow --run common
# python preprocess.py swow1rel --run common -p 20

##grounding: csqa
# python preprocess.py cpnet --run csqa
# python preprocess.py swow --run csqa
# python preprocess.py cpnet7rel --run csqa -p 20
# python preprocess.py cpnet1rel --run csqa -p 20
# python preprocess.py swow1rel --run csqa -p 20

##grounding: obqa
# python preprocess.py cpnet --run obqa -p 20
# python preprocess.py cpnet7rel --run obqa 
# python preprocess.py cpnet1rel --run obqa 
# python preprocess.py swow1rel --run obqa
# python preprocess.py swow --run obqa

# python preprocess.py swow --run obqa -p 20


## cpnet_swow
# python utils/conceptnet_swow.py # merge csv and vocab
# python preprocess.py cpnet_swow --run common -p 20
# python preprocess.py cpnet_swow --run csqa -p 30 
#python preprocess.py cpnet_swow --run obqa -p 30 

# python utils/conceptnet_pos.py False


#mcscript
python preprocess.py swow --run mcscript -p 30
python preprocess.py cpnet --run mcscript -p 30