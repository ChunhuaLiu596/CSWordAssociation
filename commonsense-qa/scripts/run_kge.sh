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

#Pre-train KG embeddings with TransE
echo "#Train KG embeddings"
#python n-n.py
# python embeddings/create_embeddings_glove.py paths.cfg swow_3rel_freq1
# python embeddings/train_kge.py TransH SGD 1 data/swow_3rel_freq1/  swow_3rel_freq1 --epoch 3
# python embeddings/kge_to_npy.py TransH SGD swow_3rel_freq1

#python embeddings/create_embeddings_glove.py paths.cfg swow_3rel_freq1
# python embeddings/train_kge.py HolE Adagrad 1 data/swow_3rel_freq1/  swow_3rel_freq1 --epoch 1000
# python embeddings/kge_to_npy.py HolE Adagrad swow_3rel_freq1

#python embeddings/create_embeddings_glove.py paths.cfg swow_3rel_freq1
#python embeddings/train_kge.py DistMult Adagrad 1 data/swow_3rel_freq1/  swow_3rel_freq1 --epoch 1000
#python embeddings/kge_to_npy.py DistMult Adagrad swow_3rel_freq1

#
# python embeddings/create_embeddings_glove.py paths.cfg swow_3rel_freq1
# python embeddings/TransE.py SGD 1 data/swow_3rel_freq1/  swow_3rel_freq1 --epoch 3
# python embeddings/TransE_to_npy.py swow_3rel_freq1
# python embeddings/convert_to_gz.py swow_3rel_freq1



# Train cpnet
# python3.6 embeddings/create_embeddings_glove.py paths.cfg cpnet
#python embeddings/train_kge.py TransH SGD 1 data/cpnet/  cpnet --epoch 1000
#python embeddings/kge_to_npy.py TransH SGD cpnet

# python embeddings/train_kge.py HolE Adagrad 1 data/cpnet/  cpnet --epoch 1000
# python embeddings/kge_to_npy.py HolE Adagrad cpnet


#python embeddings/kge_to_npy.py TransE SGD cpnet
#  python3.6 embeddings/create_embeddings_glove.py paths.cfg cpnet
# python3.6 embeddings/TransE.py SGD 1 data/cpnet/  cpnet --epoch 1000
# python3.6 embeddings/TransE_to_npy.py cpnet


#python embeddings/create_embeddings_glove.py paths.cfg cpnet
#python embeddings/train_kge.py DistMult Adagrad 1 data/cpnet/  cpnet --epoch 1000
#python embeddings/kge_to_npy.py DistMult Adagrad cpnet

#test
# python embeddings/test_kge.py TransE SGD 1 data/swow_3rel_freq1  swow_3rel_freq1 --test_only --test_triple_classification
# python embeddings/test_kge.py TransH SGD 1 data/swow_3rel_freq1  swow_3rel_freq1 --test_only --test_triple_classification
# python embeddings/test_kge.py DistMult Adagrad 1 data/swow_3rel_freq1  swow_3rel_freq1 --test_only --test_triple_classification
#
#
# python embeddings/test_kge.py TransE SGD 1 data/cpnet  cpnet --test_only --test_triple_classification
# python embeddings/test_kge.py TransH SGD 1 data/cpnet  cpnet --test_only --test_triple_classification
# python embeddings/test_kge.py DistMult Adagrad 1 data/cpnet  cpnet --test_only --test_triple_classification

# python embeddings/test_kge.py TransE SGD 1 data/swow_3rel_freq1/  swow_3rel_freq1 --test_only



#### train

#python3.6 embeddings/create_embeddings_glove.py paths.cfg swow_3rel_freq1
#python embeddings/train_kge.py TransE SGD 1 data/swow_3rel_freq1/  swow_3rel_freq1 --epoch 1000
#python embeddings/kge_to_npy.py TransE SGD swow_3rel_freq1 
#
#
#python3.6 embeddings/create_embeddings_glove.py paths.cfg cpnet
#python embeddings/train_kge.py TransE SGD 1 data/cpnet/  cpnet --epoch 1000
#python embeddings/kge_to_npy.py TransE SGD cpnet 


# source ./paths.config
# python utils/generate_openke_data.py --train_path data/swow_2rel_freq1/conceptnet.en.csv\
#   --concept_path data/swow_2rel_freq1/concept.txt\
#   --relation_path data/swow_2rel_freq1/relation.txt\
#   --output_folder data/swow_2rel_freq1/


#generate id for training
# python transe_embeddings/utils/generate_openke_data.py --train_path data/cpnet17rel_swow2rel/conceptnet.en.csv\
  # --concept_path data/cpnet17rel_swow2rel/concept.txt\
  # --relation_path data/cpnet17rel_swow2rel/relation.txt\
  # --output_folder data/cpnet17rel_swow2rel/
# 
# python3.6 transe_embeddings/embeddings/create_embeddings_glove.py transe_embeddings/paths.cfg cpnet17rel_swow2rel
# python transe_embeddings/embeddings/train_kge.py TransE SGD 1 data/cpnet17rel_swow2rel/ cpnet17rel_swow2rel --epoch 3
# python transe_embeddings/embeddings/kge_to_npy.py TransE SGD cpnet17rel_swow2rel 


# python3.6 embeddings/create_embeddings_glove.py paths.cfg swow_2rel_freq1
# python embeddings/train_kge.py TransE SGD 1 data/swow_2rel_freq1/  swow_2rel_freq1 --epoch 1
# python embeddings/kge_to_npy.py TransE SGD swow_2rel_freq1 



python transe_embeddings/utils/generate_openke_data.py --train_path data/cpnet/conceptnet.en.csv\
  --concept_path data/cpnet/concept.txt\
  --relation_path data/cpnet/relation.txt\
  --output_folder data/cpnet/

# python3.6 transe_embeddings/embeddings/create_embeddings_glove.py transe_embeddings/paths.cfg cpnet
# python transe_embeddings/embeddings/train_kge.py TransE SGD 1 data/cpnet/ cpnet --epoch 3
# python transe_embeddings/embeddings/kge_to_npy.py TransE SGD cpnet 
