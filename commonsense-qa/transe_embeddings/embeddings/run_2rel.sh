
#python create_embeddings_glove.py cpnet
#python TransE.py SGD 1 ./openke_data/cpnet/  cpnet
#python TransE_to_npy.py cpnet

#python create_embeddings_glove.py swow_2relations
#python TransE.py SGD 1 ./openke_data/swow/2relations/  swow_2relations
#python TransE_to_npy.py swow_2relations
#

#python create_embeddings_glove.py cpnet_swow
#python TransE.py SGD 1 ./openke_data/cpnet_swow/ cpnet_swow
#python TransE_to_npy.py cpnet_swow


#python create_embeddings_glove.py swow_19relations
#python TransE.py SGD 1 ./openke_data/swow/19relations/  swow_19relations
#python TransE_to_npy.py swow_19relations


#python create_embeddings_glove.py swow_2relationsAugmented
#python TransE.py SGD 1 ./openke_data/swow/2relationsAugmented/  swow_2relationsAugmented
#python TransE_to_npy.py swow_2relationsAugmented

#python create_embeddings_glove.py swow_19relationsAugmented
#python TransE.py SGD 1 ./openke_data/swow/19relationsAugmented/  swow_19relationsAugmented
#python TransE_to_npy.py swow_19relationsAugmented


#2020-8-27
#python create_embeddings_glove.py swow_2rel_freq1
#python TransE.py SGD 1 ./openke_data/swow/2rel_freq1/  swow_2rel_freq1
#python TransE_to_npy.py swow_2rel_freq1
#
python create_embeddings_glove.py swow_3rel_freq1
python TransE.py SGD 1 ./openke_data/swow/3rel_freq1/  swow_3rel_freq1 --epoch 3
python TransE_to_npy.py swow_3rel_freq1
#
#
#python create_embeddings_glove.py swow_2rel_freq2
#python TransE.py SGD 1 ./openke_data/swow/2rel_freq2/  swow_2rel_freq2
#python TransE_to_npy.py swow_2rel_freq2
#
#python create_embeddings_glove.py swow_3rel_freq2
#python TransE.py SGD 1 ./openke_data/swow/3rel_freq2/  swow_3rel_freq2
#python TransE_to_npy.py swow_3rel_freq2


### cpnet+

#python3.6 create_embeddings_glove.py swow_3rel_freq1_cpnet
#python3.6 TransE.py SGD 1 ./openke_data/swow/3rel_freq1_cpnet/  swow_3rel_freq1_cpnet --epoch 3
#python3.6 TransE_to_npy.py swow_3rel_freq1_cpnet

#python create_embeddings_glove.py swow_2rel_freq2_cpnet
#python TransE.py SGD 1 ./openke_data/swow/2rel_freq2_cpnet/  swow_2rel_freq2_cpnet
#python3.6 TransE_to_npy.py swow_2rel_freq2_cpnet

#swow_2rel_freq2_cpnetpython3.6 create_embeddings_glove.py swow_2rel_freq1_cpnet
#swow_2rel_freq2_cpnetpython3.6 TransE.py SGD 1 ./openke_data/swow/2rel_freq1_cpnet/  swow_2rel_freq1_cpnet
#swow_2rel_freq2_cpnetpython3.6 TransE_to_npy.py swow_2rel_freq1_cpnet
#python3.6 create_embeddings_glove.py swow_3rel_freq2_cpnet
#python3.6 TransE.py SGD 1 ./openke_data/swow/3rel_freq2_cpnet/  swow_3rel_freq2_cpnet
#python3.6 TransE_to_npy.py swow_3rel_freq2_cpnet
#


