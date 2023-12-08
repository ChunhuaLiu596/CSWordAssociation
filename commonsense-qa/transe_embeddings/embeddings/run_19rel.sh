#
: << COMMENT
#Do this for the first time
cd ../conceptnet
wget https://s3.amazonaws.com/conceptnet/downloads/2018/edges/conceptnet-assertions-5.6.0.csv.gz
gzip -d conceptnet-assertions-5.6.0.csv.gz
python extract_cpnet.py

cd ../triple_string
python triple_string_generation.py

# get concept and relation embeddings with frequency and vocab files
# generate glove npy and glove_vocab
cd ../embeddings/
cd glove/
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
rm glove.*.zip
cd ../
python glove_to_npy.py

COMMENT

#python create_embeddings_glove.py swow_2relations
#python TransE.py SGD 1 ./openke_data/swow/2relations/  swow_2relations
#python TransE_to_npy.py swow_2relations



python create_embeddings_glove.py swow_19relations
python TransE.py SGD 1 ./openke_data/swow/19relations/  swow_19relations
python TransE_to_npy.py swow_19relations










