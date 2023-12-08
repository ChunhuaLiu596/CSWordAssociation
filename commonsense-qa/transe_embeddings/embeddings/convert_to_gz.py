# Convert TransH from binary weights to Word2Vec format

from pathlib import Path
# import torch
import pandas as pd
import argparse
import numpy as	np

import json
import configparser
import numpy as np
import pathlib
# manually install tahbles via `pip install tables` if python complains
# about missing tables module:
# `HDFStore requires PyTables, "No module named 'tables'" problem importing`

'''
input file: direct output of TransE (json file)

output file: two npy files storing the embedding matrix of entities and relations respectively,
             the file does not deal with the names of entities and relations.
'''
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('model', default='TransE', help='model used for train a KG')
parser.add_argument('opt_method', help='SGD/Adagrad/...')
parser.add_argument('kg_name', default='conceptnet', help='specify the knowledg graph')
args = parser.parse_args()

model = args.model 
opt_method = args.opt_method
kg_name = args.kg_name

OUTPUT_PATH = f"data/{kg_name}/embs/glove_initialized/"

transe_res = OUTPUT_PATH + f"glove.{model}."+opt_method+".vec.json"

print("Loading {} ...".format(transe_res))
with open(transe_res, "r") as f:
    dic = json.load(f)
ent_embs, rel_embs = dic['ent_embeddings'], dic['rel_embeddings']

ent_embs = np.array(ent_embs, dtype="float32")

entity2id_path = f"data/{kg_name}/entity2id.txt"
with open(entity2id_path) as f:
    next(f)  # skip header
    entity2id = [word_pair.strip().split()[0] for word_pair in f.readlines()]

# use pandas to insert index
out = pd.DataFrame(ent_embs, index=entity2id)

# save the processed file
output_file  = OUTPUT_PATH + f"glove.{model}."+opt_method+".txt.gz"
out.to_csv(output_file, sep=' ', header=False, compression="gzip")
print("Save {}".format(output_file))


# OUTPUT_PATH = f"data/{kg_name}/embs/glove_initialized/"
# transe_res = OUTPUT_PATH + f"glove.{model}."+opt_method+".pt"

# '''remove ".vec.json" from filename'''
# output_name = ".".join(transe_res.split('.')[: -2]) + '.txt.gz'

# # load pre-trained trans*.pt model
# trans = torch.load(transe_res, map_location='cpu')

# entity2id_path = f"data/{kg_name}/entity2id.txt"
# with open(entity2id_path) as f:
#     next(f)  # skip header
#     entity2id = [word_pair.strip().split()[0] for word_pair in f.readlines()]

# embed = trans.get("ent_embeddings.weight").numpy()
# # use pandas to insert index
# out = pd.DataFrame(embed, index=entity2id)

# # save the processed file
# output_file = args.output + '.txt.gz'
# out.to_csv(output_file, sep=' ', header=False, compression="gzip")
# print("Save {}".format(output_file))
