import json
import configparser
import numpy as np
import pathlib


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

config = configparser.ConfigParser()
config.read("paths.cfg")

OUTPUT_PATH = f"data/{kg_name}/embs/glove_initialized/"
transe_res = OUTPUT_PATH + f"glove.{model}."+opt_method+".vec.json"

#transe_res = config[f"paths_{kg_name}"]["transe_res"]

'''remove ".vec.json" from filename'''
#output_name = ".".join(transe_res.split('.')[: -2]).lower()
output_name = ".".join(transe_res.split('.')[: -2])

#output_name = output_name.replace("SGD","sgd")

ent_embeddings_file = output_name + '.ent.npy'
rel_embeddings_file = output_name + '.rel.npy'

print("Loading {} ...".format(transe_res))
#pathlib.Path(transe_res).mkdir(parents=True, exist_ok=True)
with open(transe_res, "r") as f:
    dic = json.load(f)

ent_embs, rel_embs = dic['ent_embeddings'], dic['rel_embeddings']

ent_embs = np.array(ent_embs, dtype="float32")
rel_embs = np.array(rel_embs, dtype="float32")

np.save(ent_embeddings_file, ent_embs)
np.save(rel_embeddings_file, rel_embs)

print("Save {}, containing {} entity embeddings".format(ent_embeddings_file, ent_embs.shape))
print("Save {}, containing {} relation embeddings".format(rel_embeddings_file, rel_embs.shape))
