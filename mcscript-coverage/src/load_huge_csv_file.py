import pandas as pd 
import networkx as nx 
from tqdm import tqdm 
tqdm.pandas()
import matplotlib.pyplot as plt
from statistics import mean
from tqdm import tqdm 
from datetime import datetime 
import csv 
from utils_graph import *


def load_similarity_matrix(path='../data/S_PPMI.R123.csv'):
    # path = '../data/S_PPMI.R123.csv'
    # chunksize = 1000
    # df = pd.concat( pd.read_csv(path, chunksize=chunksize)) 
    # df = pd.read_csv(path, header =0, index_col = 0) # Note: set first column and row as index

    df = []
    with open(path, 'r') as fin:
        data = csv.DictReader(fin)
        # print(data[1])
        for i, x in enumerate(data):
            print(x) 
            if i>2 and i <3: break 

    # data = csv.DictReader(open(path))
    # for row in tqdm(data, desc='loading sim'):
    #     df.append(df)
    # df = pd.concat(df)

    # df = pd.DataFrame(data=data)
    # df.columns = df.iloc[1]
    # df.index_col = df.iloc[:1]

    return df 


path = '../data/cnsw_S_strength.R1.csv'

data = csv.DictReader(open(path))
# lines= len(list(data))
sim_data = []
cues = []
debug =True 
debug = False 
N=100
for i, x in enumerate(tqdm(data, desc='loading similarity')):
    # if i>=1:break
    if debug and i>=N: break
    nb, score = list(zip(*x.items()))

    # x = list(x.items())
    # cue = x[0][1]
    # nb, score = list(zip(*x[1:]))
    cue = score[0]
    if debug:
        score = score[1:N+1]
    else: 
        score = score[1:]
        
    # print(cue, score)
    cues.append(cue)
    sim_data.append(score)

# print(cues)
# print(sim_data)
df = pd.DataFrame(data= sim_data, columns=cues, index=cues)
print(df.head(5))
print(df['a']['aaa'])
    # for (y,v) in x:
    #     print(i, y,v, "||", len(y), len(v))