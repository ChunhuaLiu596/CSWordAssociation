# %%
#################
# read similarity based on graph
#################

#################
# read similarity based on graph
#################



def retrieve_similarity(query, simM):
    # print(query)
    scores = []
    for triple in query:
        (rel, h, t) = triple 
        if h in simM.index and t in simM.index:
            score = simM[h][t]
        else:
            score= 0 
        scores.append(score)
    
    return mean(scores) if len(scores)>0 else 0
    

def graph_similarity(path, simM, output_path=None):
    df = pd.read_csv(path)
    df['simPPMI']= df['triple_lemma'].progress_apply(lambda x: retrieve_similarity(eval(x), simM))
    # print(df.head())
    return df 

    

# %%

def load_similarity_matrix(path='../data/S_PPMI.R123.csv'):
    # path = '../data/S_PPMI.R123.csv'
    # chunksize = 1000
    # df = pd.concat( pd.read_csv(path, chunksize=chunksize)) 
    df = pd.read_csv(path, header =0, index_col = 0) # Note: set first column and row as index

    # df = []
    # with open(path, 'r') as fin:
    #     data = csv.DictReader(fin)
    #     # print(data[1])
    #     for i, x in enumerate(data):
    #         print(x) 
    #         if i>2 and i <3: break 

    # data = csv.DictReader(open(path))
    # for row in tqdm(data, desc='loading sim'):
    #     df.append(df)
    # df = pd.concat(df)

    # df = pd.DataFrame(data=data)
    # df.columns = df.iloc[1]
    # df.index_col = df.iloc[:1]

    return df 
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

t1 = datetime.now()
print("Start time: {}".format(t1))

path = '../data/cnsw_S_strength.R1.csv'
df = load_similarity_matrix()
print("Cost time {}".format(datetime.now() - t1))





# path = '../data/S_RW.R123.csv'
# path='../data/S_PPMI.R123.csv'

# path='../data/S_strength.R123.csv'
query_path = '../output/mcscript2.csv_frame_lemma.csv'
output_path = '../output/mcscript2.csv_graph.csv'    

from datetime import datetime
t1 = datetime.now() 
paths = ['../data/cnsw_S_strength.R1.csv']
# paths = ['../data/cnsw_S_PPMI.R1.csv']

for path in paths:
    simM = load_similarity_matrix(path)
    print(simM.head())
    # df = graph_similarity(query_path, simM, output_path)
    # df[['simPPMI']].plot(kind="hist", alpha=0.5)
    # print(df['simPPMI'].describe())
print("time: {}".format(datetime.now() - t1))



import os
from datetime import datetime 
import csv

print(os.getcwd())
import numpy as np 

t1 = datetime.now()
print("Start time: {}".format(t1))

path = '../data/cnsw_S_strength.R1.csv'
# chunks = []
# for chunk in pd.read_csv(path, chunksize=chunksize):
    # chunks.append(chunk)

pd_df = pd.concat(chunk)

print("Cost time {}".format(datetime.now() - t1))




from dask import dataframe as dd
import time 
import csv 
from tqdm import tqdm 
start = time.time()
path = '../data/cnsw_S_strength.R1.csv'
print("Start time: {}".format(start))

data = csv.DictReader(open(path))

end = time.time()
print("Read csv with dask: ",(end-start),"sec")


print(simM.head(1))
# df.iloc[]
# simM.set_index([simM.iloc[0][1:], simM.columns[0][1:]])
s = simM.loc['a', 'abandon']
print(s)