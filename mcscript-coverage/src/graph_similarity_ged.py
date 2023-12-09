# %%
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


# %%

GC = build_graph_from_csv('../data/conceptnet.en.csv')
GS = build_graph_from_csv('../data/swow.en.csv')


# %%
from utils_graph import GraphEditDistance

# GED = GraphEditDistance(e_del_cost =1, e_ins_cost=0, n_del_cost=1, n_ins_cost=0)
GED = GraphEditDistance(e_del_cost =1, e_ins_cost=1, n_del_cost=1, n_ins_cost=1)

def subgraph_info(triples):
    if len(triples) == 0:
        avg_deg, avg_density= 0, 0   

    G = build_graph_from_triples(triples)
   
    G1 = GC.subgraph(G.nodes)
    G2 = GS.subgraph(G.nodes)

    ged1 = GED.get_graph_edit_distance(G, G1)
    ged2 = GED.get_graph_edit_distance(G, G2)
    ged3 = GED.get_graph_edit_distance(G1, G2)
    
    avg_deg = average_degree(G)
    density = nx.density(G)

    avg_deg1 = average_degree(G1)
    density1 = nx.density(G1)

    avg_deg2 = average_degree(G2)
    density2 = nx.density(G2)

    return pd.Series([avg_deg,density, avg_deg1, density1, avg_deg2, density2, ged1, ged2, ged3])

def mean_std_statistics(df, output_path=None):
    df_mean = df[['avg_deg_mcscript2','density_mcscript2','avg_deg_conceptnet','density_conceptnet','avg_deg_swow','density_swow', 'ged1', 'ged2', 'ged3']].mean()
    df_std = df[['avg_deg_mcscript2','density_mcscript2','avg_deg_conceptnet','density_conceptnet','avg_deg_swow','density_swow', 'ged1', 'ged2', 'ged3']].std()

    df_out = pd.DataFrame([df_mean, df_std], index=['mean', 'std'])
    print(df_out)

    if output_path is not None:
        df_out.to_csv(output_path)
        print(f"statistics: {output_path}")


# %%
# path = '../data/S_RW.R123.csv'
# path='../data/S_PPMI.R123.csv'

# path='../data/S_strength.R123.csv'
query_path = '../output/mcscript2.csv_frame_lemma.csv'
output_path = '../output/mcscript2.csv_graph.csv'    

from datetime import datetime
t1 = datetime.now() 
paths = ['../data/swcn_S_strength.R1.csv',
        '../data/swcn_S_PPMI.R1.csv',
        '../data/cnsw_S_strength.R1.csv',
        '../data/cnsw_S_PPMI.R1.csv'
        ]
# paths = ['../data/cnsw_S_PPMI.R1.csv']

for path in paths:
    simM = load_similarity_matrix(path)
    df = graph_similarity(query_path, simM, output_path)
    # df[['simPPMI']].plot(kind="hist", alpha=0.5)
    print(df['simPPMI'].describe())
    print(df.info(verbose=True))
    print("Cost time: {}".format(datetime.now() - t1))

# %%
