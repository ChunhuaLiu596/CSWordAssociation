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
from statistics import mean 
from pyvis.network import Network
# %%
GC = build_graph_from_csv('../data/conceptnet.en.csv')
GS = build_graph_from_csv('../data/swow.en.csv')


# %%
path = '../output/mcscript2.csv_frame_lemma.csv'
output_path = '../output/mcscript2.csv_graph.csv'
df = pd.read_csv(path)

# %%
def neighborhood(G, nodes, K=1):
    '''
    G: a target graph 
    nodes: a list of node
    returns:
        a list:  K-hop of neighbours for nodes 
    '''
    def neighborhood_single_node(node):
        path_lengths = nx.single_source_dijkstra_path_length(G, node, cutoff=K)
        return [node for node, length in path_lengths.items()]

    neighbors =  map(neighborhood_single_node, nodes) 
    return list(neighbors)[0]


def subgraph_info(triples, T, K=1):
    '''
    triples: from MCScript, used to construct source graph 
    T : a target graph (i.e., CN/SW)
    K: the hop of neighbours for each node in MCScript graph 
    '''

    if len(triples) == 0:
        avg_centrality =  0   

    G = build_graph_from_triples(triples)
    G1 = T.subgraph(G.nodes)

    G1_neighbours = neighborhood(T, G1.nodes, K)

    print("original graph nodes")
    print("0-hop graph. Size: {} Order:{}".format(G1.size(), G1.order()))
    print(G1.nodes)
    # G1_neighbours = []
    # for node in G1.nodes:
    #     G1_neighbours.extend(neighborhood(T, node,K))
     
    
    print("one-hop neighbours")
    print(G1_neighbours)
    
    new_nodes = set(G.nodes).union(set(G1_neighbours))
    G1_k_hop = T.subgraph( new_nodes )
    
    # G1_k_hop = T.subgraph(G1_k_hop)
    print("1-hop graph. Size: {} Order:{}".format(G1_k_hop.size(), G1_k_hop.order()))

    net = Network(notebook=True, height='1000px', width='700px', directed=False)
    net.from_nx(G1_k_hop)
    net.show_buttons(filter_=['physics'])
    fig_path='../log/example.html'
    net.show(fig_path)
    # print(G1_neighbours)
    # print(G1_k_hop.nodes)

    # centrality = nx.katz_centrality(G1_k_hop)
    # for n, c in sorted(centrality.items()):
    #     print(f"{n} {c:.2f}")
    # mean_centrality = mean(centrality.values())
    # print(mean_centrality)
    # return mean_centrality
    return 0 
    

# %%
df = df.head(1)
df['triple_lemma'].progress_apply(lambda x: subgraph_info(eval(x), GS))
# G = build_graph_from_triples(triples)

# %%
