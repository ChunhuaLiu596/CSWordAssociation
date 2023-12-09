import pandas as pd
import networkx as nx 
from pyvis.network import Network


path = 'output/dev_data_50.csv.en.csv'
df = pd.read_csv(path, names=['rel', 'head', 'tail', 'weight'], sep='\t')


G = nx.from_pandas_edgelist(df,
        source='head',
        target = 'tail',
        edge_attr = 'rel'
        )

net = Network(notebook=True)
net.from_nx(G)
net.show('example.html')