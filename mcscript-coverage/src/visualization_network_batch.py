# %%
import pandas as pd
import networkx as nx 
from pyvis.network import Network
import math
import random
from utils_graph import *

# %%
# total_N = len(df.index)

print(sample_seeds)
# %%

# path = '../output/dev_data_50.csv.en.csv'
# G = graph_statistics(path)

path = '../output/mcscript2.csv_frame_lemma.csv'
# df = pd.read_csv(path) 
df = read_scripts_triples(path)

total_N = len(df.index)
sample_N = 10 
sample_seeds = random.sample([i for i in range(total_N)],k=sample_N)

show_directed = False

for N in sample_seeds: 

def generate_mcscript_graph():
    
	fig_path = f'../output/figure/{N}_mc.html'

	G = build_graph_from_triples(eval(df.iloc[N]['triple_lemma']), type='undirected')

	# nx.draw_networkx(G)

	# print(G.nodes)
	net = Network(notebook=True, height='1000px', width='700px', directed=show_directed)
	net.from_nx(G)
	net.show_buttons(filter_=['physics'])

	net.show(fig_path)
# print(nx.adjacency_matrix(G))
# save_plot(net, fig_path)
# print_statistics(G)
# net.show(fig_path)

# plot_save_subgraph(net, '../output/figure/mc.html')
# net = Network(notebook=True)
# net.from_nx(G)
# net.show('output/example.html')

# %%
    path = '../data/conceptnet.en.csv'
    fig_path=f'../output/figure/{N}_cn_path.html'
    # S1=retrieve_subgraph(path, G)

    T = build_graph_from_csv(path, type='undirected')

    S1 = T.subgraph(G.nodes)
    S2 = construct_graph_from_shortest_path(S=G, T=T)

    print_statistics(S1)
    net = Network(notebook=True, height='1000px', width='700px', directed=show_directed)
    net.from_nx(S1)
    net.show_buttons(filter_=['physics'])
    save_plot(net, fig_path)
    # save_plot(net, fig_path)
    net.show(fig_path)

    print_statistics(S2)
    net = Network(notebook=True, height='1000px', width='700px', directed=show_directed)
    net.from_nx(S2)
    net.show_buttons(filter_=['physics'])
    save_plot(net, fig_path)
    # save_plot(net, fig_path)
    net.show(fig_path)

    nx.draw(S2, with_labels=True)



# %%
    path = '../data/swow.en.csv'
    fig_path=f'../output/figure/{N}_sw_path.html'

    T = build_graph_from_csv(path, type='undirected')

    S1 = T.subgraph(G.nodes)
    S2 = construct_graph_from_shortest_path(S=G, T=T)

    print_statistics(S1)
    net = Network(notebook=True, height='1000px', width='700px', directed=show_directed)
    net.from_nx(S1)
    net.show_buttons(filter_=['physics'])
    save_plot(net, fig_path)
    # save_plot(net, fig_path)
    net.show(fig_path)

    print_statistics(S2)
    net = Network(notebook=True, height='1000px', width='700px', directed=show_directed)
    net.from_nx(S2)
    net.show_buttons(filter_=['physics'])
    save_plot(net, fig_path)
    # save_plot(net, fig_path)
    net.show(fig_path)

    nx.draw(S2)
    # save_plot(net, fig_path)
    net.show(fig_path)

# %%
# g = Network()
# g.add_node(0)
# g.add_node(1)
# g.add_edge(0, 1, label="this is an edge label")
# net = Network(notebook=True, height='1000px', width='700px', directed=True)
# net.from_nx(g)
# net.show('../output/example3.html')

# %%

# 2.0,0.09090909090909091,
# 0.4,0.044444444444444446,
# 0.2222222222222222,0.027777777777777776


# 1.8125 0.05846774193548387


# %%
# from pyvis.network import Network

# g = Network()
# g.add_node(0)
# g.add_node(1)
# g.add_edge(0, 1)
# g.show("basic.html")

# %%
# g.show("basic.html")