import pandas as pd 
import networkx as nx 
from tqdm import tqdm 
tqdm.pandas()
import functools, operator

########## Graph Construction #########

def build_graph_from_df(df, type='undirected'):
    graph_map = {
        "Undirected": nx.Graph(),
        "MultiGraph": nx.MultiGraph(),
        "MultiDiGraph": nx.MultiDiGraph()
    }
    G = nx.from_pandas_edgelist(df,
        source='head',
        target = 'tail',
        edge_attr = 'rel',
        # edge_attr = ['rel','weight'] if 'weight' in df.columns else 'rel',
        create_using = graph_map.get(type, nx.Graph())
        # create_using = nx.MultiGraph()
        )
    G.remove_nodes_from(list(nx.isolates(G)))
    return G 

def build_graph_from_csv(path, type):
    '''
     create_using = nx.MultiGraph() because  multiple relations exists between two nodes in ConceptNet,and two nodes are bi-directional connected in SWOW ??
     
    note: removing the self-loop (h,t) pairs from df before creating a graph
    '''
    df = pd.read_csv(path, names=['rel', 'head', 'tail', 'weight'], sep='\t')
    # df['head'] = df['head'].progress_apply(lambda x: x.replace("_", " "))
    # df['tail'] = df['tail'].progress_apply(lambda x: x.replace("_", " "))
    G = build_graph_from_df(df, type)

    return G

def build_graph_from_triples(triples, type):
    '''
    triples: a set/list of triples in the format of (r,h,t)
    type: graph type 
    '''
    df = pd.DataFrame(triples, columns=['rel','head','tail'])
    G = build_graph_from_df(df, type)
    return G

def construct_graph_from_shortest_path(S, T):
    '''
    S: sourge graph 
    T: targe graph 
    
    1) For all direct connected pairs in S, retrieve their shortest path from T
    2) Using these shortest paths to construct the new graph G 
    return:
         G: a sub-graph of T, nodes in G are from S  
    '''
    G = nx.Graph()
    count = 0
    for (u, v, rel) in S.edges.data(data=True):
        if u in T.nodes and v in T.nodes and nx.algorithms.shortest_paths.generic.has_path(T, u, v):
            path = nx.shortest_path(T, source=u, target=v)
            nx.add_path(G, path)    
            count+=1 
    recall_edge = count/S.size()
    return G, recall_edge 

def retrieve_subgraph(path, G, type):
    G1 = build_graph_from_csv(path, type)
    S1 = G1.subgraph(G.nodes)
    return S1


########## Statistics #########
def average_degree(G, directed=False):
    N, K = G.order(), G.size() #Number of nodes and edges
    if N==0:
        return 0

    if directed:
        return  float(K)/N
    else:
        return 2*float(K)/N 

def print_statistics(G):
    density = nx.density(G)
    avg_deg = average_degree(G)
    print(f"avg_degree: {avg_deg}, density: {density} ")
    print(nx.info(G))

def graph_statistics(path, N):
    G = build_graph_from_csv(path, N)
    print_statistics(G)
    return G 


def save_plot(net, fig_path=None):
    if fig_path is not None:
        net.save_graph(fig_path)

def mean_std_statistics(df, output_path=None):
    df_mean = df[['avg_deg_mcscript2', 'avg_deg_conceptnet','avg_deg_swow','density_mcscript2','density_conceptnet','density_swow', 'ged_cn', 'ged_sw', 'ged3']].mean().round(2)
    df_std = df[['avg_deg_mcscript2', 'avg_deg_conceptnet','avg_deg_swow','density_mcscript2','density_conceptnet','density_swow', 'ged_cn', 'ged_sw', 'ged3']].std().round(2)

    df = pd.concat([df_mean.T, df_std.T], axis=1)
    df.columns = ['mean', 'std']
    df['mean'] = df['mean'].apply(str).str.replace('.', ',')
    df['std'] = df['std'].apply(str).str.replace('.', ',')
   
    df["mean±std"] = df['mean'] +"±"+ df['std']
    df["mean±std"] = df["mean±std"].apply(lambda x: x.replace(",", "."))
    print( df["mean±std"])

    if output_path is not None:
        df.to_csv(output_path)
        print(f"statistics: {output_path}")

#### For nc.graph_edit_distance 
class GraphEditDistance(object):
    def __init__(self, e_del_cost =1, e_ins_cost=0, n_del_cost=1, n_ins_cost=0):
       self.e_del_cost = e_del_cost 
       self.e_ins_cost = e_ins_cost 
       self.n_del_cost = n_del_cost 
       self.n_ins_cost = n_ins_cost 

    def node_subst_cost(self, node1, node2):
        # check if the nodes are equal, if yes then apply no cost, else apply 3
        if node1['label'] == node2['label']:
            return 0
        return 3

    def node_del_cost(self, node):
        return self.n_del_cost  # here you apply the cost for node deletion

    def node_ins_cost(self, node):
        return self.n_ins_cost  # here you apply the cost for node insertion

    def edge_subst_cost(self, edge1, edge2):
        # check if the nodes are equal, if yes then apply no cost, else apply 3
        # if node1['label'] == node2['label']:
        #     return 0
        return 1

    def edge_del_cost(self, edge, ):
        return self.e_del_cost  

    def edge_ins_cost(self, edge):
        return self.e_ins_cost   
    
    def get_graph_edit_distance(self, G, H, normalize_nodes=True, normalize_edges=False):
        '''
        normalize: if True, normalize the GED with the size of  G 
        '''
        ged= nx.optimize_graph_edit_distance(
            G,
            H,
            node_ins_cost=self.node_del_cost,
            edge_del_cost=self.edge_del_cost,
            edge_ins_cost=self.edge_ins_cost,
        )
        ged = next(ged)
        if normalize_nodes:
            ged = ged/G.order() 
        elif normalize_edges:
            ged = ged/G.size() 
        return  ged  


def read_scripts_triples(path):
    '''
    read mcscripts triples to DataFrame
    remova invalid ones 
    '''
    df = pd.read_csv(path)
    df['triple_num'] = df['triple_lemma'].progress_apply(lambda x: len(eval(x)))
    N = len(df.index)

    df = df.query('triple_num>0')

    print(f"\nremoving invalid scripts {N - len(df.index)} \n remaining {len(df.index)} scripts")
    return df 


def flat_list(a):
    return functools.reduce(operator.iconcat, a, [])
