# %%
import sys
import re
from tqdm import tqdm
import json
import spacy
import networkx as nx
from collections import Counter

def is_negated(node, pattern="seeds"):
    neg_list = ('no', 'not', 'none', 'nor', 'no_one','nobody', 'nothing', 'neither', 'nowhere', 'never', 'hardly', 'barely', 'scarcely', 'non', 'without', 'fail', 'cannot', 'cant', 'nolonger', 'dont', 'wont' )
    # 'no one'
    # neg_list =['except', 'prevent', 'neglected', 'refused', 'absence', 'without', 'fail', 'nor', 'n\'t']

    # neg_dict = {'prefixes': ['in', 'un', 'im', 'dis', 'ir'], 'infixes': ['less'], 'suffixes': ['less']}

    # node = node.replace("_", " ")
    if pattern=="seeds":
        return bool(node in neg_list)
    else:
        neg_flag=False
        for n in node.split("_"):
            if n in neg_list:
                neg_flag=True
                break
        return neg_flag

    # else:
        # pattern = re.compile(r'(?:{})_(?:{}) ' % '|'.join(neg_list))
        # node = node.replace("_", " ")
        # pattern = re.compile(r'*({} )*' .format('|'.join(neg_list)))
        # pattern = re.compile(r'( .?{0}(_|$))'.format('|'.join(neg_list)))
        # return  bool(re.match(pattern, node)) 

nlp = spacy.load('en_core_web_sm')

def is_negated_spacy(node, pattern="seeds"):
    node = node.replace("_", " ")
    neg_flag=False

    doc = nlp(node)
    for tok in doc:
        negation_tokens = [tok for tok in doc if tok.dep_ == 'neg']
        if len(negation_tokens)>0:
            negation_head_tokens = [token.head for token in negation_tokens]
            
            # for token in negation_head_tokens:
                # print(token.text, token.dep_, token.head.text, token.head.pos_, [child for child in token.children])
            neg_flag=True
    return neg_flag


def load_negated_nodes(path, outpath, debug, pattern="seeds"):
    negated_words = set()
    with open(path, 'r') as fo, open(outpath, 'w') as fout:
        for i, line in enumerate(tqdm(fo.readlines())):
            line = line.strip()
            if is_negated(line, pattern):
            # if is_negated_spacy(line):
                # print(line)
                negated_words.add(line)
                fout.write(line+"\n")
            if debug =='True' and i>10000: break

    print("write {}".format(outpath))
    return negated_words


# def get_cpnet_simple(nx_graph):
#     cpnet_simple = nx.Graph()
#     for u, v, data in nx_graph.edges(data=True):
#         w = data['weight'] if 'weight' in data else 1.0
#         if cpnet_simple.has_edge(u, v):
#             cpnet_simple[u][v]['weight'] += w
#         else:
#             cpnet_simple.add_edge(u, v, weight=w)
#     return cpnet_simple

def load_negated_pairs(negation_nodes, path, outpath, kg_name):
    negated_pairs=[]
    # G=nx.MultiGraph()
    with open(path, "r") as fo, open(outpath, 'w') as fout:
        for line in tqdm(fo.readlines()):
            rel, subj, obj, weight = line.strip().split("\t")
            if kg_name =='swow' and rel =='bidirectionalassociated':
                continue
            if subj in negation_nodes or obj in negation_nodes:
                negated_pairs.append(line)
                fout.write(line+"\n")
    print("write {}".format(outpath))
    return negated_pairs



# %%
def load_graph(path):
    graph = nx.MultiDiGraph()
    with open(path, "r") as fo:
        for line in tqdm(fo.readlines()):
            rel, subj, obj, weight = line.strip().split("\t")
            if graph.has_edge(subj, obj):
                if rel not in graph[subj][obj]:
                    # print(rel, subj, obj)
                    graph.add_edge(subj, obj, key=rel, rel=rel, weight=weight)
            else:
                graph.add_edge(subj, obj, key=rel, rel=rel,weight=weight)
            # graph.add_edge(subj, obj, rel=rel, weight=weight)
            # graph.add_edge(obj, subj, rel="_"+rel, weight=weight)
    return graph

# %%
def detect_neg():
   for data_dir,kg_name in zip(data_dirs, kg_names):
        path_node = data_dir +'concept.txt'
        path_triple = data_dir + 'conceptnet.en.csv'
        out_node = data_dir + 'negated_nodes.json'
        out_triple = data_dir + 'negated_triples.json'
        negated_nodes = load_negated_nodes(path_node, out_node, debug, pattern)
        negated_pairs = load_negated_pairs(negated_nodes, path_triple, out_triple, kg_name)
    
        print("{}: neg  {} nodes appearning in {} triples".format(kg_name, len(negated_nodes), len(negated_pairs)))

# %%
debug = sys.argv[1]
pattern=sys.argv[2]
data_dirs=[ 'swow/','cpnet47rel/']
kg_names = [ 'swow', 'cpnet']
graph_sw =  load_graph(data_dirs[0]+  'conceptnet.en.csv')
for k,v in graph_sw['thirsty'].items():
    if len(k.split("_"))>1:
        print(k,v['forwardassociated']['rel'])

# graph = load_graph(data_dirs[1]+  'conceptnet.en.csv')
# # %%
# def rel_distribution(path):
#     rel_count = Counter()
#     with open(path, "r") as fo:
#         for line in tqdm(fo.readlines()):
#             rel, subj, obj, weight = line.strip().split("\t")
#             rel_count[rel] +=1
#     return rel_count.most_common()

# %%
def get_negpairs_rel(graph, swow_neg_path, outpath):
    rel_seen = Counter()
    with  open(swow_neg_path) as fo, open(outpath, 'w') as fout:
        for line in tqdm(fo.readlines()):
            rel, subj, obj, weight = line.strip().split("\t")
            if graph.has_edge(subj, obj):
                edge_data = graph[subj][obj]
                for idx, data  in edge_data.items():
                    rel_cn = data['rel']
                    weight_cn = data['weight']
                    rel_seen[rel_cn] +=1
                    # print(rel_cn)
                    fout.write("{}\t{}\t{}\t{}\n".format(rel_cn, subj, obj, weight_cn))
            # elif graph.has_edge(obj, subj):
            #     edge_data = graph[obj][subj]
            #     for idx, data  in edge_data.items():                        
            #         rel_cn = data['rel']
            #         weight_cn = data['weight']
            #         rel_seen[rel_cn] +=1
            #         # print(rel_cn)
            #         fout.write("{}\t{}\t{}\t{}\n".format(rel_cn, obj, subj, weight_cn))
    print()
    all = 0
    for x in rel_seen.most_common():
        print("{} \t {}".format(x[0], x[1]))
        all +=x[1]
    print("total {} neg sw triples recalled from CN".format(all))

# %%
get_negpairs_rel(graph, data_dirs[0]+"negated_triples.json", data_dirs[0] + "negated_triples_cn_rel.json")
cn=rel_distribution(data_dirs[1]+"negated_triples.json")
all = 0
for x in cn:
    print("{} \t {}".format(x[0], x[1]))
    all +=x[1]
print("total {} neg cn triples".format(all))

# %%
