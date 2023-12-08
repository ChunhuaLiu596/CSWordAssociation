import os, sys
import networkx as nx
from networkx.utils import open_file
import nltk
import json
import math
from tqdm import tqdm
import numpy as np
import pickle
from utils import check_path

discard_relations=('relatedto','synonym', 'antonym', 'derivedfrom', 'formof', 'etymologicallyderivedfrom','etymologicallyrelatedto', 'language','capital', 'field', 'genre', 'genus', 'knownfor', 'leader', 'occupation', 'product', 'notdesires', 'nothasproperty','notcapableof')


relation_groups=[
    'relatedto',
    'synonym',
    'antonym',
    'derivedfrom',
    'formof',
    'isa',
    'hasproperty',
    'madeof',
    'partof',
    'definedas',
    'instanceof',
    'hasa',
    'createdby',
    'language',
    'capital',
    'field',
    'genre',
    'genus',
    'leader',
    'occupation',
    'product',
    'atlocation',
    'locatednear',
    'hascontext',
    'similarto',
    'symbolof',
    'knownfor',
    'hassubevent',
    'hasfirstsubevent',
    'haslastsubevent',
    'hasprerequisite',
    'entails',
    'mannerof',
    'causes',
    'causesdesire',
    'motivatedbygoal',
    'desires',
    'influencedby',
    'usedfor',
    'receivesaction',
    'capableof',
    'distinctfrom',
    'notdesires',
    'nothasproperty',
    'notcapableof',
    'etymologicallyderivedfrom',
    'etymologicallyrelatedto'
    ]

merged_relations = [
     'relatedto',
    'synonym',
    'antonym',
    'derivedfrom',
    'formof',
    'isa',
    'hasproperty',
    'madeof',
    'partof',
    'definedas',
    'instanceof',
    'hasa',
    'createdby',
    'language',
    'capital',
    'field',
    'genre',
    'genus',
    'leader',
    'occupation',
    'product',
    'atlocation',
    'locatednear',
    'hascontext',
    'similarto',
    'symbolof',
    'knownfor',
    'hassubevent',
    'hasfirstsubevent',
    'haslastsubevent',
    'hasprerequisite',
    'entails',
    'mannerof',
    'causes',
    'causesdesire',
    'motivatedbygoal',
    'desires',
    'influencedby',
    'usedfor',
    'receivesaction',
    'capableof',
    'distinctfrom',
    'notdesires',
    'nothasproperty',
    'notcapableof',
    'etymologicallyderivedfrom',
    'etymologicallyrelatedto'
]

###-----------
def load_vocab(data_dir):
    rel_path = os.path.join(data_dir, 'relation_vocab.pkl')
    ent_path = os.path.join(data_dir, 'entity_vocab.pkl')

    with open(rel_path, 'rb') as handle:
        rel_vocab = pickle.load(handle)

    with open(ent_path, 'rb') as handle:
        ent_vocab = pickle.load(handle)

    return rel_vocab['i2r'], rel_vocab['r2i'], ent_vocab['i2e'], ent_vocab['e2i']

def load_merge_relation():
    relation_mapping = dict()
    for line in relation_groups:
        ls = line.strip().split('/')
        rel = ls[0]
        for l in ls:
            if l.startswith("*"):
                relation_mapping[l[1:]] = "*" + rel
            else:
                relation_mapping[l] = rel
    return relation_mapping

def del_pos(s):
    """
    Deletes part-of-speech encoding from an entity string, if present.
    :param s: Entity string.
    :return: Entity string with part-of-speech encoding removed.
    """
    if s.endswith("/n") or s.endswith("/a") or s.endswith("/v") or s.endswith("/r"):
        s = s[:-2]
    return s


def extract_english(conceptnet_path, output_csv_path, output_vocab_path):
    """
    Reads original conceptnet csv file and extracts all English relations (head and tail are both English entities) into
    a new file, with the following format for each line: <relation> <head> <tail> <weight>.
    :return:
    """
    print('extracting English concepts and relations from ConceptNet...')
    relation_mapping=load_merge_relation()
    num_lines = sum(1 for line in open(conceptnet_path, 'r', encoding='utf-8'))
    cpnet_vocab = []
    concepts_seen = set()
    with open(conceptnet_path, 'r', encoding="utf8") as fin, \
            open(output_csv_path, 'w', encoding="utf8") as fout:
        for line in tqdm(fin, total=num_lines, desc='Extracting english'):
            toks = line.strip().split('\t')
            if toks[2].startswith('/c/en/') and toks[3].startswith('/c/en/'):
                """
                Some preprocessing:
                    - Remove part-of-speech encoding.
                    - Split("/")[-1] to trim the "/c/en/" and just get the entity name, convert all to 
                    - Lowercase for uniformity.
                """
                rel = toks[1].split("/")[-1].lower()
                head = del_pos(toks[2]).split("/")[-1].lower()
                tail = del_pos(toks[3]).split("/")[-1].lower()

                if not head.replace("_", "").replace("-", "").isalpha():
                    continue
                if not tail.replace("_", "").replace("-", "").isalpha():
                    continue
                if rel not in relation_mapping:
                    continue

                rel = relation_mapping[rel]
                if head == tail:  # delete self loops 44,664 
                    continue

                if rel.startswith("*"):
                    head, tail, rel = tail, head, rel[1:]

                data = json.loads(toks[4])

                fout.write('\t'.join([rel, head, tail, str(data["weight"])]) + '\n')

                for w in [head, tail]:
                    if w not in concepts_seen:
                        concepts_seen.add(w)
                        cpnet_vocab.append(w)

    with open(output_vocab_path, 'w') as fout:
        for word in cpnet_vocab:
            fout.write(word + '\n')

    print(f'extracted ConceptNet csv file saved to {output_csv_path}')
    print(f'extracted concept vocabulary saved to {output_vocab_path}')
    print()



def extract_english_all_relations(conceptnet_path):
    """
    Reads original conceptnet csv file and extracts all English relations (head and tail are both English entities) into
    a new file, with the following format for each line: <relation> <head> <tail> <weight>.
    :return:
    """
    print('extracting English concepts and relations from ConceptNet...')
    relation_mapping={}
    num_lines = sum(1 for line in open(conceptnet_path, 'r', encoding='utf-8'))
    cpnet_vocab = []
    concepts_seen = set()
    with open(conceptnet_path, 'r', encoding="utf8") as fin:
        for line in tqdm(fin, total=num_lines):
            toks = line.strip().split('\t')
            if toks[2].startswith('/c/en/') and toks[3].startswith('/c/en/'):
                """
                Some preprocessing:
                    - Remove part-of-speech encoding.
                    - Split("/")[-1] to trim the "/c/en/" and just get the entity name, convert all to 
                    - Lowercase for uniformity.
                """
                rel = toks[1].split("/")[-1].lower()
                head = del_pos(toks[2]).split("/")[-1].lower()
                tail = del_pos(toks[3]).split("/")[-1].lower()

                if not head.replace("_", "").replace("-", "").isalpha():
                    continue
                if not tail.replace("_", "").replace("-", "").isalpha():
                    continue

                if head == tail:
                    continue 

                if rel not in relation_mapping:
                    relation_mapping[rel]=rel

                data = json.loads(toks[4])

                for w in [head, tail]:
                    if w not in concepts_seen:
                        concepts_seen.add(w)
                        cpnet_vocab.append(w)

    print("relation types: {}".format(len(relation_mapping.keys())))
    print(relation_mapping.keys())
    print()
    return set(relation_mapping.keys())



def construct_graph(cpnet_csv_path, cpnet_vocab_path, output_graph_path, output_ent_path, output_rel_path,prune=True):
    print('generating ConceptNet graph file...')

    nltk.download('stopwords', quiet=True)
    nltk_stopwords = nltk.corpus.stopwords.words('english')
    nltk_stopwords += ["like", "gone", "did", "going", "would", "could",
                       "get", "in", "up", "may", "wanter"]  # issue: mismatch with the stop words in grouding.py

    blacklist = set(["uk", "us", "take", "make", "object", "person", "people"])  # issue: mismatch with the blacklist in grouding.py

    concept2id = {}
    id2concept = {}
    with open(cpnet_vocab_path, "r", encoding="utf8") as fin:
        id2concept = [w.strip() for w in fin]
    concept2id = {w: i for i, w in enumerate(id2concept)}

    id2relation = merged_relations
    relation2id = {r: i for i, r in enumerate(id2relation)}

    graph = nx.MultiDiGraph()
    nrow = sum(1 for _ in open(cpnet_csv_path, 'r', encoding='utf-8'))
    with open(cpnet_csv_path, "r", encoding="utf8") as fin:

        def not_save(cpt):
            if cpt in blacklist:
                return True
            '''originally phrases like "branch out" would not be kept in the graph'''
            # for t in cpt.split("_"):
            #     if t in nltk_stopwords:
            #         return True
            return False

        attrs = set()

        for line in tqdm(fin, total=nrow):
            ls = line.strip().split('\t')
            rel = relation2id[ls[0]]
            subj = concept2id[ls[1]]
            obj = concept2id[ls[2]]
            weight = float(ls[3])
            if prune and (not_save(ls[1]) or not_save(ls[2]) or id2relation[rel] == "hascontext"):
                continue
            # if id2relation[rel] == "relatedto" or id2relation[rel] == "antonym":
            # weight -= 0.3
            # continue
            if subj == obj:  # delete loops
                continue
            # weight = 1 + float(math.exp(1 - weight))  # issue: ???

            if (subj, obj, rel) not in attrs:
                graph.add_edge(subj, obj, rel=rel, weight=weight)
                attrs.add((subj, obj, rel))
                graph.add_edge(obj, subj, rel=rel + len(relation2id), weight=weight)
                attrs.add((obj, subj, rel + len(relation2id)))

    nx.write_gpickle(graph, output_graph_path)
    print(f"graph file saved to {output_graph_path}")
    print('num of nodes: {}'.format(graph.number_of_nodes()))
    print('num of edges: {}'.format(graph.number_of_edges()))

    print()

    id2relation_reverse= ["_"+r for r in id2relation]
    id2relation +=id2relation_reverse
    relation2id = {r: i for i, r in enumerate(id2relation)}

    ent_dict = {'i2e': id2concept, 'e2i':concept2id}
    rel_dict = {'i2r': id2relation, 'r2i':relation2id}

    with open(output_rel_path, 'wb') as handle:
        pickle.dump(rel_dict, handle)
    print(f"save {output_rel_path}")

    with open(output_ent_path, 'wb') as handle:
        pickle.dump(ent_dict, handle)

    print(f"save {output_ent_path}")

    print("Finish")



def main():
    print("Generating CN47rel graph ...")
    data_dir='data/cpnet47rel/'
    check_path(data_dir)
    # conceptnet_path = os.path.join(data_dir, 'conceptnet-assertions-5.5.0.csv')
    conceptnet_path = '/home/chunhua/Commonsense/MHGRN/data/cpnet/conceptnet-assertions-5.6.0.csv' 
    output_csv_path  =  os.path.join(data_dir, 'conceptnet.en.csv')
    output_vocab_path = os.path.join(data_dir, 'concept.txt')
    output_graph_path = os.path.join(data_dir, 'graph.nx')

    output_ent_path = os.path.join(data_dir, 'entity_vocab.pkl')
    output_rel_path = os.path.join(data_dir, 'relation_vocab.pkl')


    extract_english(conceptnet_path, output_csv_path, output_vocab_path)
    construct_graph(output_csv_path, output_vocab_path, output_graph_path, output_ent_path, output_rel_path, prune=False)
    # construct_graph(output_csv_path, output_vocab_path, output_graph_path, output_ent_path, output_rel_path, prune=True)

    i2r, r2i, i2e, e2i = load_vocab(data_dir)
    print(i2r)
    print(r2i)


def check_relation_types():
    '''
    check whether relation groups equal to the original reported (Wang, 2020) 
    '''
    relation_mapping = load_merge_relation()
    print({i:rel for i, rel in enumerate(relation_mapping.keys())})


    print("-"*60) 
    print("Reported ConceptNet")
    data_dir='data/conceptnet'
    data_path = os.path.join(data_dir, 'conceptnet_graph.nx')
    # 
    kg_full = nx.read_gpickle(data_path)
    print('num of nodes: {}'.format(kg_full.number_of_nodes()))
    print('num of edges: {}'.format(kg_full.number_of_edges()))
    
    i2r_old, r2i_old, i2e_old, e2i_old = load_vocab(data_dir)
    print(i2r_old)
    print(r2i_old)

    # assert set(relation_mapping.keys()) == set(i2r_old[:40]) 
    # print("relation types before merging are the same as (Wang, 2020)")
    print("double check finished!")
    

if __name__=='__main__':
    check_relation_types()
    main()

    ## testing
    # print("CN 5.5.0")
    # conceptnet_path='/home/chunhua/Commonsense/Commonsense-Path-Generator/learning-generator/data/conceptnet_debug/conceptnet-assertions-5.5.0.csv'
    # rel1 = extract_english_all_relations(conceptnet_path)

    # print("CN 5.6.0")
    # conceptnet_path='./data/conceptnet_debug/conceptnet-assertions-5.6.0.csv'
    # rel2 = extract_english_all_relations(conceptnet_path)

    # print("rel2 - rel1: ".format(rel2.difference(rel1)))
    # sys.exit()
    
    # assert i2r_old==i2r, r2i_old==r2i
    # i=0
    # for u, v, data in kg_full.edges(data=True):
        # print(u,v, data)
        # i+=1
        # if i>10:
            # sys.exit()
