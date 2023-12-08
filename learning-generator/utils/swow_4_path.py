import os, sys
import networkx as nx
from networkx.utils import open_file
import nltk
import json
import math
from tqdm import tqdm
import numpy as np
import pickle
import re
import argparse
import csv
import json
import string
from collections import Counter
from utils import check_path


swow_rel_forward = "forwardassociated"
swow_rel_bidirectional = "bidirectionalassociated"
swow_rel_backward = "backwardassociated"

selected_relations = [swow_rel_forward, swow_rel_bidirectional]


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
    for rel in selected_relations:
        if rel.startswith("*"):
            relation_mapping[rel[1:]] = "*" + rel
        else:
            relation_mapping[rel] = rel
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


class SWOW(object):
    def __init__(self, swow_file, output_csv_path=None,  output_vocab_path=None, word_pair_freq=1):

        self.swow_data = self.load_swow_en(swow_file)
        self.swow_cue_responses, self.concepts = self.forward_associations(self.swow_data, word_pair_freq, unify_nodes=True)
        self.swow_cue_responses_relation = self.add_relations(self.swow_cue_responses)
        if output_csv_path is not None:
            self.write_forward_associations_relation(self.swow_cue_responses_relation, output_csv_path, output_vocab_path)

    def load_swow_en(self, input_file):
        cues, R1, R2, R3 = list(),list(),list(),list()
        reader =csv.DictReader(open(input_file))
        for row in reader:
            cues.append(row['cue'].lower())
            R1.append(row['R1'].lower())
            R2.append( row['R2'].lower())
            R3.append( row['R3'].lower())

        swow_data = list(zip(cues, R1, R2, R3))
        print("Loaded %d lines from %s"%(len(cues),input_file))
        return swow_data

    def unify_sw_nodes(self, node):
        '''unify entity format with ConceptNet5.6, in which entity is concatenated by words with _'''
        '''keep words concatenated by -, like 'self-esteem', 'self-important' '''
        node_list_raw = re.split(' ', node)

        blacklist = ['a'] # a club,
        if len(node_list_raw)>1 and node_list_raw[0] in blacklist:
            node_list_raw.remove(node_list_raw[0])

        if node_list_raw[0].startswith("-"): #-free (gluten -free)
            node_list_raw[0] = node_list_raw[0][1:]

        if node_list_raw[0].startswith("_"): #_position
            node_list_raw[0] = node_list_raw[0][1:]

         #cases: beard_-_eyebrows_-_mustache,  bearskin___________disrobe_________reveal, bear__wine
        node_list =  []
        for node in node_list_raw:
            node = node.replace("_-_", "")
            node = node.replace("___________", "")
            node = node.replace("__", "")
            node = node.replace("_", "")
            node = node.replace("__","")
            node = node.replace("__","")
            #node = node.replace("-", "_") #real text contains -, eg, self-important
            if node: # if not empty string, "- Johnson wife of lyndon"
                node_list.append(node)

        node_len = len(node_list)
        if node_len >0:
            node_phrase = "_".join(node_list)
            #if not en_dict.check(node_phrase):
            #   print(node_phrase)
            return node_phrase, node_len
        else: #filter empty node
            #print("empty node: {}".format(node_list_raw))
            return None, None

    def forward_associations(self, swow_data, word_pair_freq, unify_nodes=False):
        cue_responses={}
        concepts=set()
        phrase_seen = dict()
        for i, (cue, r1, r2, r3) in enumerate(swow_data):

            cue = cue.lower()
            if unify_nodes:
                phrase_ori = cue
                cue, phrase_len = self.unify_sw_nodes(cue)
                if phrase_len is None:
                    continue
                if phrase_len >1:
                    if cue not in phrase_seen:
                        phrase_seen[cue]=[phrase_ori]
                    else:
                        phrase_seen[cue].extend([phrase_ori])

            for r in [r1, r2, r3]:
                #if cue not in cue_responses.keys() and r!="NA" or "na":
                r = r.lower()

                if r=='na': continue
                if cue == r: continue #about 1000, e.g., read, aimless, sheen, elbows

                if unify_nodes:
                    phrase_ori = r
                    r, phrase_len = self.unify_sw_nodes(r)

                    if phrase_len is None:
                        continue

                    if phrase_len >1:
                        if r not in phrase_seen:
                            phrase_seen[r]=[phrase_ori]
                        else:
                            phrase_seen[r].extend([phrase_ori])

                if not cue.replace("_", "").replace("-", "").replace(" ","").replace("''","").isalpha():
                    #print("cue: {}".format(cue))
                    continue
                if not r.replace("_", "").replace("-", "").replace(" ","").replace("''","").isalpha():
                    #print("response: {}".format(r))
                    continue

                if cue in string.punctuation or r in string.punctuation:
                    print(f"dirty data: {cue}, {r}")
                    continue

                if cue not in cue_responses.keys() :
                    cue_responses[cue]={r:1}
                    concepts.add(cue)
                    concepts.add(r)
                else:
                    cue_responses = self.add_elements(cue_responses, cue, r)
                    concepts.add(r)


        num_swow_triplets = sum([len(x) for x in cue_responses.values()])
        print("Number of original triplets in SWOW is {}".format(num_swow_triplets))
        if word_pair_freq >1:
            cue_responses = self.filter_frequency(cue_responses, word_pair_freq)
            cut_down_num = num_swow_triplets - sum([len(x) for x in cue_responses.values()])
            print("Cutting down {} triplets whose wordpair_frequency<{}".format(cut_down_num, word_pair_freq))

            num_swow_triplets = sum([len(x) for x in cue_responses.values()])
            print("Number of original triplets in SWOW is {} (after cutting down)".format(num_swow_triplets))

        return cue_responses, concepts

    def add_relations(self, cue_responses):
        cue_responses_relation= set()
        count_bi = 0
        count_fw = 0
        for cue, vs in cue_responses.items():
            for response, freq in vs.items():
                rel_forward = swow_rel_forward.lower()
                cue_responses_relation.add((rel_forward, cue, response, freq))
                count_fw +=1

                if response in cue_responses and cue in cue_responses[response]:
                    rel_bidirection = swow_rel_bidirectional.lower()
                    cue_responses_relation.add((rel_bidirection, cue, response, freq))
                    count_bi+=1
        print("Add {} forward association triples".format(count_fw))
        print("Add {} bi-directional association triples".format(count_bi))
        return cue_responses_relation

    def write_forward_associations_relation(self, cue_responses_relation,
                    output_csv_path, output_vocab_path):
        '''
        input: (rel, heat, tail, freq)
        '''
        cpnet_vocab = []
        # cpnet_vocab.append(PAD_TOKEN)

        concepts_seen = set()
        check_path(output_csv_path)
        fout = open(output_csv_path, "w", encoding="utf8")
        cue_responses_relation = list(cue_responses_relation)
        cnt=0
        for (rel, head, tail, freq) in cue_responses_relation:
            fout.write('\t'.join([rel, head, tail, str(freq)]) + '\n')
            cnt+=1
            for w in [head, tail]:
                if w not in concepts_seen:
                    concepts_seen.add(w)
                    cpnet_vocab.append(w)

        check_path(output_vocab_path)
        with open(output_vocab_path, 'w') as fout:
            for word in cpnet_vocab:
                fout.write(word + '\n')

        print('extracted {} triples to {}'.format(cnt, output_csv_path))
        print('extracted {} concpet vocabulary to {}'.format(len(cpnet_vocab), output_vocab_path))
        print()

        return cpnet_vocab


    def add_elements_dict2d(self,outter, outter_key, inner_key,value):
        if outter_key not in outter.keys():
            outter.update({outter_key:{inner_key:value}})
        else:
            outter[outter_key].update({inner_key:value})
        return outter

    def add_elements(self,outter, outter_key, inner_key):
        if inner_key not in outter[outter_key].keys():
            outter[outter_key].update({inner_key:1})
        else:
            outter[outter_key][inner_key]+=1
        return outter

    def filter_frequency(self,cue_responses, word_pair_freq=2):
        new_cue_responses={}

        for i, (cue,responses) in enumerate(tqdm(cue_responses.items())):
            for response,frequency in responses.items():
                if response == 'NA' or response=='na': continue

                if frequency >= word_pair_freq:
                    self.add_elements_dict2d(outter=new_cue_responses,
                                        outter_key=cue,
                                        inner_key=response,
                                        value=frequency)
        return new_cue_responses

def extract_english_swow(swow_path, output_csv_path, output_vocab_path, word_pair_freq, language='en'):
    """
    Reads original conceptnet csv file and extracts all English relations (head and tail are both English entities) into
    a new file, with the following format for each line: <relation> <head> <tail> <weight>.
    :return:
    """
    print('extracting {} concepts and relations from SWOW...'.format(language))
    SWOW(swow_path, output_csv_path, output_vocab_path, word_pair_freq)


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

    id2relation = selected_relations
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
    # conceptnet_path = os.path.join(data_dir, 'conceptnet-assertions-5.5.0.csv')
    # conceptnet_path = '/home/chunhua/Commonsense/MHGRN/data/cpnet/conceptnet-assertions-5.6.0.csv' 

    data_dir='data/swow'
    swow_path='/scratch/chunhua/Commonsense-bkup/OpenEntitiyAlignment/conceptnet-swow-data/data/swow/SWOW-EN.R100.csv' 
    output_csv_path  =  os.path.join(data_dir, 'swow.en.csv')
    output_vocab_path = os.path.join(data_dir, 'concept.txt')
    output_graph_path = os.path.join(data_dir, 'graph.nx')

    output_ent_path = os.path.join(data_dir, 'entity_vocab.pkl')
    output_rel_path = os.path.join(data_dir, 'relation_vocab.pkl')

    extract_english_swow(swow_path, output_csv_path, output_vocab_path, word_pair_freq=1)
    construct_graph(output_csv_path, output_vocab_path, output_graph_path, output_ent_path, output_rel_path, prune=False)

    i2r, r2i, i2e, e2i = load_vocab(data_dir)
    print(i2r)
    print(r2i)

    print("-"*60) 
    print("ConceptNet graph")
    data_dir='data/conceptnet'
    data_path = os.path.join(data_dir, 'conceptnet_graph.nx')
    # 
    kg_full = nx.read_gpickle(data_path)
    print('num of nodes: {}'.format(kg_full.number_of_nodes()))
    print('num of edges: {}'.format(kg_full.number_of_edges()))

if __name__=='__main__':
    main()