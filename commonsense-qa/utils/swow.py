import networkx as nx
import nltk
from nltk.corpus import stopwords
import json
import math
from tqdm import tqdm
import numpy as np
import sys
import os, sys
import pickle
import re
import argparse
import csv
import json
import string
from collections import Counter


try:
    from .utils import check_file, check_path
except ImportError:
    from utils import check_file


__all__ = ['extract_english', 'construct_graph', 'merged_relations', 'merged_relations_1rel', 'relation_groups']
global relation_groups, relation_groups_1rel

swow_rel_forward = "forwardassociated"
swow_rel_bidirectional = "bidirectionalassociated"
swow_rel_backward = "backwardassociated"


relation_groups = [
  swow_rel_forward,
  swow_rel_bidirectional 
]

merged_relations = [
   swow_rel_forward,
   swow_rel_bidirectional,
]

relation_groups_1rel= [
  swow_rel_forward,
]

merged_relations_1rel = [
   swow_rel_forward,
]


def load_merge_relation(kg_name):
    relation_mapping = dict()

    if kg_name == 'swow':
        relation_groups = relation_groups
    if kg_name == 'swow1rel':
        relation_groups = relation_groups_1rel

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


class SWOW(object):
    def __init__(self, swow_file, output_csv_path=None,  output_vocab_path=None, kg_name='swow', word_pair_freq=1):
        self.kg_name = kg_name 
        self.swow_data = self.load_swow_en(swow_file)
        self.swow_cue_responses, self.concepts = self.forward_associations(self.swow_data, word_pair_freq, unify_nodes=True)
        self.swow_cue_responses_relation = self.add_relations(self.swow_cue_responses)
        if output_csv_path is not None:
            self.write_cues(self.swow_cue_responses.keys(), output_path="./data/swow/swow_cues.csv")
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
                    print(f"dirty data: {cur}, {r}")
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
        cue_responses_relation= list() # bugfix: use list() instead of set(), guaranteeing vocab order to be the same for everytime 
        count_bi = 0
        count_fw = 0
        for cue, vs in cue_responses.items():
            for response, freq in vs.items():
                rel_forward = swow_rel_forward.lower()
                cue_responses_relation.append((rel_forward, cue, response, freq))
                count_fw +=1

                if self.kg_name == 'swow':
                    if response in cue_responses and cue in cue_responses[response]:
                        rel_bidirection = swow_rel_bidirectional.lower()
                        cue_responses_relation.append((rel_bidirection, cue, response, freq))
                        count_bi+=1
        print("Add {} forward association triples".format(count_fw))
        print("Add {} bi-directional association triples".format(count_bi))
        return cue_responses_relation

    def write_cues(self, cues, output_path):
        with open(output_path, 'w') as fout:
            for cue in cues:
              fout.write(cue+'\n')
        print("write {} {} cues".format(output_path, len(cues)))

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
        # cue_responses_relation = list(cue_responses_relation)
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

def extract_english(swow_path, output_csv_path, output_vocab_path, kg_name, word_pair_freq=1, language='en'):
    """
    Reads original conceptnet csv file and extracts all English relations (head and tail are both English entities) into
    a new file, with the following format for each line: <relation> <head> <tail> <weight>.
    :return:
    """
    print('extracting {} concepts and relations from SWOW...'.format(language))
    SWOW(swow_path, output_csv_path, output_vocab_path, kg_name, word_pair_freq)


def construct_graph(cpnet_csv_path, cpnet_vocab_path, output_path, prune=True, kg_name = 'swow'):
    print('generating {} graph file...'.format(kg_name))
    # nltk.download('stopwords', quiet=True)
    nltk_stopwords = nltk.corpus.stopwords.words('english')
    nltk_stopwords += ["like", "gone", "did", "going", "would", "could",
                       "get", "in", "up", "may", "wanter"]  # issue: mismatch with the stop words in grouding.py
    blacklist = set(["uk", "us", "take", "make", "object", "person", "people"])  # issue: mismatch with the blacklist in grouding.py

    concept2id = {}
    id2concept = {}
    with open(cpnet_vocab_path, "r", encoding="utf8") as fin:
        id2concept = [w.strip() for w in fin]
    concept2id = {w: i for i, w in enumerate(id2concept)}

    if kg_name=='swow':
        id2relation = merged_relations
    elif kg_name=='swow1rel':
        id2relation = merged_relations_1rel

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

    nx.write_gpickle(graph, output_path)
    print(f"graph file saved to {output_path}")
    print()


def glove_init(input, output, concept_file):
    embeddings_file = output + '.npy'
    vocabulary_file = output.split('.')[0] + '.vocab.txt'
    output_dir = '/'.join(output.split('/')[:-1])
    output_prefix = output.split('/')[-1]

    words = []
    vectors = []
    vocab_exist = check_file(vocabulary_file)
    print("loading embedding")
    with open(input, 'rb') as f:
        for line in f:
            fields = line.split()
            if len(fields) <= 2:
                continue
            if not vocab_exist:
                word = fields[0].decode('utf-8')
                words.append(word)
            vector = np.fromiter((float(x) for x in fields[1:]),
                                 dtype=np.float)

            vectors.append(vector)
        dim = vector.shape[0]
    print("converting")
    matrix = np.array(vectors, dtype="float32")
    print("writing")
    np.save(embeddings_file, matrix)
    text = '\n'.join(words)
    if not vocab_exist:
        with open(vocabulary_file, 'wb') as f:
            f.write(text.encode('utf-8'))

    def load_glove_from_npy(glove_vec_path, glove_vocab_path):
        vectors = np.load(glove_vec_path)
        with open(glove_vocab_path, "r", encoding="utf8") as f:
            vocab = [l.strip() for l in f.readlines()]

        assert (len(vectors) == len(vocab))

        glove_embeddings = {}
        for i in range(0, len(vectors)):
            glove_embeddings[vocab[i]] = vectors[i]
        print("Read " + str(len(glove_embeddings)) + " glove vectors.")
        return glove_embeddings

    def weighted_average(avg, new, n):
        # TODO: maybe a better name for this function?
        return ((n - 1) / n) * avg + (new / n)

    def max_pooling(old, new):
        # TODO: maybe a better name for this function?
        return np.maximum(old, new)

    def write_embeddings_npy(embeddings, embeddings_cnt, npy_path, vocab_path):
        words = []
        vectors = []
        for key, vec in embeddings.items():
            words.append(key)
            vectors.append(vec)

        matrix = np.array(vectors, dtype="float32")
        print(matrix.shape)

        print("Writing embeddings matrix to " + npy_path, flush=True)
        np.save(npy_path, matrix)
        print("Finished writing embeddings matrix to " + npy_path, flush=True)

        if not check_file(vocab_path):
            print("Writing vocab file to " + vocab_path, flush=True)
            to_write = ["\t".join([w, str(embeddings_cnt[w])]) for w in words]
            with open(vocab_path, "w", encoding="utf8") as f:
                f.write("\n".join(to_write))
            print("Finished writing vocab file to " + vocab_path, flush=True)

    def create_embeddings_glove(pooling="max", dim=100):
        print("Pooling: " + pooling)

        with open(concept_file, "r", encoding="utf8") as f:
            triple_str_json = json.load(f)
        print("Loaded " + str(len(triple_str_json)) + " triple strings.")

        glove_embeddings = load_glove_from_npy(embeddings_file, vocabulary_file)
        print("Loaded glove.", flush=True)

        concept_embeddings = {}
        concept_embeddings_cnt = {}
        rel_embeddings = {}
        rel_embeddings_cnt = {}

        for i in tqdm(range(len(triple_str_json))):
            data = triple_str_json[i]

            words = data["string"].strip().split(" ")

            rel = data["rel"]
            subj_start = data["subj_start"]
            subj_end = data["subj_end"]
            obj_start = data["obj_start"]
            obj_end = data["obj_end"]

            subj_words = words[subj_start:subj_end]
            obj_words = words[obj_start:obj_end]

            subj = " ".join(subj_words)
            obj = " ".join(obj_words)

            # counting the frequency (only used for the avg pooling)
            if subj not in concept_embeddings:
                concept_embeddings[subj] = np.zeros((dim,))
                concept_embeddings_cnt[subj] = 0
            concept_embeddings_cnt[subj] += 1

            if obj not in concept_embeddings:
                concept_embeddings[obj] = np.zeros((dim,))
                concept_embeddings_cnt[obj] = 0
            concept_embeddings_cnt[obj] += 1

            if rel not in rel_embeddings:
                rel_embeddings[rel] = np.zeros((dim,))
                rel_embeddings_cnt[rel] = 0
            rel_embeddings_cnt[rel] += 1

            if pooling == "avg":
                subj_encoding_sum = sum([glove_embeddings.get(word, np.zeros((dim,))) for word in subj])
                obj_encoding_sum = sum([glove_embeddings.get(word, np.zeros((dim,))) for word in obj])

                if rel in ["relatedto", "antonym"]:
                    # Symmetric relation.
                    rel_encoding_sum = sum([glove_embeddings.get(word, np.zeros((dim,))) for word in
                                            words]) - subj_encoding_sum - obj_encoding_sum
                else:
                    # Asymmetrical relation.
                    rel_encoding_sum = obj_encoding_sum - subj_encoding_sum

                subj_len = subj_end - subj_start
                obj_len = obj_end - obj_start

                subj_encoding = subj_encoding_sum / subj_len
                obj_encoding = obj_encoding_sum / obj_len
                rel_encoding = rel_encoding_sum / (len(words) - subj_len - obj_len)

                concept_embeddings[subj] = subj_encoding
                concept_embeddings[obj] = obj_encoding
                rel_embeddings[rel] = weighted_average(rel_embeddings[rel], rel_encoding, rel_embeddings_cnt[rel])

            elif pooling == "max":
                subj_encoding = np.amax([glove_embeddings.get(word, np.zeros((dim,))) for word in subj_words], axis=0)
                obj_encoding = np.amax([glove_embeddings.get(word, np.zeros((dim,))) for word in obj_words], axis=0)

                mask_rel = []
                for j in range(len(words)):
                    if subj_start <= j < subj_end or obj_start <= j < obj_end:
                        continue
                    mask_rel.append(j)
                rel_vecs = [glove_embeddings.get(words[i], np.zeros((dim,))) for i in mask_rel]
                rel_encoding = np.amax(rel_vecs, axis=0)

                # here it is actually avg over max for relation
                concept_embeddings[subj] = max_pooling(concept_embeddings[subj], subj_encoding)
                concept_embeddings[obj] = max_pooling(concept_embeddings[obj], obj_encoding)
                rel_embeddings[rel] = weighted_average(rel_embeddings[rel], rel_encoding, rel_embeddings_cnt[rel])

        print(str(len(concept_embeddings)) + " concept embeddings")
        print(str(len(rel_embeddings)) + " relation embeddings")

        write_embeddings_npy(concept_embeddings, concept_embeddings_cnt, f'{output_dir}/concept.{output_prefix}.{pooling}.npy',
                             f'{output_dir}/concept.glove.{pooling}.txt')
        write_embeddings_npy(rel_embeddings, rel_embeddings_cnt, f'{output_dir}/relation.{output_prefix}.{pooling}.npy',
                             f'{output_dir}/relation.glove.{pooling}.txt')

    create_embeddings_glove(dim=dim)


if __name__ == "__main__":
    glove_init("../data/glove/glove.6B.200d.txt", "../data/glove/glove.200d", '../data/glove/tp_str_corpus.json')
