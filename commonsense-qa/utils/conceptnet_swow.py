import networkx as nx
import nltk
from nltk.corpus import stopwords
import json
import math
from tqdm import tqdm
import numpy as np
import sys

try:
    from .utils import check_file, check_path
except ImportError:
    from utils import check_file, check_path

__all__ = ['extract_english', 'construct_graph', 'merged_relations', 'load_merge_relation']
global relation_groups, relation_groups_7rel, relation_groups_1rel

relation_groups = [
    'forwardassociated',
    'bidirectionalassociated',
    'atlocation/locatednear',
    'capableof',
    'causes/causesdesire/*motivatedbygoal',
    'createdby',
    'desires',
    'antonym/distinctfrom',
    'hascontext',
    'hasproperty',
    'hassubevent/hasfirstsubevent/haslastsubevent/hasprerequisite/entails/mannerof',
    'isa/instanceof/definedas',
    'madeof',
    'notcapableof',
    'notdesires',
    'partof/*hasa',
    'relatedto/similarto/synonym',
    'usedfor',
    'receivesaction',
]

merged_relations = [
    'forwardassociated',
    'bidirectionalassociated',
    'antonym',
    'atlocation',
    'capableof',
    'causes',
    'createdby',
    'isa',
    'desires',
    'hassubevent',
    'partof',
    'hascontext',
    'hasproperty',
    'madeof',
    'notcapableof',
    'notdesires',
    'receivesaction',
    'relatedto',
    'usedfor',
]

def extract_english():
    pass

def load_merge_relation(kg_name):
    global relation_groups, relation_groups_1rel
    relation_mapping = dict()

    for i, line in enumerate(relation_groups):
        ls = line.strip().split('/')
        rel = ls[0]
        for l in ls:
            if l.startswith("*"):
                relation_mapping[l[1:]] = "*" + rel
            else:
                relation_mapping[l] = rel
    return relation_mapping

def merge_vocab(swow_vocab_path, cpnet_vocab_path, output_vocab_path):
    with open(cpnet_vocab_path, "r", encoding="utf8") as fin, open(swow_vocab_path, "r", encoding="utf8") as fins:
        concepts_vocab = [w.strip() for w in fin]
        concepts_seen = set(concepts_vocab )

        concepts_vocab_swow = [w.strip() for w in fins]
        for w in concepts_vocab_swow:
            if w not in concepts_seen:
                concepts_seen.add(w)
                concepts_vocab.append(w)

    with open(output_vocab_path, "w", encoding="utf8") as fout:
        for word in concepts_vocab:
            fout.write(word + '\n')

    print("write {} concepts ".format(len(concepts_vocab)))

def merge_rel_emb(cpnet_rel_path, swow_rel_path, output_rel_path):
    swow_rel_emb = np.load(swow_rel_path)
    cpnet_rel_emb = np.load(cpnet_rel_path)

    merge_rel_emb = np.concatenate((swow_rel_emb, cpnet_rel_emb), 0)

    np.save(output_rel_path, merge_rel_emb)
    print("Save {}, containing {} relation embeddings".format(output_rel_path, merge_rel_emb.shape))

# def merge_ent_emb(cpnet_vocab_path, cpnet_ent_path, swow_vocab_path, swow_ent_path, output_ent_path):

#     with open(cpnet_ent_path, "r", encoding="utf8") as fin:
#         cpnet_ent_emb = np.load(cpnet_ent_path)
#         swow_ent_emb = np.load(swow_ent_path)
    
#         id2concept_cpnet = [w.strip() for w in fin]

#     concept2id = {}
#     id2concept = {}
#     with open(cpnet_vocab_path, "r", encoding="utf8") as fin:
#         id2concept = [w.strip() for w in fin]
#     concept2id = {w: i for i, w in enumerate(id2concept)}



def merge_csv( cpnet_csv_path,swow_csv_path, output_csv_path):
    count_new=0
    graph = nx.MultiDiGraph()

    fo =  open(output_csv_path, "w")
    for line in tqdm(open(cpnet_csv_path, "r").readlines(), desc="Loading CN"):
        rel, subj, obj, weight = line.strip().split("\t")
        graph.add_edge(subj, obj, rel=rel, weight=weight)
        graph.add_edge(obj, subj, rel="_"+rel, weight=weight)
        fo.write(line)

    with open (swow_csv_path, "r") as fs:
        for line in tqdm(fs.readlines(), desc="Merging"):
            rel, subj, obj, weight = line.strip().split("\t")
            if graph.has_node(subj) and graph.has_node(obj):
                if graph.has_edge(subj, obj) is False: 
                    count_new += 1
                    fo.write(line)
            else:
                count_new += 1
                fo.write(line)

    print("Write {} new triples from SWOW".format(count_new))

def del_pos(s):
    """
    Deletes part-of-speech encoding from an entity string, if present.
    :param s: Entity string.
    :return: Entity string with part-of-speech encoding removed.
    """
    if s.endswith("/n") or s.endswith("/a") or s.endswith("/v") or s.endswith("/r"):
        s = s[:-2]
    return s

def construct_graph(cpnet_csv_path, cpnet_vocab_path, output_path, prune=True, kg_name='cpnet'):
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


def check_rels(kg_name):
    rel_mapping = load_merge_relation('cpnet')
    rel_group1 = set(rel_mapping.keys())
    print(len(rel_group1), rel_group1)

    if kg_name == 'cpnet7rel':
        rel_mapping_7rel = load_merge_relation('cpnet7rel')
        rel_group2 = set(rel_mapping_7rel.keys()) 

    if kg_name == 'cpnet1rel':
        rel_mapping_1rel = load_merge_relation('cpnet1rel')
        rel_group2 = set(rel_mapping_1rel.keys()) 

    # print(rel_group1 -  rel_group2)
    # print(rel_group2- rel_group1)
    # print(len(rel_group2), rel_group2)

    assert rel_group1 == rel_group2
    print("check relation groups finishes. {} relations before merging".format(len(rel_group1)))

if __name__ == "__main__":
    # glove_init("../data/glove/glove.6B.200d.txt", "../data/glove/glove.200d", '../data/glove/tp_str_corpus.json')
    # check_rels()
    merge_csv("./data/cpnet/conceptnet.en.csv", "./data/swow/conceptnet.en.csv", "./data/cpnet_swow/conceptnet.en.csv")
    # merge_vocab("./data/cpnet/concept.txt", "./data/swow/concept.txt", "./data/cpnet_swow/concept.txt")
    # merge_rel_emb("./data/transe/glove.transe.sgd.rel.npy", "./data/swow/glove.TransE.SGD.rel.npy", "./data/cpnet_swow/glove.transe.sgd.rel.npy")

    # merge_ent_emb("./data/cpnet/concept_roberta_emb.npy", "./data/swow/concept_roberta_emb.npy", "./data/cpnet_swow/concept_roberta_emb.npy")