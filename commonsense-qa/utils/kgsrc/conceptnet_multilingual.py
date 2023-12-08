import os
import networkx as nx
import nltk
import json
import math
from tqdm import tqdm
import numpy as np
import sys
import argparse
from reader import  ConceptNetTSVReader, SwowTSVReader, Triple2GraphReader
from graph_statistics import get_statistics
#try:
#    from .utils import check_file
#except ImportError:
#    from utils import check_file

__all__ = ['extract_english', 'construct_graph', 'merged_relations']

#languages_list = ['zh', 'fr', 'en', 'de', 'it', 'pt', 'es', 'ru', 'ja', 'nl', 'vi']
# languages_list = ['en', 'nl']
languages_list = ['en']
relation_groups = [
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

relation_text = [
    'is the antonym of',
    'is at location of',
    'is capable of',
    'causes',
    'is created by',
    'is a kind of',
    'desires',
    'has subevent',
    'is part of',
    'has context',
    'has property',
    'is made of',
    'is not capable of',
    'does not desires',
    'is',
    'is related to',
    'is used for',
]

PAD_TOKEN="_PAD"

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

def extract_specify_language(conceptnet_path, lang, out_folder, debug=False):
    """
    Reads original conceptnet csv file and extracts all English relations (head and tail are both English entities) into
    a new file, with the following format for each line: <relation> <head> <tail> <weight>.
    :return:
    """
    print(f'extracting {lang} concepts and relations from ConceptNet...')
    relation_mapping = load_merge_relation()
    num_lines = sum(1 for line in open(conceptnet_path, 'r', encoding='utf-8'))
    cpnet_vocab = []
    cpnet_vocab.append(PAD_TOKEN)
    triples_seen = set()
    concepts_seen = set()
    relations_seen = set()
    count=0

    output_csv_path = os.path.join(out_folder,f'conceptnet.{lang}.csv')
    output_vocab_path = os.path.join(out_folder, f'concept.{lang}.txt')
    output_relation_path = os.path.join(out_folder, f'relation.{lang}.txt')

    #language_to_triples = dict()
    with open(conceptnet_path, 'r', encoding="utf8") as fin, \
            open(output_csv_path, 'w', encoding="utf8") as fout:
        for line in tqdm(fin, total=num_lines):
            toks = line.strip().split('\t')

            head_c = toks[2].split("/")[2]  #get the languages
            tail_c = toks[3].split("/")[2]  #get the languages
            if head_c == tail_c == lang:
                #if head_c == lang or tail_c == lang:
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

                #if rel not in relation_mapping:
                #    continue

                #rel = relation_mapping[rel]
                #if rel.startswith("*"):
                #    head, tail, rel = tail, head, rel[1:]relaitons

                data = json.loads(toks[4])

                #if lang not in language_to_triples:
                #    language_to_triples[lang] = 1
                #else:
                #    language_to_triples[lang] +=1
                if (rel, head, tail) not in triples_seen:
                    #and head_c != tail_c:
                    triples_seen.add((rel, head, tail))
                    #head_out =  "{}:{}".format(head, head_c) 
                    #tail_out =  "{}:{}".format(tail, tail_c) 
                    #fout.write('\t'.join([rel, head_out, tail_out, str(data["weight"])]) + '\n')
                    fout.write('\t'.join([rel, head, tail, str(data["weight"])]) + '\n')
                    count+=1
                    if debug and count>100:
                        break

                    if rel not in relations_seen:
                        relations_seen.add(rel)
                    to_add = list()
                    if head_c == lang:
                        to_add.append(head) 
                    if tail_c == lang:
                        to_add.append(tail)

                    #for w in [head, tail]:
                    for w in to_add:
                        if w not in concepts_seen:
                            concepts_seen.add(w)
                            cpnet_vocab.append(w)

    #print("Total {} languages".format(len(language_to_triples.keys())))
    #for key, v in sorted(language_to_triples.items(), key=lambda x:x[1], reverse=True):
    #    print("{}\t{}".format(key,v))
    with open(output_vocab_path, 'w') as fout:
        for word in cpnet_vocab:
            fout.write(word + '\n')

    with open(output_relation_path, 'w') as fout:
        for word in sorted(relations_seen):
            fout.write(word + '\n')

    print(f'extracted ConceptNet csv file with {count} triples, saved to {output_csv_path}')
    print(f'extracted {len(cpnet_vocab)} concept vocabulary saved to {output_vocab_path}')
    print(f'extracted {len(relations_seen)} relation vocabulary saved to {output_relation_path}')
    print()

def extract_multi_languages(args):
    for lang in languages_list:
        extract_specify_language(args.conceptnet_path, lang, args.out_folder, debug=args.debug)
        if args.debug:
            break

def get_kg_statistics(args):
    lang_to_stis = dict()
    for lang in languages_list:
       input_csv_path = os.path.join(args.out_folder, f'conceptnet.{lang}.csv')
       #(edges_num, nodes_num, relations_num, density, degree)

       (edges_num, nodes_num, relations_num, density, degree, ent_entropy, rel_entropy) = get_statistics(lang, input_csv_path, ConceptNetTSVReader, 'rhtw')

       if lang not in lang_to_stis:
            lang_to_stis[lang] = (edges_num, nodes_num, relations_num, density, degree, ent_entropy, rel_entropy)
           #lang_to_stis[lang] = (edges_num, nodes_num, relations_num, density, degree)
       if args.debug:
           break

    print("lang\tedges_num \tnodes_num\trelations_num\tdensity\tdegree\tent_entropy\trel_entropy")
    for lang in languages_list:
        stis = lang_to_stis[lang]
        out = "{}\t{}\t{}\t{}\t{}\t{}\t{}".format(stis[0], stis[1], stis[2], stis[3], stis[4],stis[5], stis[6])
        print(lang, out)
        if args.debug:
           break



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--conceptnet_path", type=str, default='/home/chunhua/Commonsense/MHGRN/data/cpnet/conceptnet-assertions-5.6.0.csv')
    #parser.add_argument("--conceptnet_path", type=str, default='/home/chunhua/Commonsense/MHGRN/data/cpnet/conceptnet-assertions-5.7.0.csv')
    parser.add_argument("--out_folder", type=str, default='../../data/cpnet/multilingual/')
    parser.add_argument("--debug", action='store_true')
    args = parser.parse_args()

    extract_multi_languages(args)
    get_kg_statistics(args)
