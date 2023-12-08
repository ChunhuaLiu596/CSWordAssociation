from collections import Counter
import argparse
import numpy as np
import sys
import os
import re
import json
import time
import random
import pdb
import datetime
import spacy
import copy
from tqdm import tqdm

import graph
from reader import  ConceptNetTSVReader, SwowTSVReader, Triple2GraphReader
from utils_writer import *
import logging as logger
logger.getLogger().setLevel(logger.INFO)
PAD_TOKEN="_PAD"
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner', 'textcat'])

class Ground(object):
    def __init__(self, args):
        self.input_order = args.input_order
        self.write_non_overlap = args.write_non_overlap
        self.match_mode = args.match_mode
        self.add_isphrase_rel = args.add_isphrase_rel
        self.isphrase_rel_name = 'isphrase'
        self.debug = args.debug

        self.sw_net = self.build_net(args.swow_source_file, ConceptNetTSVReader, "SWOW")
        if self.match_mode !='hard':
            self.concept_to_lemma = self.entity_lemma_dict(list(self.sw_net.graph.node2id.keys()))

        #self.cn_net = self.build_net(args.swow_source_file, ConceptNetTSVReader, "SWOW")
        self.cn_net = self.build_net(args.conceptnet_source_file, ConceptNetTSVReader, "ConceptNet")

        self.overlap_edges_set, non_overlap = self.find_overlap_edges()
        self.overlap_nodes, self.non_overlap_nodes = self.find_overlap_nodes()

        self.write_triples(self.sw_net.edge_set, args.output_csv_path,
                    args.output_vocab_path, args.output_relation_path)
        #self.write_triples(self.overlap_edges_set, args.output_csv_path,
        #            args.output_vocab_path, args.output_relation_path)

        if self.write_non_overlap:
            self.write_triples(non_overlap,  args.output_csv_path_non_overlap,
                        args.output_vocab_path_non_overlap, args.output_relation_path_non_overlap,self.non_overlap_nodes)

    def entity_lemma_dict(self, concepts):
        concept_to_lemma = dict()

        #doc = nlp(concepts.replace("_", " "))
        #lcs = list()
        #for token in doc:
        #    lcs.extend("_".join(token.lemma_))

        for concept in concepts:
             lemma = list(lemmatize(nlp, concept))[0]
             if concept not in concept_to_lemma:
                 concept_to_lemma[concept] = lemma
        return concept_to_lemma


    def build_net(self, data_path, reader_cls, dataset):
        print("Loading {} ...".format(dataset))
        network = reader_cls(dataset, self.input_order)
        network.read_network(data_path)
        network.print_summary()
        return network

    def detect_single_instance(self):
        node1_name = 'handshake'
        node2_name = 'deal'
        rel = self.cn_net.graph.find_rel_by_node_name(node1_name, node2_name)
        if rel is not None:
            for x in rel:
                print(x)
        rel = self.cn_net.graph.find_rel_by_node_name(node2_name, node1_name)
        if rel is not None:
            for x in rel:
                print(x)

    def hard_ground(self, src, tgt):
        ''' 
        1. Retrieve relation type with raw src and tgt
            [(src, tgt)]
        ''' 
        rel_cn = self.cn_net.graph.find_rel_by_node_name(src, tgt)
        return [rel_cn] if rel_cn is not None else None

    def half_soft_ground(self, src, tgt):
        ''' 
        1. Retrieve relation type with raw src and tgt
            [(src, tgt)]
        2. If returns None, lemma src or tgt to find the relations 
            [(src_lemma, tgt_lemma), (src, tge_lemma), (src_lemma, tgt_lemma)]
        '''
        rel_cn = self.cn_net.graph.find_rel_by_node_name(src, tgt)

        if rel_cn is not None:
            return [rel_cn] if rel_cn is not None else None

        else:
            rel_cns = list()
            src_lemma = self.concept_to_lemma[src]
            tgt_lemma = self.concept_to_lemma[tgt]
            query = [(src, tgt_lemma), (src_lemma, tgt), (src_lemma, tgt_lemma)]

            for (s, t) in query:
                rel_cn = self.cn_net.graph.find_rel_by_node_name(s,t)
                if rel_cn is not None:
                    rel_cns.append(rel_cn)

            return rel_cns if len(rel_cns)>0 else None

    def total_soft_ground(self, src, tgt):
        '''
        Retrieve relation type with raw src and tgt, as well as lemma src and tgt
        [(src, tgt), (src_lemma, tgt_lemma), (src, tge_lemma), (src_lemma, tgt_lemma)]
        '''
        src_lemma = self.concept_to_lemma[src]
        tgt_lemma = self.concept_to_lemma[tgt]

        query = [(src, tgt), (src, tgt_lemma), (src_lemma, tgt), (src_lemma, tgt_lemma)]

        rel_cns = list()
        for (s, t) in query:
            rel_cn = self.cn_net.graph.find_rel_by_node_name(s,t)
            if rel_cn is not None:
                rel_cns.append(rel_cn)

        return rel_cns if len(rel_cns)>0 else None

    def detect_rel_cns(self, src, tgt):
        if self.match_mode == 'hard':
            rel_cns = self.hard_ground(src, tgt)
        if self.match_mode  == 'half_soft':
           rel_cns =  self.half_soft_ground(src, tgt)
        if self.match_mode  == 'total_soft':
           rel_cns =  self.total_soft_ground(src, tgt)
        return rel_cns

    def find_overlap_edges(self):
        overlap_edges_set = set()
        overlap_edges_nodes = set()
        overlap_edges_seen = set()

        overlap_rels = set()
        isphrase_edges_set = set()
        isphrase_edges_nodes = set()

        non_overlap_seen = set()
        non_overlap_edges_set = set()
        non_overlap_edges_nodes = set()

        count = list()
        count_lemma = list()

        def add_edges(edge, direction="src->tgt", lemma=True):
            if direction=="src->tgt":
                src = edge.src.name
                tgt = edge.tgt.name
            elif direction=="tgt->src":
                src = edge.tgt.name
                tgt = edge.src.name
            weight = edge.weight

            #1. detect relation from ConceptNet
            rel_cns = self.detect_rel_cns(src, tgt)

            if rel_cns is not None:
                for rel_cn in rel_cns:
                    for rel in rel_cn:
                        triple = (rel, src, tgt)
                        if triple not in overlap_edges_seen:
                            overlap_edges_seen.add(triple)
                            overlap_edges_set.add(triple + (weight,))

                            self.sw_net.edge_set.add(triple + (weight,))
                            count.append(1)

                            overlap_edges_nodes.update([src, tgt])
                            overlap_rels.add(rel)

            #2.detect whether head and tail can constitute a phrase in ConceptNet
            src_tgt = "_".join([src, tgt])
            is_phrase = self.cn_net.graph.find_node(src_tgt)

            if is_phrase !=-1: 
                triple = (self.isphrase_rel_name, src, tgt)
                if triple not in isphrase_edges_set:
                    isphrase_edges_set.add(triple)
                    isphrase_edges_nodes.update([src, tgt])

                    if self.add_isphrase_rel:
                        self.sw_net.edge_set.add(triple + (weight,))

            return rel_cns 

        n = self.sw_net.graph.edgeCount
        debug_count =0 
        for i, edge in enumerate(tqdm(self.sw_net.graph.iter_edges(), total=n, desc='retrieving overlapping edges')):
            rels_fw = add_edges(edge, "src->tgt")
            rels_bw = add_edges(edge, "tgt->src")

            if rels_fw is None and rels_bw is None:
                tuple = (edge.src.name, edge.tgt.name)
                if tuple not in non_overlap_seen :
                    if ((self.isphrase_rel_name,) + tuple) not in isphrase_edges_set:
                        non_overlap_seen.add(tuple)
                        non_overlap_edges_set.add((edge.relation.name,) + tuple + (edge.weight,))
                        non_overlap_edges_nodes.update([edge.src.name, edge.tgt.name])

            if self.debug:
                debug_count +=1
                if debug_count >100:
                    break

        if self.add_isphrase_rel:
            overlap_edges_num = len(overlap_edges_seen) + len(isphrase_edges_set)
        else:
            overlap_edges_num = len(overlap_edges_seen) 

        print("overlap_edges_num: {} with {} nodes, isphrase_edges_num: {} with {} nodes, non_overlap_edges_num: {} with {} nodes ".format(\
                overlap_edges_num, len(overlap_edges_nodes), len(isphrase_edges_set), len(isphrase_edges_nodes),\
                len(non_overlap_seen), len(non_overlap_edges_nodes)))

        assert len(self.sw_net.edge_set) - self.sw_net.graph.edgeCount == overlap_edges_num
        print("Original {} triples, newly add {} triples to SWOW, total {} triples".format(\
            self.sw_net.graph.edgeCount, overlap_edges_num, len(self.sw_net.edge_set)))


        #assert overlap_edges_set.issubset(self.cn_net.edge_set)
        #assert len(self.sw_net.edge_set.intersection(self.cn_net.edge_set))== len(overlap_edges_set), "{} {}".format( len(self.sw_net.edge_set.intersection(self.cn_net.edge_set)), len(overlap_edges_set))
        if self.add_isphrase_rel:
            return overlap_edges_set|isphrase_edges_set, non_overlap_edges_set
        else:
            return overlap_edges_set, non_overlap_edges_set

    def find_overlap_nodes(self, detect_soft_overlap=True):
        sw_nodes = self.sw_net.graph.node2id.keys()
        cn_nodes = self.cn_net.graph.node2id.keys()

        overlap_nodes = set(sw_nodes).intersection(set(cn_nodes))
        non_overlap_nodes = set(sw_nodes).difference(set(cn_nodes))

        print("hard mode: SWOW total_nodes_num: {}, overlap_nodes_num: {}, non_overlap_nodes_num: {}".format(\
               len(sw_nodes), len(overlap_nodes), len(non_overlap_nodes)))

        if detect_soft_overlap:
            soft_overlap, soft_non_overlap = self.find_soft_overlap_nodes(non_overlap_nodes, cn_nodes)
            overlap_nodes |= soft_overlap
            non_overlap_nodes = soft_non_overlap 

            print("soft mode: SWOW total_nodes_num: {}, overlap_nodes_num: {}, non_overlap_nodes_num: {}".format(\
               len(sw_nodes), len(overlap_nodes), len(non_overlap_nodes)))
        return overlap_nodes, non_overlap_nodes

    def find_soft_overlap_nodes(self, hard_non_overlap, cn_nodes):
        soft_overlap = set()

        soft_non_overlap = copy.deepcopy(hard_non_overlap)
        for node in hard_non_overlap:
            node_lemma = lemmatize(nlp, node)
            for token in node_lemma:
                if token in cn_nodes:
                    soft_overlap.add(token)
                    soft_non_overlap.remove(node)
        
        print("hard_non_overlap: {}, soft_overlap: {}, soft_non_overlap: {}".format(len(hard_non_overlap), len(soft_overlap), len(soft_non_overlap)))

        return  soft_overlap, soft_non_overlap

    def write_triples(self, cue_responses_relation,
                    output_csv_path, output_vocab_path, output_relation_path,
                    output_vocab=None):
        '''
        input: (rel, heat, tail, freq)
        '''
        cpnet_vocab = []
        cpnet_vocab.append(PAD_TOKEN)

        concepts_seen = set()
        relation_vocab = set()
        fout = open(output_csv_path, "w", encoding="utf8")
        cue_responses_relation = list(cue_responses_relation)
        cnt=0
        for (rel, head, tail, freq) in cue_responses_relation:
            fout.write('\t'.join([rel, head, tail, str(freq)]) + '\n')
            cnt+=1
            relation_vocab.add(rel)
            for w in [head, tail]:
                if w not in concepts_seen:
                    concepts_seen.add(w)
                    cpnet_vocab.append(w)

        with open(output_vocab_path, 'w') as fout:
            write_vocab = output_vocab if output_vocab is not None else cpnet_vocab
            for word in write_vocab:
                fout.write(word + '\n')

        relation_list = list(relation_vocab)
        with open(output_relation_path, 'w') as fout:
            for word in relation_list:
                fout.write(word + '\n')
        print()
        print(f'extracted %d triples to %s'%(cnt, output_csv_path))
        print(f'extracted %d concpet vocabulary to %s'%(len(write_vocab), output_vocab_path))
        print(f'extracted %d relations to %s'%(len(relation_list), output_relation_path))
        print()

def lemmatize(nlp, concept):

    doc = nlp(concept.replace("_", " "))
    lcs = set()
    lcs.add("_".join([token.lemma_ for token in doc]))  # all lemma
    return lcs


if __name__=='__main__':
    parser= argparse.ArgumentParser()
    parser.add_argument('--conceptnet_source_file',type=str, default='./data/cn100k/cn100k_train_valid_test.txt')
    parser.add_argument('--swow_source_file', type=str, default='./data/swow/swow_triple_freq2.filter')
    parser.add_argument('--input_order', type=str, default="rht")
    parser.add_argument('--match_mode', type=str, default="hard")

    parser.add_argument('--output_csv_path', type=str)
    parser.add_argument('--output_vocab_path', type=str)
    parser.add_argument('--output_relation_path', type=str)

    parser.add_argument('--output_csv_path_non_overlap', type=str)
    parser.add_argument('--output_vocab_path_non_overlap', type=str)
    parser.add_argument('--output_relation_path_non_overlap', type=str)

    parser.add_argument('--add_isphrase_rel', action='store_true')
    parser.add_argument('--write_non_overlap', action='store_true')
    parser.add_argument('--debug', action='store_true')

    args= parser.parse_args()

    Ground(args)

    #print("detecting case")
    #node1_name = 'paper'
    #node2_name = 'write'

    #rel = cn_net.graph.find_rel_by_node_name(node1_name, node2_name)
    #if rel is not None:
    #    for x in rel:
    #        print(x)

    #node1_name = 'although'
    #node2_name = 'but'
    #rel = cn_net.graph.find_rel_by_node_name(node1_name, node2_name)
    #if rel is not None:
    #    for x in rel:
    #        print(x)
    #else:
    #    print("rel doesn't exist")

