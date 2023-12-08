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
from graphs_data import GraphsData
from parser_utils import get_parser

import logging as logger
logger.getLogger().setLevel(logger.INFO)
PAD_TOKEN="_PAD"
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner', 'textcat'])

class Ground(GraphsData):
    def __init__(self, args):
        super().__init__(args)
        self.args = args

        self.overlap_edges_set = set()
        self.overlap_edges_nodes = set()
        self.overlap_edges_seen = set()

        self.overlap_rels = set()
        self.isphrase_edges_set = set()
        self.isphrase_edges_nodes = set()

        self.non_overlap_seen = set()
        self.non_overlap_edges_set = set()
        self.non_overlap_edges_nodes = set()

        self.overlap_nodes = set()
        self.non_overlap_nodes = set()
        #-------- find overlap ------
        self.find_overlap_edges()
        self.find_overlap_nodes()
        #----------------------------

        self.overlap_edges_num = len(self.overlap_edges_seen)
        self.add_additional_overlap()
        self.print_summary()

        #----------------------------
        if self.args.write_ground_triples:
            self.write_triples(self.overlap_edges_set, args.output_csv_path_cn)
            # self.write_triples(self.overlap_edges_set, args.output_csv_path_sw)

    def find_overlap_edges(self):
       
        count_lemma = list()
        n = self.net_sw.graph.edgeCount
        debug_count =0

        for i, edge in enumerate(tqdm(self.net_sw.graph.iter_edges(), total=n,\
                                      desc='retrieving overlapping edges from ConceptNet')):
            if not self.args.swap_retrieval:
                rels_fw = self.add_edge(edge, "src->tgt")
                if rels_fw is None:
                    self.add_non_overlap(edge)
            else:
                rels_fw = self.add_edge(edge, "src->tgt")
                rels_bw = self.add_edge(edge, "tgt->src") 
                if rels_fw is None and rels_bw is None:
                     self.add_non_overlap(edge)

            if self.debug:
                debug_count +=1  
                if debug_count >100:
                    break        

    def add_edge(self, edge, direction="src->tgt", lemma=True):
        if direction=="src->tgt":
            src = edge.src.name
            tgt = edge.tgt.name
        elif direction=="tgt->src":
            src = edge.tgt.name
            tgt = edge.src.name
        weight = edge.weight
        rel = edge.relation.name
      
        #1. detect relation from ConceptNet
        rel_cns = self.detect_rel_cns(src, tgt)

        if rel_cns is not None:
            for rel_cn in rel_cns:
                for rel in rel_cn:
                    rel_value = rel[0]
                    weight_value = float(rel[1])
                    triple = (rel_value, src, tgt)

                    if triple not in self.overlap_edges_seen:
                        self.overlap_edges_seen.add(triple)
                        # self.overlap_edges_set.add(triple + (weight_value,))
                        self.overlap_edges_set.add(triple + (weight,))

                        if self.args.add_cn_triples_to_swow: 
                            self.net_sw.edge_set.add(triple + (weight,))

                        self.overlap_edges_nodes.update([src, tgt])
                        self.overlap_rels.add(rel_value)

        #2.detect whether head and tail can constitute a phrase in ConceptNet
        if self.args.add_isphrase_rel:
            self.add_is_phrase_overlap(src, tgt, weight)
        return rel_cns

    def add_is_phrase_overlap(self, src, tgt, weight):
        src_tgt = "_".join([src, tgt])
        is_phrase = self.net_cn.graph.find_node(src_tgt)

        if is_phrase !=-1:
            triple = (self.isphrase_rel_name, src, tgt)
            if triple not in self.isphrase_edges_set:
                self.isphrase_edges_set.add(triple)
                self.isphrase_edges_nodes.update([src, tgt])

                if self.add_isphrase_rel:
                    self.net_sw.edge_set.add(triple + (weight,))

    def add_non_overlap(self, edge):
        tuple = (edge.src.name, edge.tgt.name)
        if tuple not in self.non_overlap_seen:
           if ((self.isphrase_rel_name,) + tuple) not in self.isphrase_edges_set:
               self.non_overlap_seen.add(tuple)
               self.non_overlap_edges_set.add((edge.relation.name,) + tuple + (edge.weight,))
               self.non_overlap_edges_nodes.update([edge.src.name, edge.tgt.name])

    def add_additional_overlap(self):
        if self.add_isphrase_rel:
           self.overlap_edges_set|=self.isphrase_edges_set
           self.overlap_edges_num +=len(self.isphrase_edges_set)

    def print_summary(self):
        print("overlap_edges_num: {} with {} nodes, \nisphrase_edges_num: {} with {} nodes, \nnon_overlap_edges_num: {} with {} nodes. ".format(\
                self.overlap_edges_num, len(self.overlap_edges_nodes),\
                len(self.isphrase_edges_set), len(self.isphrase_edges_nodes),\
                len(self.non_overlap_seen), len(self.non_overlap_edges_nodes)))

        if self.args.add_cn_triples_to_swow:
            assert len(self.net_sw.edge_set) - self.net_sw.graph.edgeCount == self.overlap_edges_num
            print("Original {} triples SWOW, newly add {} triples, total {} triples".format(\
            self.net_sw.graph.edgeCount, self.overlap_edges_num, len(self.net_sw.edge_set)))
        if self.args.match_mode == 'hard':
            assert overlap_edges_set.issubset(self.net_cn.edge_set)
            assert len(self.net_sw.edge_set.intersection(self.net_cn.edge_set))== len(overlap_edges_set), "{} {}".format( len(self.net_sw.edge_set.intersection(self.net_cn.edge_set)), len(overlap_edges_set))

    def find_overlap_nodes(self):
        sw_nodes = self.net_sw.graph.node2id.keys()
        cn_nodes = self.net_cn.graph.node2id.keys()

        self.overlap_nodes = set(sw_nodes).intersection(set(cn_nodes))
        self.non_overlap_nodes = set(sw_nodes).difference(set(cn_nodes))

        if self.args.match_mode=='hard':
            print("hard mode: SWOW total_nodes_num: {}, overlap_nodes_num: {}, non_overlap_nodes_num: {}".format(\
               len(sw_nodes), len(self.overlap_nodes), len(self.non_overlap_nodes)))
        else:
            soft_overlap, soft_non_overlap = self.find_soft_overlap_nodes(self.non_overlap_nodes, cn_nodes)
            self.overlap_nodes |= soft_overlap
            self.non_overlap_nodes = soft_non_overlap

            print("soft mode: SWOW total_nodes_num: {}, overlap_nodes_num: {}, non_overlap_nodes_num: {}".format(\
               len(sw_nodes), len(self.overlap_nodes), len(self.non_overlap_nodes)))
        #return overlap_nodes, non_overlap_nodes

    def find_soft_overlap_nodes(self, hard_non_overlap, cn_nodes):
        soft_overlap = set()

        soft_non_overlap = copy.deepcopy(hard_non_overlap)
        n = len(hard_non_overlap)
        for i, node in enumerate(tqdm(hard_non_overlap, total=n, desc='finding soft overlap nodes')):
            #node_lemma = self.lemmatize(nlp, node)
            token = self.concept_to_lemma[node]
            #for token in node_lemma:
            #    if token in cn_nodes:
            if token in cn_nodes:
                 soft_overlap.add(token)
                 soft_non_overlap.remove(node)

        print("hard_non_overlap: {}, soft_overlap: {}, soft_non_overlap: {}".format(len(hard_non_overlap), len(soft_overlap), len(soft_non_overlap)))

        return  soft_overlap, soft_non_overlap

    def detect_single_instance(self):
        node1_name = 'handshake'
        node2_name = 'deal'
        rel = self.net_cn.graph.find_rel_by_node_name(node1_name, node2_name)
        if rel is not None:
            for x in rel:
                print(x)
        rel = self.net_cn.graph.find_rel_by_node_name(node2_name, node1_name)
        if rel is not None:
            for x in rel:
                print(x)

    def hard_ground(self, src, tgt):
        '''
        1. Retrieve relation type with raw src and tgt
            [(src, tgt)]
        '''
        rel_cn = self.net_cn.graph.find_rel_by_node_name(src, tgt, weight=True)
        return [rel_cn] if rel_cn is not None else None

    def half_soft_ground(self, src, tgt):
        '''
        1. Retrieve relation type with raw src and tgt
            [(src, tgt)]
        2. If returns None, lemma src or tgt to find the relations
            [(src_lemma, tgt_lemma), (src, tge_lemma), (src_lemma, tgt_lemma)]
        '''
        rel_cn = self.net_cn.graph.find_rel_by_node_name(src, tgt, weight=True)

        if rel_cn is not None:
            return [rel_cn] if rel_cn is not None else None

        else:
            rel_cns = list()
            src_lemma = self.concept_to_lemma[src]
            tgt_lemma = self.concept_to_lemma[tgt]
            query = [(src, tgt_lemma), (src_lemma, tgt), (src_lemma, tgt_lemma)]

            for (s, t) in query:
                rel_cn = self.net_cn.graph.find_rel_by_node_name(s,t)
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
            rel_cn = self.net_cn.graph.find_rel_by_node_name(s,t, weight=True)
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

    def write_triples(self, triples, output_csv_path, output_vocab_path=None, output_relation_path=None,
                    output_vocab=None):
        '''
        input: (rel, heat, tail, freq)
        '''
        cpnet_vocab = []
        cpnet_vocab.append(PAD_TOKEN)

        concepts_seen = set()
        relation_vocab = set()
        fout = open(output_csv_path, "w", encoding="utf8")
        triples = list(triples)
        cnt=0
        for (rel, head, tail, freq) in triples:
            fout.write('\t'.join([rel, head, tail, str(freq)]) + '\n')
            cnt+=1
            relation_vocab.add(rel)
            for w in [head, tail]:
                if w not in concepts_seen:
                    concepts_seen.add(w)
                    cpnet_vocab.append(w)
        print()
        print(f'extracted %d triples to %s'%(cnt, output_csv_path))

        if output_vocab_path is not None:
            with open(output_vocab_path, 'w') as fout:
                write_vocab = output_vocab if output_vocab is not None else cpnet_vocab
                for word in write_vocab:
                    fout.write(word + '\n')
            print(f'extracted %d concpet vocabulary to %s'%(len(write_vocab), output_vocab_path))

        if output_relation_path is not None:
            relation_list = list(relation_vocab)
            with open(output_relation_path, 'w') as fout:
                for word in relation_list:
                    fout.write(word + '\n')
            print(f'extracted %d relations to %s'%(len(relation_list), output_relation_path))

        print()

if __name__=='__main__':
    # parser= argparse.ArgumentParser()
    # parser.add_argument('--conceptnet_source_file',type=str, default='./data/cn100k/cn100k_train_valid_test.txt')
    # parser.add_argument('--swow_source_file', type=str, default='./data/swow/swow_triple_freq2.filter')
    # parser.add_argument('--input_order', type=str, default="rht")
    # parser.add_argument('--match_mode', type=str, default="hard", choices=["hard", "total_soft"])
    # parser.add_argument('--align_dir',type=str, default="data/alignment/C_S_V0.1")

    # parser.add_argument('--write_ground_triples', action='store_true')
    # parser.add_argument('--output_csv_path_cn', type=str)
    # parser.add_argument('--output_vocab_path_cn', type=str)
    # parser.add_argument('--output_relation_path_cn', type=str)

    # parser.add_argument('--output_csv_path_sw', type=str)

    # parser.add_argument('--output_csv_path_non_overlap', type=str)
    # parser.add_argument('--output_vocab_path_non_overlap', type=str)
    # parser.add_argument('--output_relation_path_non_overlap', type=str)

    # parser.add_argument('--add_isphrase_rel', action='store_true')
    # parser.add_argument('--write_non_overlap', action='store_true')
    # parser.add_argument('--debug', action='store_true')

    # parser.add_argument('--swap_retrieval', action='store_true', help='swap head and tail to retrieve edge when finding overlap edges')

    # args= parser.parse_args()

    parser = get_parser()
    args = parser.parse_args()

    Ground(args)

    #print("detecting case")
    #node1_name = 'paper'
    #node2_name = 'write'

    #rel = net_cn.graph.find_rel_by_node_name(node1_name, node2_name)
    #if rel is not None:
    #    for x in rel:
    #        print(x)

    #node1_name = 'although'
    #node2_name = 'but'
    #rel = net_cn.graph.find_rel_by_node_name(node1_name, node2_name)
    #if rel is not None:
    #    for x in rel:
    #        print(x)
    #else:
    #    print("rel doesn't exist")

