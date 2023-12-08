from collections import Counter
import argparse
import numpy as np
import sys
import os
import json
import time
import random
import pdb
import datetime
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import graph
from reader import  ConceptNetTSVReader, SwowTSVReader, Triple2GraphReader
from utils_writer import *
import logging as logger
logger.getLogger().setLevel(logger.INFO)
PAD_TOKEN="_PAD"

class Merge(object):
    def __init__(self, args):
        #self.sw_net = self.build_net(args.swow_source_file, SwowTSVReader)
        self.input_order = args.input_order
        self.sw_net = self.build_net(args.swow_source_file, ConceptNetTSVReader, "SWOW") 
        self.cn_net = self.build_net(args.conceptnet_source_file, ConceptNetTSVReader, "ConceptNet")

        all_edge_list = self.cn_net.edge_list + self.sw_net.edge_list
        print("Conceptnet edge num: {}, SWOW edge num: {}, merged edge num: {}".format(len(self.cn_net.edge_list), len(self.sw_net.edge_list), len(all_edge_list)))
        self.write_files(all_edge_list, args.output_csv_path, 
                        args.output_vocab_path, args.output_relation_path)

    def build_net(self, data_path, reader_cls, dataset):
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

    def write_files(self, cue_responses_relation,
                    output_csv_path, output_vocab_path, output_relation_path):
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
            for word in cpnet_vocab:
                fout.write(word + '\n')

        relation_list = list(relation_vocab)
        with open(output_relation_path, 'w') as fout:
            for word in relation_list:
                fout.write(word + '\n')

        print(f'extracted %d triples to %s'%(cnt, output_csv_path))
        print(f'extracted %d concpet vocabulary to %s'%(len(cpnet_vocab), output_vocab_path))
        print(f'extracted %d relatinos to %s'%(len(relation_list), output_relation_path))
        print()

if __name__=='__main__':
    parser= argparse.ArgumentParser()
    parser.add_argument('--conceptnet_source_file',type=str, default='./data/cn100k/cn100k_train_valid_test.txt')
    parser.add_argument('--swow_source_file', type=str, default='./data/swow/swow_triple_freq2.filter')
    parser.add_argument('--input_order', type=str, default="rht")
    parser.add_argument('--output_csv_path', type=str)
    parser.add_argument('--output_vocab_path', type=str)
    parser.add_argument('--output_relation_path', type=str)

    args= parser.parse_args()

    Merge(args)

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

