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

from reader import  ConceptNetTSVReader, SwowTSVReader, Triple2GraphReader
from utils_writer import *
import graph
from ground import Ground
from alignment import Alignment
import logging as logger
logger.getLogger().setLevel(logger.INFO)

class GraphsData(object):
    def __init__(self, args):
        self.input_order = args.input_order
        self.write_non_overlap = args.write_non_overlap
        self.match_mode = args.match_mode
        self.add_isphrase_rel = args.add_isphrase_rel
        self.isphrase_rel_name = 'isphrase'
        self.debug = args.debug


        self.sw_net = self.build_net(args.swow_source_file, ConceptNetTSVReader, "SWOW")
        #self.sw_net = self.build_net(args.swow_source_file, SwowTSVReader)

        if self.match_mode !='hard':
            self.concept_to_lemma = self.entity_lemma_dict(list(self.sw_net.graph.node2id.keys()))
        self.cn_net = self.build_net(args.conceptnet_source_file, ConceptNetTSVReader, "ConceptNet")


    def entity_lemma_dict(self, concepts):
        concept_to_lemma = dict()

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

    def lemmatize(self, nlp, concept):
        doc = nlp(concept.replace("_", " "))
        lcs = set()
        lcs.add("_".join([token.lemma_ for token in doc]))  # all lemma
        return lcs


class AlignConceptNetSWOW(GraphsData):
    def __init__(self, args):
        super().__init__(args)
        self.overlap = Ground()
        self.overlap.overlap_edges_nodes()

        self.alignment = Alignment(self.overlap)
        #self.write_files()

    def write_files(self):
        write_eval_to_files(self.align_dir, self.alignment.test_net, 'test')
        write_eval_to_files(self.align_dir, self.alignment.valid_net, 'valid')
        #write_train_to_files(out_dir, self.cn_net, self.sw_net, self.overlap_edges_set,\
        #                        self.overlap_nodes, self.overlap_rels)



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
    parser.add_argument('--align_dir',type=str, default="data/alignment/C_S_V0.1")

    args= parser.parse_args()

    AlignConceptNetSWOW(args)

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

