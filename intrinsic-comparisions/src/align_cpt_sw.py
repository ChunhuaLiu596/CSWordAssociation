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
from graphs_data import GraphsData


currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from utils import check_path
from generate_openke_data import build_data_4_OpenKE

import logging as logger
logger.getLogger().setLevel(logger.INFO)

class AlignConceptNetSWOW(object):
    def __init__(self, args):
        
        self.args = args
        self.out_dir = self.args.out_dir
        self.align_dir = self.args.out_dir + '/align_cn_sw' 
        check_path(self.align_dir)
        args.align_dir = self.align_dir

        self.align = Alignment(args)
        self.align.generate_aligned_datasets()

        self.write_files(self.out_dir, self.args.input_order, swow_prefix=args.swow_prefix)

    def write_files(self, out_dir, input_order, swow_prefix):
        '''
        CN:
        SW:
        Align:
        '''
        check_path(out_dir)


        #--------------- write cn set -----------
        print("-"*60)
        cn_dir = out_dir + '/cpnet'
        check_path(cn_dir)
        train_path = cn_dir+'/conceptnet.train.csv' 
        valid_path = cn_dir+'/conceptnet.valid.csv' 
        test_path= cn_dir+'/conceptnet.test.csv'
        concept_path = cn_dir+'/concept.txt' 
        concept_degree_path = cn_dir+'/concept_degree.txt' 
        relation_path =  cn_dir + '/relation.txt'
        write_relation_triples(self.align.net_cn.edge_set, train_path, input_order)
        write_relation_triples(self.align.net_cn_test.edge_set, test_path, input_order)
        write_relation_triples(self.align.net_cn_valid.edge_set, valid_path, input_order)

        #for OpenKE training
        nodes_cn = set(self.align.net_cn.graph.node2id.keys()) | set(self.align.net_cn_test.graph.node2id.keys()) | set(self.align.net_cn_valid.graph.node2id.keys())
        write_nodes(nodes_cn, concept_path)

        relations_cn = set(self.align.net_cn.graph.relation2id.keys()) | set(self.align.net_cn_test.graph.relation2id.keys()) | set(self.align.net_cn_valid.graph.relation2id.keys())
        write_nodes(relations_cn, relation_path)
        
        build_data_4_OpenKE(train_path, concept_path, relation_path, output_folder=cn_dir, valid_path, test_path, )


        #--------------- write sw set -----------
        print("-"*60)
        sw_dir = out_dir + f'/{swow_prefix}'
        check_path(sw_dir)
        
        train_path = sw_dir+'/swow.train.csv' 
        valid_path = sw_dir+'/swow.valid.csv' 
        test_path= sw_dir+'/swow.test.csv'
        concept_path = sw_dir+'/concept.txt' 
        relation_path =  sw_dir + '/relation.txt'
        write_relation_triples(self.align.net_sw.edge_set, train_path, input_order)
        write_relation_triples(self.align.net_sw_test.edge_set, test_path,  input_order)
        write_relation_triples(self.align.net_sw_valid.edge_set, valid_path, input_order)

        nodes_sw = set(self.align.net_sw.graph.node2id.keys()) | set(self.align.net_sw_test.graph.node2id.keys()) | set(self.align.net_sw_valid.graph.node2id.keys())
        write_nodes(nodes_sw, concept_path)

        relations_sw = set(self.align.net_sw.graph.relation2id.keys()) | set(self.align.net_sw_test.graph.relation2id.keys()) | set(self.align.net_sw_valid.graph.relation2id.keys())
        write_nodes(relations_sw, relation_path) 
        #write alignment set
        print("-"*60)

        build_data_4_OpenKE(train_path, concept_path, relation_path, output_folder=sw_dir, valid_path, test_path)
        
        #--------------- write alignment entities and relations -----------
        align_dir = self.align_dir
        write_links(self.align.overlap_nodes, align_dir  + '/ent_links_train', link_type='nodes')
        write_links(self.align.net_cn_valid.graph.node2id.keys(), align_dir + '/ent_links_valid', link_type='nodes')
        write_links(self.align.net_cn_test.graph.node2id.keys(), align_dir + '/ent_links_test', link_type='nodes')
        write_links(self.align.overlap_rels, outpath=align_dir+'/rel_links', link_type='relation')


if __name__=='__main__':
    parser= argparse.ArgumentParser()
    parser.add_argument('--conceptnet_source_file',type=str, default='./data/cn100k/cn100k_train_valid_test.txt')
    parser.add_argument('--swow_source_file', type=str, default='./data/swow/swow_triple_freq2.filter')
    parser.add_argument('--input_order', type=str, default="rht")

    parser.add_argument('--out_dir',type=str, default="data/alignment/C_S_V0.1")
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

    parser.add_argument('--add_cn_triples_to_swow', action='store_true')
    parser.add_argument('--swap_retrieval', action='store_true', help='swap head and tail to retrieve edge when finding overlap edges')
    parser.add_argument('--sample_node_size', type=int, default=3000, help='node number for test set')
    parser.add_argument('--swow_prefix', type=str, default='swow_3rel_freq1', help='swow kg types')
    parser.add_argument('--align_dir', type=str)
    
    args= parser.parse_args()

    AlignConceptNetSWOW(args)
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

