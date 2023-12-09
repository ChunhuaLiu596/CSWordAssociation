from collections import Counter
import argparse
import numpy as np
import sys
import os
import json
import time
import random
import pdb
import spacy
import datetime
# import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tqdm import tqdm

from reader import  ConceptNetTSVReader, SwowTSVReader, Triple2GraphReader
from utils_writer import *
import graph

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from utils import check_path
from generate_openke_data import build_data_4_OpenKE
from utils import check_path

import logging as logger
logger.getLogger().setLevel(logger.INFO)

PAD_TOKEN="_PAD"
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner', 'textcat'])

class GraphData(object):
    def __init__(self, args):
        self.input_order = args.input_order
        self.debug = args.debug

        self.net = self.build_net(args.source_file, ConceptNetTSVReader, args.kg_name)

        if args.write_files:
            self.write_files(args.out_dir, args.kg_name, args.input_order)

    def build_net(self, data_path, reader_cls, kg_name):
        # print("Loading {} ...".format(kg_name))
        network = reader_cls(kg_name, self.input_order)
        network.read_network(data_path)
        network.print_summary()
        return network

    def write_files(self, out_dir, kg_name, input_order):
        '''
        '''
        check_path(out_dir)
        #--------------- write cn/sw set -----------
        print("-"*60)
        dir = out_dir + kg_name + "/"
        print(dir)
        check_path(dir)

        train_path = dir+'4tuple.train.csv' 
        
        concept_path = dir+'concept.txt' 
        concept_degree_path = dir+'concept_degree.txt' 
        relation_path =  dir + 'relation.txt'

        write_relation_triples(self.net.edge_set, train_path, input_order)

        #for OpenKE training
        nodes = set(self.net.graph.node2id.keys())
        write_nodes(nodes, concept_path)

        #get node degree
        node_degrees_sorted = self.map_node2degree(nodes)
        write_node2degree(node_degrees_sorted, concept_degree_path)

        relations = set(self.net.graph.relation2id.keys())
        write_nodes(relations, relation_path)
        
        build_data_4_OpenKE(train_path, concept_path, relation_path, output_folder=dir)
    
    def map_node2degree(self, nodes):
        node_list = self.net.graph.iter_nodes()
        node_degrees = {node.name: node.get_degree() for node in node_list}
        # node_degrees_sorted = [(node, node_degrees.get(node, 0)) for node in nodes]
        node_degrees_sorted = [(node, node_degrees.get(node, 1)) for node in nodes]
        return node_degrees_sorted

    def map_loadnode2degree(self, concept_path, concept_degree_path):
        nodes = []
        with open (args.concept_path, "r") as fin:
            lines = fin.readlines()
            nodes.extend([line.strip() for line in lines])
        node_degrees_sorted = self.map_node2degree(nodes)
        write_node2degree(node_degrees_sorted, concept_degree_path)

if __name__=='__main__':
    parser= argparse.ArgumentParser()
    parser.add_argument('--source_file',type=str, default='./data/cn100k/cn100k_train_valid_test.txt')
    parser.add_argument('--kg_name', type=str, default="cpnet")
    parser.add_argument('--input_order', type=str, default="rht")

    parser.add_argument('--out_dir',type=str, default="data/alignment/C_S_V0.1")
    parser.add_argument('--write_files', action='store_true')


    parser.add_argument('--concept_path', type=str)
    parser.add_argument('--concept_degree_path', type=str)

    parser.add_argument('--output_relation_path', type=str)

    parser.add_argument('--output_csv_path_non_overlap', type=str)
    parser.add_argument('--output_vocab_path_non_overlap', type=str)
    parser.add_argument('--output_relation_path_non_overlap', type=str)
    parser.add_argument('--debug', action='store_true')

    args= parser.parse_args()

    data = GraphData(args)
    if args.concept_path is not None:
        data.map_loadnode2degree(args.concept_path, args.concept_degree_path)
    # map_loadnode2degree(args.concept_path, args.concept_degree_path)

