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
#import seaborn

import graph
from reader import  ConceptNetTSVReader, SwowTSVReader, Triple2GraphReader
from utils_writer import *
import logging as logger
logger.getLogger().setLevel(logger.INFO)

def get_statistics(dataset, data_path, reader_cls, inp_order, out_network=False):
        network = reader_cls(dataset, inp_order)
        network.read_network(data_path)
        network.print_summary()
        edges_num = network.graph.edgeCount
        nodes_num = len(network.graph.nodes)
        relations_num = network.relations_num
        density = network.density
        degree = network.degree
        ent_entropy = network.ent_entropy
        rel_entropy = network.rel_entropy

        words_per_concept, mutli_word_ratio, len2cnt = network.graph.node_tokens_len_statistics()
        print("multiword concepts (%): {:.2f}".format(mutli_word_ratio*100))
        print("words per concept (avg) {:.2f}".format(words_per_concept))
        print("len-num distribution: {}".format(len2cnt))

        print_r_distribution=False
        if print_r_distribution:
            r_distribution = network.graph.find_relation_distribution()
            print("r_distribution")
            for x in r_distribution:
                print("{}\t{}".format(x[0],x[1]))

        if inp_order=='rhtws':
            source2weight = network.print_source_weight()

        if out_network:
            return network
        else:
            return (edges_num, nodes_num, relations_num, density, degree, ent_entropy, rel_entropy)

def print_statistics(stis):
    print("edges_num \tnodes_num\trelations_num\tdensity\tdegree\tent_entropy\trel_entropy")
    out = "{}\t{}\t{}\t{}\t{}\t{}\t{}".format(stis[0], stis[1], stis[2], stis[3], stis[4],stis[5])
    print(out)

def get_neighborhoods(network, node_name):
    print("edge count {}".format(network.graph.edgeCount))
    neighborhoods = network.graph.find_neighborhoods(node_name)
    print("neighborhoods of {}".format(node_name))
    for n in neighborhoods:
        print("\t".join(n))

#def visualize_distribution():

if __name__=='__main__':
    parser= argparse.ArgumentParser()
    parser.add_argument('--dataset',type=str, default='None')
    parser.add_argument('--source_file',type=str, default='./data/cn100k/cn100k_train_valid_test.txt')
    parser.add_argument('--swow_source_file',type=str, default='./data/swow/swow_triple_freq2.filter')
    parser.add_argument('--input_order',type=str, default='rht')
    parser.add_argument('--node_name',type=str, default='wifi')
    args= parser.parse_args()

    #sw_net = get_statistics(args.swow_source_file, SwowTSVReader)
    cn_net_stis = get_statistics(args.dataset, args.source_file, ConceptNetTSVReader, args.input_order, out_network=False)
    # print(cn_net_stis)
    # print_statistics(cn_net_stis)

    # get_neighborhoods(cn_net, args.node_name)

    #sw_net = get_statistics(args.dataset, args.swow_source_file, ConceptNetTSVReader, 'hrt')
