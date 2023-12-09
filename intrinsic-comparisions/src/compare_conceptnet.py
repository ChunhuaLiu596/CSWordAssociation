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

import nltk
ps=nltk.stem.PorterStemmer()


def build_network(dataset, data_path, reader_cls, inp_order):
    network = reader_cls(dataset, inp_order)
    network.read_network(data_path)
    network.print_summary()
    return network

def get_node_name(edge, direction="src->tgt"):
        #triple_old = (edge.relation.name, edge.src.name, edge.tgt.name)
    if direction=="src->tgt":
        src = edge.src.name
        tgt = edge.tgt.name
    elif direction=="tgt->src":
        src = edge.tgt.name
        tgt = edge.src.name
    relation = edge.relation.name
    return src, tgt, relation 

def compare_nets(args):
    net_cn1 = build_network('cn100k', args.old_file, ConceptNetTSVReader, 'rht')
    net_cn2 = build_network('cn5.7', args.new_file, ConceptNetTSVReader, 'rht')

    print("net_cn1 relations: {}".format(net_cn1.graph.relation2id))
    print("net_cn2 relations: {}".format(net_cn2.graph.relation2id))
    #import sys 
    #sys.exit()

    same=set()
    different = set()
    not_exist = set() 
    for i, edge in enumerate(net_cn1.graph.iter_edges()):
        src, tgt, relation  = get_node_name(edge, "src->tgt")
        rels = net_cn2.graph.find_rel_by_node_name(src, tgt)
        if rels is not None:
            for rel in rels:
                if relation == rel:
                    same.add((rel, src, tgt))
                else:
                    different.add((relation, src, tgt, "  ||  ",  rel, src, tgt))
        else:
            not_exist.add((relation, src, tgt, "  ||  ", "NONE", src, tgt))
    write_files(same, different, not_exist)

def write_files(same, different, not_exist):
    print("Same triple num: {}, different triple num: {}, not exist in new conceptnet num: {}".format(len(same), len(different), len(not_exist)))
    print("Inconsistent triple")
    with open(args.out_path_same, 'w') as f:
        for x in same:
            f.write("{}\n".format(x))

    with open(args.out_path_diff, 'w') as f:
        for x in different:
            f.write("{}\n".format(x))

    with open(args.out_path_notexist, 'w') as f:
        for x in not_exist:
            f.write("{}\n".format(x))


if __name__=='__main__':
    parser= argparse.ArgumentParser()
    parser.add_argument('--dataset',type=str, default='None')
    parser.add_argument('--out_path_same',type=str, default='/home/chunhua/Commonsense/OpenEntitiyAlignment/conceptnet-swow-data/log/same_all.txt')
    parser.add_argument('--out_path_diff',type=str, default='/home/chunhua/Commonsense/OpenEntitiyAlignment/conceptnet-swow-data/log/diff_all.txt')
    parser.add_argument('--out_path_notexist',type=str, default='/home/chunhua/Commonsense/OpenEntitiyAlignment/conceptnet-swow-data/log/not-exist_all.txt')
    parser.add_argument('--old_file',type=str, default='/home/chunhua/Commonsense/OpenEntitiyAlignment/datasets/C_S_V0_leakage/rel_triples_test')
    parser.add_argument('--new_file',type=str, default='/home/chunhua/Commonsense/OpenEntitiyAlignment/datasets/C_S_V0_leakage/rel_triples_valid')
    args= parser.parse_args()

    compare_nets(args)
