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

    src_stemmed = ps.stem(src) #swow has plural, cn only has singular
    tgt_stemmed = ps.stem(tgt)
    return src, tgt, src_stemmed, tgt_stemmed


def remove_leakage(train_net, eval_net):
    count = [0]

    def judge_edges(eval_net, src, tgt, src_stemmed, tgt_stemmed, stemmed=False):
        if stemmed:
            rel_cn = eval_net.graph.find_rel_by_node_name(src_stemmed, tgt_stemmed)
        else:
            rel_cn = eval_net.graph.find_rel_by_node_name(src, tgt)

        if rel_cn is not None:
            for rel in rel_cn:
                triple = (rel, src, tgt)
                if triple in train_net.edge_set:
                    train_net.edge_set.discard(triple)
                    print(triple)
                    count[0]+=1

                triple = ("FW-REL", src, tgt)
                if triple in train_net.edge_set:
                    train_net.edge_set.discard(triple)
                    count[0]+=1
                    print(triple)

                triple = ("FW-REL", tgt, src)
                if triple in train_net.edge_set:
                    train_net.edge_set.discard(triple)
                    count[0]+=1
                    print(triple)


    for i, edge in enumerate(train_net.graph.iter_edges()):
        src, tgt, src_stemmed, tgt_stemmed = get_node_name(edge, "src->tgt")
        judge_edges(eval_net, src, tgt, src_stemmed, tgt_stemmed, True)
        judge_edges(eval_net, src, tgt, src_stemmed, tgt_stemmed, False)

        src, tgt, src_stemmed, tgt_stemmed = get_node_name(edge, "tgt->src")
        judge_edges(eval_net, src, tgt, src_stemmed, tgt_stemmed, True)
        judge_edges(eval_net, src, tgt, src_stemmed, tgt_stemmed, False)


    print("Total {} leakage".format(count))
    return train_net


def read_nets(args):
    net_sw = build_network(args.dataset, args.swow_file, ConceptNetTSVReader, 'hrt')
    ori_triple_num= len(net_sw.edge_set)
    net_valid = build_network(args.dataset, args.valid_file, ConceptNetTSVReader, 'hrt')
    net_test = build_network(args.dataset, args.test_file, ConceptNetTSVReader, 'hrt')

    print("Remove valid leakage")
    net_sw = remove_leakage(net_sw, net_valid)

    print("Remove test leakage")
    remove_leakage(net_sw, net_test)

    write_relation_triples(net_sw.edge_set, args.out_path +'/rel_triples_2', 'rht')
    print("Original triple number: {}".format(ori_triple_num))
    print("Now triple number: {}".format(len(net_sw.edge_set)))


if __name__=='__main__':
    parser= argparse.ArgumentParser()
    parser.add_argument('--dataset',type=str, default='None')
    parser.add_argument('--out_path',type=str, default='/home/chunhua/Commonsense/OpenEntitiyAlignment/datasets/C_S_V0')
    parser.add_argument('--test_file',type=str, default='/home/chunhua/Commonsense/OpenEntitiyAlignment/datasets/C_S_V0_leakage/rel_triples_test')
    parser.add_argument('--valid_file',type=str, default='/home/chunhua/Commonsense/OpenEntitiyAlignment/datasets/C_S_V0_leakage/rel_triples_valid')
    parser.add_argument('--swow_file',type=str, default='/home/chunhua/Commonsense/OpenEntitiyAlignment/datasets/C_S_V0_leakage/rel_triples_2')
    args= parser.parse_args()


    read_nets(args)

