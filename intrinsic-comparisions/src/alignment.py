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
from tqdm import tqdm

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import graph
from graphs_data import GraphsData
from ground import Ground
from reader import  ConceptNetTSVReader, SwowTSVReader, Triple2GraphReader
from utils_writer import *
import logging as logger
logger.getLogger().setLevel(logger.INFO)

class Alignment(Ground):

    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.net_cn_test = None
        self.net_cn_valid = None
        self.net_sw_test = None
        self.net_sw_valid = None

    def generate_aligned_datasets(self):
        gap_triples = 1e4
        gap_nodes = 1e4
        sample_try=0
        while gap_triples>300 or gap_nodes>100:
            sample_try+=1
            print("Try {} times sampling...".format(sample_try))
            net_cn_test, net_cn_valid  =self.sample_test_valid(self.overlap_edges_set, self.overlap_edges_nodes, self.args.sample_node_size)
            gap_triples = abs(len(net_cn_test.edge_set) - len(net_cn_valid.edge_set))
            gap_nodes = abs(len(net_cn_test.graph.node2id.keys()) - len(net_cn_valid.graph.node2id.keys()))


        #self.plot_sampled_graph_statistics(net_cn_test, net_cn_valid)
        #write_relation_triples(self.overlap_edges_set, out_dir+"/rel_triples_overlap12", input_order='rht')
        self.net_cn_test = net_cn_test
        self.net_cn_valid = net_cn_valid

        self.net_sw_test = self.swow_test_set(self.net_cn_test, 'test')
        self.net_sw_valid = self.swow_test_set(self.net_cn_valid, 'valid')

        cn_edges_num_ori = len(self.net_cn.edge_set)
        sw_edges_num_ori = len(self.net_sw.edge_set)

        print("before removing, cn edge num: {}".format(len(self.net_cn.edge_set)))
        print("before removing, sw edge num: {}".format(len(self.net_sw.edge_set)))

        self.remove_sampled_from_train(self.net_cn_test, self.net_sw_test)
        self.remove_sampled_from_train(self.net_cn_valid, self.net_sw_valid)

        print("after removing, cn edge num: {}".format(len(self.net_cn.edge_set)))
        print("after removing, sw edge num: {}".format(len(self.net_sw.edge_set)))

        if self.args.match_mode =="hard":
            self.check_data(self.net_cn, self.net_cn_test, self.net_cn_valid, cn_edges_num_ori, 'equal')
            self.check_data(self.net_sw, self.net_sw_test, self.net_sw_valid, sw_edges_num_ori, 'less')

        print("Finish sampling .... ")

    def check_data(self, net, test_net, valid_net, edges_num_ori, metric):
        edges_num_cur = len(net.edge_set) + len(test_net.edge_set) + len(valid_net.edge_set)
        if metric == 'equal':
            assert edges_num_cur == edges_num_ori, "current: {}, ori: {}".format(edges_num_cur, edges_num_ori)
        elif metric == 'less':
            assert edges_num_cur <= edges_num_ori, "current: {}, ori: {}".format(edges_num_cur, edges_num_ori)

    def remove_sampled_from_train(self, net_cn, net_sw):
        self.overlap_nodes -=net_cn.graph.node2id.keys()
        self.overlap_edges_set -=net_cn.edge_set

        self.net_cn.edge_set -=net_cn.edge_set
        self.net_sw.edge_set -=net_sw.edge_set

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

    def sample_test_valid(self, overlap_edges_set, overlap_edges_nodes, sample_node_size=3000):
        '''sample overlap nodes and overlap triples'''
        sampled_nodes = list(random.sample(overlap_edges_nodes, sample_node_size))
        random.shuffle(sampled_nodes)
        random.shuffle(list(overlap_edges_set))

        sample_nodes_test_pool = sampled_nodes[:int(sample_node_size/2)]
        sample_nodes_valid_pool = sampled_nodes[int(sample_node_size/2):]
        #sample_nodes_test_pool=set()
        #sample_nodes_valid_pool= set()
        #for i in range(0, len(sampled_nodes), 2):
        #    sample_nodes_valid_pool.add(sampled_nodes[i])
        #    sample_nodes_test_pool.add(sampled_nodes[i+1])

        sampled_edges_test = set()
        sampled_edges_valid = set()
        sampled_nodes_test = set()
        sampled_nodes_valid = set()

        for triple in overlap_edges_set:
            if self.input_order == 'rht':
                rel, src, tgt = triple
            elif self.input_order == 'rhtw':
                rel, src, tgt, weight = triple

            if src in sample_nodes_valid_pool and tgt in sample_nodes_valid_pool:
                sampled_edges_valid.add(triple)
                sampled_nodes_valid.update([src, tgt])
            elif src in sample_nodes_test_pool and tgt in sample_nodes_test_pool:
                sampled_edges_test.add(triple)
                sampled_nodes_test.update([src, tgt])

        assert sampled_nodes_valid.isdisjoint(sampled_nodes_test)
        assert sampled_edges_valid.isdisjoint(sampled_edges_test)
        if self.args.match_mode == 'hard':
            assert sampled_edges_valid.issubset(self.net_cn.edge_set), f"{sampled_edges_valid.difference(self.net_cn.edge_set)}"
            assert sampled_edges_test.issubset(self.net_cn.edge_set), f"{sampled_edges_test.difference(self.net_cn.edge_set)}"

        net_cn_test = Triple2GraphReader(sampled_edges_test, 'test', input_order = self.args.input_order)
        net_cn_valid = Triple2GraphReader(sampled_edges_valid, 'valid', input_order = self.args.input_order)

        return net_cn_test, net_cn_valid

    def swow_test_set(self, net, prefix='test'):
        sampled_edges_sw = set()
        sampled_edges_sw_list =list()
        n = net.graph.edgeCount
        count = 0    
        def sample_relation(edge_sw):
            i = 0
            while i < len(edge_sw):
                i+=1
                edge_sampled = random.sample(edge_sw, 1)
                rel = edge_sampled[0][0]
                weight = edge_sampled[0][1]
                triple = (rel, src, tgt, weight)
                if triple not in sampled_edges_sw:
                    return triple
 
        for i, edge in enumerate(tqdm(net.graph.iter_edges(), total=n,\
                                      desc=f'retrieving swow {prefix}')):
            src = edge.src.name
            tgt = edge.tgt.name

            edge_sw = self.net_sw.graph.find_rel_by_node_name(src, tgt, weight=True)
            #ensure CN and SW have same (head, tail) pairs

            if edge_sw is None:                    
                print(edge.relation.name, src, tgt, edge.weight)
                net.edge_set.remove((edge.relation.name, src, tgt, edge.weight))
            else:
                triple = sample_relation(edge_sw)
                if triple not in sampled_edges_sw and triple is not None:
                    sampled_edges_sw.add(triple)
                    sampled_edges_sw_list.append(triple)
                else:
                    count +=1
                    net.edge_set.remove((edge.relation.name, src, tgt, edge.weight))
                #In SWOW, the corelaiton between forward and backward are strong,
                #if ('backwardassociated', 'work', 'boss', 19.0) is slected,
                #remove ('forwardassociated', 'work', 'boss', 19.0)  and ('mutualassociated', 'work', 'boss', 19.0)
                for (rel, weight) in edge_sw:
                    triple = (rel, src, tgt, float(weight))
                    if triple in self.net_sw.edge_set:
                        self.net_sw.edge_set.remove(triple)

        net_sampled = Triple2GraphReader(sampled_edges_sw, prefix, input_order = self.args.input_order)

        def check_triples(net, net_sampled):
            net_ent_pairs= set()
            net_sampled_ent_pairs = set()

            for tuple in net.edge_set:
                net_ent_pairs.add((tuple[1], tuple[2]))

            for tuple in net_sampled.edge_set:
                net_sampled_ent_pairs.add((tuple[1], tuple[2]))
            net_unique = net_ent_pairs - net_sampled_ent_pairs
            sampled_unique = net_sampled_ent_pairs - net_ent_pairs
            print("net_unique: {}".format(net_unique))
            print("sampled_unique: {}".format(sampled_unique))
        check_triples(net, net_sampled)

        #print("after - before : {}".format( sampled_edges_sw - net_sampled.edge_set ))
        #print("Number of edge in net: {}".format(n))
        #print(f"Remove {count} edges from net")
        #print("Number of sampled_edges_sw_list: {}, counter: {}".format(len(sampled_edges_sw_list), Counter(sampled_edges_sw_list).most_common(20)))
        #print("Number of sampled_edges_sw: {}".format(len(sampled_edges_sw)))

        #print("Number of edge_set of net_sampled: {}, edgeCount: {}".format(len(net_sampled.edge_set), net_sampled.graph.edgeCount))
        #print("net_sampled.edge_set_list: {}, {}".format(len(net_sampled.edge_set_list), Counter(net_sampled.edge_set_list).most_common(20)))
        # i = 0
        # for x in net_sampled.edge_set:
        #     i = i+1
        #     print(x)
        #     if i>10:
        #         break
        # i = 0
        # print("-"*50)
        # for x in self.net_sw.edge_set:
        #     i = i+1
        #     print(x)
        #     if i>10:
        #         break
        return net_sampled

    def plot_sampled_graph_statistics(self, test_net, valid_net):

        def sampled_graph_statistics(sampled_net, columns):
            degree_list= list()
            for node_name in sampled_net.graph.node2id.keys():
                degree = sampled_net.graph.nodes[sampled_net.graph.node2id[node_name]].get_degree()
                cn_degree = self.net_cn.graph.nodes[self.net_cn.graph.node2id[node_name]].get_degree()
                sw_degree = self.net_sw.graph.nodes[self.net_sw.graph.node2id[node_name]].get_degree()
                degree_list.append([node_name, degree, cn_degree, sw_degree])

            df = pd.DataFrame(degree_list, columns=columns)
            return df


        test_df = sampled_graph_statistics(test_net, columns=['node_test', 'test_degree', 'cn_degree', 'sw_degree'])
        valid_df = sampled_graph_statistics(valid_net, columns=['node_valid', 'valid_degree', 'cn_degree', 'sw_degree'])

        f,  ax = plt.subplots(3, 2, figsize=(10, 10), sharey='row', sharex='row')

        sns.distplot(test_df["test_degree"], hist=True, kde=False, norm_hist=False, rug=False, label="test_degree", ax=ax[0,0], color='g')
        sns.distplot(test_df["cn_degree"], hist=True, kde=False, norm_hist=False, rug=False, label="cn_degree", ax=ax[1,0], color='r')
        sns.distplot(test_df["sw_degree"], hist=True, kde=False, norm_hist=False, rug=False, label="sw_degree", ax=ax[2,0], color='b')

        sns.distplot(valid_df["valid_degree"], hist=True, kde=False, norm_hist=False, rug=False, label="valid_degree", ax=ax[0,1], color='g' )
        sns.distplot(valid_df["cn_degree"], hist=True, kde=False, norm_hist=False, rug=False, label="cn_degree", ax=ax[1,1], color='r')
        sns.distplot(valid_df["sw_degree"], hist=True, kde=False, norm_hist=False, rug=False, ax=ax[2,1], color='b')
        #ax.set_ylabel(ylabel)
        #ax.set_xlabel("Degree")
        #ax.lines[1].set_linestyle("--")
        plt.legend()
        #plt.supertitle("Ent degrees in various graphs.", fontsize=14)
        plt.show()

        plt.savefig('log/{}.png'.format("ent_degree_distribution"), format='png')


if __name__=='__main__':
    parser= argparse.ArgumentParser()
    parser.add_argument('--conceptnet_source_file',type=str, default='./data/cn100k/cn100k_train_valid_test.txt')
    parser.add_argument('--swow_source_file',type=str, default='./data/swow/swow_triple_freq2.filter')
    parser.add_argument('--align_dir',type=str, default="data/alignment/C_S_V0.1")
    parser.add_argument('--input_order',type=str, default="rht")

    args= parser.parse_args()

    Alignment(args)

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

