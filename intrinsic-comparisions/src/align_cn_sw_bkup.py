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

class Alignment(object):
    def __init__(self, args):
        #self.sw_net = self.build_net(args.swow_source_file, SwowTSVReader)
        self.input_order = args.input_order
        self.sw_net = self.build_net(args.swow_source_file, ConceptNetTSVReader, "SWOW") 
        self.cn_net = self.build_net(args.conceptnet_source_file, ConceptNetTSVReader, "ConceptNet")

        self.overlap_edges_set, self.overlap_edges_nodes, self.overlap_rels= self.find_overlap_edges()
        self.overlap_nodes = self.find_overlap_nodes()
        self.generate_aligned_datasets(args.align_dir)

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

    def find_overlap_nodes(self):
        sw_nodes = self.sw_net.graph.node2id.keys()
        cn_nodes = self.cn_net.graph.node2id.keys()
        overlap_nodes = set(sw_nodes).intersection(set(cn_nodes))
        print("overlap_nodes num : {}".format(len(overlap_nodes)))
        return overlap_nodes

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
            rel, src, tgt = triple
            if src in sample_nodes_valid_pool and tgt in sample_nodes_valid_pool:
                sampled_edges_valid.add(triple)
                sampled_nodes_valid.update([src, tgt])
            elif src in sample_nodes_test_pool and tgt in sample_nodes_test_pool:
                sampled_edges_test.add(triple)
                sampled_nodes_test.update([src, tgt])

        assert sampled_nodes_valid.isdisjoint(sampled_nodes_test)
        assert sampled_edges_valid.isdisjoint(sampled_edges_test)

        assert sampled_edges_valid.issubset(self.cn_net.edge_set)
        assert sampled_edges_test.issubset(self.cn_net.edge_set)

        test_net = Triple2GraphReader(sampled_edges_test, 'test')
        valid_net = Triple2GraphReader(sampled_edges_valid, 'valid')

        return test_net, valid_net

    def generate_aligned_datasets(self, out_dir):
        gap_triples = 1e4
        gap_nodes = 1e4
        sample_try=0
        while gap_triples>50 or gap_nodes>10:
            sample_try+=1
            print("Try {} times sampling...".format(sample_try))
            test_net, valid_net  =self.sample_test_valid(self.overlap_edges_set, self.overlap_edges_nodes)
            gap_triples = abs(len(test_net.edge_set) - len(valid_net.edge_set))
            gap_nodes = abs(len(test_net.graph.node2id.keys()) - len(valid_net.graph.node2id.keys()))

        self.plot_sampled_graph_statistics(test_net, valid_net)
        write_relation_triples(self.overlap_edges_set, out_dir+"/rel_triples_overlap12", input_order='rht')

        def remove_sampled_from_train(net):
            self.overlap_nodes -=net.graph.node2id.keys()
            self.overlap_edges_set -= net.edge_set

            self.cn_net.edge_set -=net.edge_set
            self.sw_net.edge_set -=net.edge_set

        cn_edges_num_ori = len(self.cn_net.edge_set)
        sw_edges_num_ori = len(self.sw_net.edge_set)

        remove_sampled_from_train(test_net)
        remove_sampled_from_train(valid_net)

        def check_data(net, edges_num_ori):
            edges_num_cur = len(net.edge_set) + len(test_net.edge_set) + len(valid_net.edge_set)
            assert edges_num_cur == edges_num_ori, "current: {}, ori: {}".format(edges_num_cur, edges_num_ori)
        check_data(self.cn_net, cn_edges_num_ori)
        check_data(self.sw_net, sw_edges_num_ori)

        print("Finish sampling .... ")
        write_eval_to_files(out_dir, test_net, 'test')
        write_eval_to_files(out_dir, valid_net, 'valid')
        write_train_to_files(out_dir, self.cn_net, self.sw_net, self.overlap_edges_set,\
                                self.overlap_nodes, self.overlap_rels)

    def plot_sampled_graph_statistics(self, test_net, valid_net):

        def sampled_graph_statistics(sampled_net, columns):
            degree_list= list()
            for node_name in sampled_net.graph.node2id.keys():
                degree = sampled_net.graph.nodes[sampled_net.graph.node2id[node_name]].get_degree()
                cn_degree = self.cn_net.graph.nodes[self.cn_net.graph.node2id[node_name]].get_degree()
                sw_degree = self.sw_net.graph.nodes[self.sw_net.graph.node2id[node_name]].get_degree()
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

    def compare_old_version(self, new_triples):
        old_triples = set()
        with open('./data/swow/conceptnet_swow_edges.overlap') as f:
            data = f.readlines()

        for inst in data:
            inst = inst.strip()
            if inst:
                inst = inst.split('\t')
                rel, src, tgt = inst
                weight = 1.0
                src = src.lower()
                tgt = tgt.lower()
                old_triples.add((rel, tgt, src))

        diff = old_triples.difference(new_triples)
        print("old - new, total: {}".format(len(diff)))
        #for x in diff:
        #    print("old ", x)

        diff = new_triples.difference(old_triples)
        print("new - old total: {}".format(len(diff)))
        #for x in diff:
        #    print("new ", x)

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

