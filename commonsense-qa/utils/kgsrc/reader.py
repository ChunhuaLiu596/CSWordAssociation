__author__ = "chaitanya"

import logging as logger
from graph import Graph

import csv
import json
import os
#import pandas as pd
import math
import random
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import Counter
import reader_utils
from tqdm import tqdm

class Reader:
    def __init__(self):
        self.density = None
        self.degree = None
        self.relations_num  = None
        self.rel_entropy = None
        self.ent_entropy = None

    def print_summary(self):

        print("\n\nGraph Summary")
        print("\nNodes: %d" % len(self.graph.nodes))
        print("Edges: %d" % self.graph.edgeCount)
        if self.graph.relation2id is not None:
            self.relations_num = len(self.graph.relation2id)
            print("Relations: %d" % len(self.graph.relation2id))
        density = self.graph.edgeCount / (len(self.graph.nodes) * (len(self.graph.nodes) - 1))
        self.density = density

        print("Density: %f" % density)

        self.print_degree()
        self.print_ent_rel_entropy()
        print("\n******************* Sample Edges *******************")

        for i, edge in enumerate(self.graph.iter_edges()):
            print(edge)
            if (i+1) % 10 == 0:
                break

        print("***************** ***************** *****************\n")

    def print_degree(self):
        #self.print_summary()
        node_list = self.graph.iter_nodes()
        node_degrees = [node.get_degree() for node in node_list]
        degree_counter = Counter(node_degrees)
        avg_degree = sum([k * v for k, v in degree_counter.items()]) / sum([v for k, v in degree_counter.items()])
        print("Average Degree: {}".format(avg_degree))
        self.degree = avg_degree
        self.node2degree=dict()
        for node in node_list:
            self.node2degree[node.name]=node.get_degree

    def print_source_weight(self):
        edges = list(self.graph.iter_edges())
        source2weight = {}
        source2count = {}
        for i, edge in enumerate(edges):
            weight, source = edge.weight, edge.source
            weight = round(weight,2)
            if source not in source2count:
                source2count[source]=1
            else:
                source2count[source]+=1

            if source not in source2weight:
                source2weight[source] = [weight]
            else:
                source2weight[source].append(weight)

        print("Source weight distribution: ")
        for source, weights in source2weight.items():
            print("source: {} {}".format(source, Counter(weights)))

        print("Source count distribution: {}".format(source2count))
        return source2weight

    def print_ent_rel_entropy(self):
        #goal: generate probibility for each relation
        #count the number of each relation appears
        rel2count = {}
        ent2count = {}
        edges = list(self.graph.iter_edges())
        for i, edge in enumerate(edges):
            src, rel, tgt = edge.src, edge.relation, edge.tgt

            if rel not in rel2count:
                rel2count[rel]=1
            else:
                rel2count[rel]=+1

            if src not in ent2count:
                ent2count[src]=1
            else:
                ent2count[src]=+1

            if tgt not in ent2count:
                ent2count[tgt]=1
            else:
                ent2count[tgt]=+1

        ent2prob = {k : (v/self.graph.edgeCount) for k,v in ent2count.items()}
        ent2entropy = {k : -v*math.log(v) for k,v in ent2prob.items()}
        self.ent_entropy = sum(v for k, v in ent2entropy.items())
        print("Entity Entropy : {}".format(self.ent_entropy))


        rel2prob = {k : (v/self.graph.edgeCount) for k,v in rel2count.items()}
        rel2entropy = {k : -v*math.log(v) for k,v in rel2prob.items()}
        self.rel_entropy = sum(v for k, v in rel2entropy.items())
        print("Relation Entropy : {}".format(self.rel_entropy))

    def gen_negative_examples(self, tgt_size=None, sampling_type="random"):

        print("Generating negative examples..")

        existing_edges = list(self.graph.iter_edges())
        existing_nodes = list(self.graph.iter_nodes())
        existing_relations = list(self.graph.iter_relations())

        if tgt_size:
            selected_edges = random.sample(existing_edges, tgt_size)
        else:
            selected_edges = existing_edges

        # Generate 3 negative samples per example
        idx = 0

        for i, edge in enumerate(selected_edges):
            src, rel, tgt = edge.src, edge.relation, edge.tgt

            rand_nodes = []
            while len(rand_nodes) != 2:
                sample = random.sample(existing_nodes, 1)
                if sample not in [src, tgt]:
                    rand_nodes.append(sample[0])

            found = False
            while not found:
                sample = random.sample(existing_relations, 1)
                if sample != rel:
                    rand_rel = sample
                    found = True

            self.add_example(src.name, rand_nodes[0].name, rel.name, 1.0, 0)
            self.add_example(rand_nodes[1].name, tgt.name, rel.name, 1.0, 0)
            self.add_example(src.name, tgt.name, rand_rel[0].name, 1.0, 0)
            idx += 3

        print("Added %d negative examples using %s sampling" %(idx, sampling_type))


class ConceptNetTSVReader(Reader):

    def __init__(self, dataset='ConceptNet', inp_order ='rht',word_vocab=None):
        print("Reading {}".format(dataset))
        self.dataset = dataset
        self.graph = Graph()
        self.rel2id = {}
        self.edge_set=set()
        self.edge_list=list()
        self.inp_order = inp_order
        if word_vocab is not None:
            for word, idx in word_vocab.items():
                #self.graph.load_node(word, idx)
                self.graph.add_node(word)

    def read_network(self, data_path):
        with open(data_path) as f:
            data = f.readlines()

        for inst in data:
            inst = inst.strip()
            if inst:
                inst = inst.split('\t')
                if self.inp_order=="rht":
                    rel, src, tgt = inst
                    weight = 1.0
                elif self.inp_order=="hrt":
                    src, rel, tgt = inst
                    weight = 1.0
                elif self.inp_order=="htr":
                    src, tgt, rel = inst
                    weight = 1.0
                elif self.inp_order=="rhtw":
                     rel, src, tgt, weight = inst
                elif self.inp_order=="rhtws":
                     rel, src, tgt, weight, source = inst
                src = src.lower()
                tgt = tgt.lower()

                if self.inp_order=="rhtws":
                    self.add_example(src, tgt, rel, float(weight), train_network=None, source=source)
                else:
                    self.add_example(src, tgt, rel, float(weight))

                if self.inp_order=="rhtw":
                    self.edge_set.add((rel, src, tgt, float(weight)))
                    self.edge_list.append((rel, src, tgt, float(weight)))
                else:
                    self.edge_set.add((rel, src, tgt))
                    self.edge_list.append((rel, src, tgt))

        self.rel2id = self.graph.relation2id

    def add_example(self, src, tgt, relation, weight, label=1, train_network=None, source=None):

        src_id = self.graph.find_node(src)
        if src_id == -1:
            src_id = self.graph.add_node(src)

        tgt_id = self.graph.find_node(tgt)
        if tgt_id == -1:
            tgt_id = self.graph.add_node(tgt)

        relation_id = self.graph.find_relation(relation)
        if relation_id == -1:
            relation_id = self.graph.add_relation(relation)
        edge = self.graph.add_edge(self.graph.nodes[src_id],
                                    self.graph.nodes[tgt_id],
                                    self.graph.relations[relation_id],
                                    label,
                                    weight,
                                    None,
                                    source)

        # add nodes/relations from evaluation graphs to training graph too
        if train_network is not None and label == 1:
            src_id = train_network.graph.find_node(src)
            if src_id == -1:
                src_id = train_network.graph.add_node(src)

            tgt_id = train_network.graph.find_node(tgt)
            if tgt_id == -1:
                tgt_id = train_network.graph.add_node(tgt)

            relation_id = train_network.graph.find_relation(relation)
            if relation_id == -1:
                relation_id = train_network.graph.add_relation(relation)

        return edge




class SwowTSVReader(Reader):

    def __init__(self, dataset='swow', word_vocab=None):
        print("Reading {}".format(dataset))
        self.dataset = dataset
        self.rel_token='FW-REL'
        self.graph = Graph()
        self.edge_set=set()
        if word_vocab is not None:
            for word, idx in word_vocab.items():
                self.graph.add_node(word)

    def read_network(self, data_path, train_network=None):
        with open(data_path) as f:
            data = f.readlines()

        for inst in data:
            inst = inst.strip()
            if inst:
                inst = inst.split('\t')
                src, tgt, weight = inst
                weight = 1.0
                src = src.lower()
                tgt = tgt.lower()
                rel= self.rel_token
                self.add_example(src, tgt, rel, float(weight))
                self.edge_set.add((rel, src, tgt))

    def add_example(self, src, tgt, relation, weight, label=1, train_network=None):

        src_id = self.graph.find_node(src)
        if src_id == -1:
            src_id = self.graph.add_node(src)

        tgt_id = self.graph.find_node(tgt)
        if tgt_id == -1:
            tgt_id = self.graph.add_node(tgt)

        relation_id = self.graph.find_relation(relation)
        if relation_id == -1:
            relation_id = self.graph.add_relation(relation)

        edge = self.graph.add_edge(self.graph.nodes[src_id],
                                    self.graph.nodes[tgt_id],
                                    self.graph.relations[relation_id],
                                    label,
                                    weight)

        return edge

    def add_true_reverse_edges(self, tgt_id, src_id, label, weight):

        #1. add reverse edge for current triples
        relation = "BW-REL"
        relation_id = self.graph.find_relation(relation)
        if relation_id == -1:
            relation_id = self.graph.add_relation(relation)

        edge = self.graph.add_edge(self.graph.nodes[src_id],
                                    self.graph.nodes[tgt_id],
                                    self.graph.relations[relation_id],
                                    label,
                                    weight)# how to get the weight of the tgt->src effectively???

        #2. add reverse edge for previous edges
        edge = self.graph.add_edge(self.graph.nodes[tgt_id],
                                    self.graph.nodes[src_id],
                                    self.graph.relations[relation_id],
                                    label,
                                    weight)



class Triple2GraphReader(Reader):

    def __init__(self, triples, dataset='ConceptNet', word_vocab=None, input_order='rht'):
        print("Reading {}".format(dataset))
        self.dataset = dataset
        self.input_order = input_order
        self.graph = Graph()
        self.rel2id = {}
        if word_vocab is not None:
            for word, idx in word_vocab.items():
                #self.graph.load_node(word, idx)
                self.graph.add_node(word)
        self.edge_set=set()
        self.edge_set_list= list()
        self.read_network(triples)
        self.print_summary()
        #self.print_degree()

    def read_network(self, data):
        n = len(data)
        for i, inst in enumerate(tqdm(data, total=n, desc=f'reading {self.dataset} network')):
            if inst:
                if self.input_order == 'rht':
                    rel, src, tgt = inst
                    weight = 1.0
                elif self.input_order == 'rhtw':
                   rel, src, tgt, weight = inst
                src = src.lower()
                tgt = tgt.lower()
                self.add_example(src, tgt, rel, float(weight))

                if self.input_order == 'rht':
                    self.edge_set.add((rel, src, tgt))
                elif self.input_order == 'rhtw':
                     self.edge_set.add((rel, src, tgt, float(weight)))
                     self.edge_set_list.append((rel, src, tgt, float(weight)))
        self.rel2id = self.graph.relation2id

    def add_example(self, src, tgt, relation, weight, label=1, train_network=None):

        src_id = self.graph.find_node(src)
        if src_id == -1:
            src_id = self.graph.add_node(src)

        tgt_id = self.graph.find_node(tgt)
        if tgt_id == -1:
            tgt_id = self.graph.add_node(tgt)

        relation_id = self.graph.find_relation(relation)
        if relation_id == -1:
            relation_id = self.graph.add_relation(relation)

        edge = self.graph.add_edge(self.graph.nodes[src_id],
                                    self.graph.nodes[tgt_id],
                                    self.graph.relations[relation_id],
                                    label,
                                    weight)

        # add nodes/relations from evaluation graphs to training graph too
        if train_network is not None and label == 1:
            src_id = train_network.graph.find_node(src)
            if src_id == -1:
                src_id = train_network.graph.add_node(src)

            tgt_id = train_network.graph.find_node(tgt)
            if tgt_id == -1:
                tgt_id = train_network.graph.add_node(tgt)

            relation_id = train_network.graph.find_relation(relation)
            if relation_id == -1:
                relation_id = train_network.graph.add_relation(relation)

        return edge




def load_data(dataset, reader_cls, data_dir, sim_relations,load_eval=True, exist_relation=True, test_prefix="test", load_vocab=False, load_vocab_file=None):
    if load_vocab:
        word_vocab = reader_utils.load_vocab(load_vocab_file)
    else:
        word_vocab =None

    train_network = reader_cls(dataset, word_vocab)
    dev_network = reader_cls(dataset, word_vocab)
    test_network = reader_cls(dataset, word_vocab)

    train_network.read_network(data_dir=data_dir, split="train")
    train_network.print_summary()
    node_list = train_network.graph.iter_nodes()
    node_degrees = [node.get_degree() for node in node_list]
    degree_counter = Counter(node_degrees)
    avg_degree = sum([k * v for k, v in degree_counter.items()]) / sum([v for k, v in degree_counter.items()])
    print("Average Degree: ", avg_degree)

    if load_eval:
        dev_network.read_network(data_dir=data_dir, split="valid", train_network=train_network)
        test_network.read_network(data_dir=data_dir, split=test_prefix, train_network=train_network)

    if not load_vocab:
        word_vocab = train_network.graph.node2id
        #if not os.path.exists(load_vocab_file):
        reader_utils.write_vocab(word_vocab, load_vocab_file)

    # Add sim nodes
    if sim_relations:
        train_network.add_sim_edges_bert()

    train_data, _ = reader_utils.prepare_batch_dgl(word_vocab, train_network, train_network, exist_relation)
    print("Adding sim edges..")
    if load_eval:
        test_data, test_labels = reader_utils.prepare_batch_dgl(word_vocab, test_network, train_network, exist_relation)
        valid_data, valid_labels = reader_utils.prepare_batch_dgl(word_vocab, dev_network, train_network,exist_relation)
        return train_data, valid_data, test_data, valid_labels, test_labels, train_network, word_vocab
    else:
        return train_data,  train_network
