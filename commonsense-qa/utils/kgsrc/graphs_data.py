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
# import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tqdm import tqdm

from reader import  ConceptNetTSVReader, SwowTSVReader, Triple2GraphReader
from utils_writer import *
import graph
import logging as logger
logger.getLogger().setLevel(logger.INFO)
PAD_TOKEN="_PAD"
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner', 'textcat'])

class GraphsData(object):
    def __init__(self, args):
        self.input_order = args.input_order
        self.write_non_overlap = args.write_non_overlap
        self.match_mode = args.match_mode
        self.add_isphrase_rel = args.add_isphrase_rel
        self.align_dir = args.align_dir
        self.isphrase_rel_name = 'isphrase'
        self.debug = args.debug

        self.net_sw = self.build_net(args.swow_source_file, ConceptNetTSVReader, "SWOW")
        self.net_cn = self.build_net(args.conceptnet_source_file, ConceptNetTSVReader, "ConceptNet")
        #self.net_sw = self.build_net(args.swow_source_file, SwowTSVReader)

        if self.match_mode !='hard':
            if args.pivot_kg=="ConceptNet":
                self.concept_to_lemma = self.entity_lemma_dict(list(self.net_sw.graph.node2id.keys()))

            if args.pivot_kg=="SWOW":
                self.concept_to_lemma = self.entity_lemma_dict(list(self.net_cn.graph.node2id.keys()))

        # self.r_ht = self.net_cn.graph.find_relation_distribution()

    def entity_lemma_dict(self, concepts):
        concept_to_lemma = dict()

        filename = f"{self.align_dir}/swow_concept_to_lemma.json"
        if os.path.exists(filename):
           fin = open(filename,'r')
           concept_to_lemma  = json.load(fin)
           assert set(concepts).issubset(set(concept_to_lemma.keys()))
        else:
            n = len(concepts)
            for i, concept in enumerate(tqdm(concepts, total=n,\
                                          desc='lemmatizing concepts')):
                 lemma = list(self.lemmatize(nlp, concept))[0]
                 if concept not in concept_to_lemma:
                     concept_to_lemma[concept] = lemma

            fout = open(filename, 'w') 
            json.dump(concept_to_lemma,fout)
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
