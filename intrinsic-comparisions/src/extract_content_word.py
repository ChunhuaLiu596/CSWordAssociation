import os, sys
import nltk
from nltk import Tree
import json
from tqdm import tqdm
import spacy   #python -m spacy download en_core_web_sm
# import textacy #conda install -c conda-forge textacy
from collections import Counter
# import matplotlib.pyplot as plt
import numpy as np
# import benepar #pip install benepar
# from benepar.spacy_plugin import BeneparComponent
from multiprocessing import Pool
import argparse
from pos_utils import * 
import pandas as pd
# import seaborn as sns

nlp = customized_spacy_nlp(split_hyphens=False)

def get_pos_spacy(concepts, kg_name, concept_pos_path, lower_case=True, convert_num=True, debug=False, reload=False):
    '''
     # total = sum(doc_tagged.values())
    # doc_tagged_ratio = [(k,round(v/total, 4)) for k,v in doc_tagged.items()]
    # doc_tagged_ratio = Counter(dict(doc_tagged_ratio)).most_common()
    '''
    phrase_concepts = []
    with open(concept_pos_path, "w") as fw:
        doc_tagged=Counter()
        for i, concept in enumerate(tqdm(concepts, total=len(concepts))): 
            doc = nlp(concept) 
            if len(concept.split()) <2:
                for token in doc:
                    fw.write("{}\t{}\n".format(concept, token.pos_))
                    doc_tagged[token.pos_] +=1 
            else:
                if phrasal_verb_recognizer(doc):
                    fw.write("{}\t{}\n".format(concept,'VERB'))
                    doc_tagged['VERB'] +=1
                elif compound_noun_recognizer(doc):
                    fw.write("{}\t{}\n".format(concept,'NOUN'))
                    doc_tagged['NOUN'] +=1
                else:
                    phrase_concepts.append(concept)

    doc_tagged_norm = normalization_by_values(doc_tagged)

    plot_util(doc_tagged_norm, f"analysis/{kg_name}_unigram_pos.png")
    print("{} pos statistics: {}".format(kg_name, doc_tagged_norm))

    return doc_tagged.most_common(), doc_tagged_norm, phrase_concepts



def extract_content():
    concept= 'to make a bonfire so that we can enjoy a evening gathering at the campsite'
    doc = nlp(concept)  
    # is_compound_noun, compounds = compound_noun_recognizer(doc) 
    # print(is_compound_noun, compounds )
    words = content_word_recognizer(doc)    
    print(words)

extract_content()