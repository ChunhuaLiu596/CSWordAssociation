# filtering warnings
# from matcher_span_single_test import generate_pair_matcher, generate_pattern
import warnings
# with warnings.catch_warnings():
#     warnings.filterwarnings("ignore",category=FutureWarning)
#     import tensorflow as tf # needed to for filtering warning

import logging
# logger = logging.getLogger("tensorflow")
# logger.setLevel(logging.ERROR) # filtering out unnecessary warning from spacy

import os, sys
import pandas as pd
import spacy
import re
from tqdm import tqdm
tqdm.pandas()

from datetime import datetime
from pathlib import Path
import numpy as np
import argparse

from utils import read_pair
from utils_matcher import generate_lemma_pattern_pair, generate_pattern_phrase, generate_orth_pattern_concept
from fast_spacy.fast_utils import lemmatize_parallel, get_stopwords
stopwords = get_stopwords()

def save_json(df_pair, output_path):
    df_pair.to_json(output_path, orient="records", lines=True)
    print(f"Save {output_path} : {len(df_pair.index)} lines")


def remove_same_lemma(df):
    '''
    remove head and tail which have the same lemma
    '''
    # df['equal'] = np.where(df["head_lemma"] == df["tail_lemma"], True, False)
    df['lemma_equal'] = [True if x==y else False for x, y in zip(df["head_lemma"], df["tail_lemma"])]
    df = df[df['lemma_equal'] == False]
    print("removing ht share a lemma:", len(df.index))
    return df

def detect_stopwords(words):
    return True if all(word in stopwords for word in words) else False

def remove_stopwords(df):
    '''
    remove head or tail belongs to stopwords
    '''
    # df['head_lemma_stop']= df['head_lemma'].apply(detect_stopwords)
    # df['tail_lemma_stop']= df['tail_lemma'].apply(detect_stopwords)
    df.loc[:,'head_lemma_stop'] = df['head_lemma'].apply(detect_stopwords)
    df.loc[:,'tail_lemma_stop'] = df['tail_lemma'].apply(detect_stopwords)
    df = df.query('head_lemma_stop != True & tail_lemma_stop != True')

    print("removing stopwords:", len(df.index))
    return df

# def remove_repeat_lemma(df):
    '''
    remove (h1.lemma =h2.lemma, t1.lemma=t2.lemma)
    '''

def pair_cleapup(df, lemmatized=False):
    if lemmatized:
        df = remove_same_lemma(df)
    # df = remove_stopwords(df)
    return df

# def single_kg(input_path, args):
#     df =  read_pair(path=input_path)
#     if debug:
#         df=df.head(50000)
#     n_jobs = 20
#     chunksize=50000
#     t1 = datetime.now()

#     if args.lemma:
#         df['head_lemma'] = lemmatize_parallel(df['head'], total_length=len(df.index),n_jobs=n_jobs, chunksize=chunksize)
#         df['tail_lemma'] = lemmatize_parallel(df['tail'], total_length=len(df.index),n_jobs=n_jobs, chunksize=chunksize)

#     df = pair_cleapup(df)

#     df['query'] = df['head_lemma'] + df['tail_lemma']
#     # columns_to_keep = ['pair', 'head', 'tail', 'rel', 'weight','head_lemma', 'tail_lemma', 'query' ]
#     # df = df.loc[:, columns_to_keep]

#     df['matcher_lemma']=df[['head_lemma', 'tail_lemma']].apply(lambda x: generate_lemma_pattern_pair(*x), axis=1)
#     df['matcher_orth']=df[['head', 'tail']].apply(lambda x: generate_lemma_pattern_pair(*x, lemmatized=False, to_lemmatize=False), axis=1)
#     columns_to_keep = ['pair', 'head', 'tail', 'rel', 'weight','head_lemma', 'tail_lemma', 'query','matcher']
#     # df = df.loc[:, columns_to_keep]
#     return df

def single_kg_orth(input_path, debug=False):
    df =  read_pair(path=input_path)
    n_jobs = 20
    chunksize=50000
    if debug:
        df=df.head(50)
        n_jobs = 1
        chunksize=50

    t1 = datetime.now()

    # df = pair_cleapup(df)

    # df['head_matcher']=df['head'].progress_apply(lambda x: generate_orth_pattern_concept(x))
    # df['tail_matcher']=df['tail'].progress_apply(lambda x: generate_orth_pattern_concept(x))
    # df['query'] = df['pair']

    # columns_to_keep = ['pair', 'head', 'tail', 'rel', 'weight', 'head_matcher', 'tail_matcher', 'query']
    columns_to_keep = ['rel', 'head', 'tail', 'weight']
    df = df.loc[:, columns_to_keep]

    return df

def single_kg_lemma(input_path, debug=False):
    df =  read_pair(path=input_path)
    n_jobs = 20
    chunksize=50000
    if debug:
        df=df.head(50)
        n_jobs = 1
        chunksize=50

    t1 = datetime.now()

    # df = pair_cleapup(df, lemmatized=True)

    df['head_lemma'] = lemmatize_parallel(df['head'], total_length=len(df.index),n_jobs=n_jobs, chunksize=chunksize)
    df['tail_lemma'] = lemmatize_parallel(df['tail'], total_length=len(df.index),n_jobs=n_jobs, chunksize=chunksize)
    df['head_lemma'] = df['head_lemma'].apply(lambda x: " ".join(x))
    df['tail_lemma'] = df['tail_lemma'].apply(lambda x: " ".join(x))

    # df['matcher_lemma']=df[['head_lemma', 'tail_lemma']].apply(lambda x: generate_lemma_pattern_pair(*x), axis=1)
    # df['matcher_orth']=df[['head', 'tail']].apply(lambda x: generate_lemma_pattern_pair(*x, lemmatized=False, to_lemmatize=False), axis=1)
    df = df.query('head_lemma.str.len()>0 & tail_lemma.str.len()>0')

    df['query'] = df['head_lemma'] + df['tail_lemma']

    columns_to_keep = ['rel', 'head_lemma', 'tail_lemma','weight']

    # columns_to_keep = ['head', 'tail', 'rel', 'weight','head_lemma', 'tail_lemma']
    # columns_to_keep = ['pair', 'head', 'tail', 'rel', 'weight','head_lemma', 'tail_lemma', 'query','matcher']
    df = df.loc[:, columns_to_keep]
    return df


attr_func={"orth": single_kg_orth, "lemma":single_kg_lemma}

def preprocess_kg(debug, attr='orth', input_paths=None, output_paths=None):
    if input_paths is None and output_paths is None:
        input_paths = ['data/swow/conceptnet.en.csv', 'data/conceptnet/conceptnet.en.csv']
        output_paths = [f'data/swow/conceptnet.en.{uniq}_spacy.json', f'data/conceptnet/conceptnet.en.{attr}_spacy.json']
    # input_paths = ['data/swow/conceptnet.en.csv']
    # output_paths = ['data/swow/conceptnet.en.uniq.json']
    t1 = datetime.now()

    for input_path, output_path in zip(input_paths, output_paths):
        df = attr_func.get(attr)(input_path)
        print(df.tail(5))
        df.to_csv(output_path, index=False, sep='\t', header=False)
        print(f"save {output_path}  {len(df.index)} lines")
        print(f"Cost time: {datetime.now()-t1}")

def merge_kg(debug, attr='orth'):
    t1 = datetime.now()
    input_paths = ['data/swow/conceptnet.en.csv', 'data/conceptnet/conceptnet.en.csv']
    output_path = f'data/swcn/conceptnet.en.{attr}_spacy.csv'
    # input_paths = ['data/swow/conceptnet.en.csv']
    # output_paths = ['data/swow/conceptnet.en.uniq.json']
    dfs = []
    for input_path in input_paths:
        # df = single_kg(input_path)
        df = attr_func.get(attr)(input_path, debug)
        dfs.append(df)

    df = pd.concat(dfs).reset_index(drop=True)
    print(df.tail(5))
    df.to_csv(output_path)
    # save_json(df, output_path)
    print(f"save {output_path}  {len(df.index)} lines")
    print(f"Cost time: {datetime.now()-t1}")

def add_arguments():
    parser=argparse.add_arguments("")

if __name__=='__main__':
    # debug=True
    debug=False
    # merge_kg(debug, attr='orth')
    #merge_kg(debug, attr='lemma')
    kg_name = sys.argv[1]
    input_paths = [sys.argv[2]]
    print(kg_name)
    print(input_paths )
    output_paths = [sys.argv[3]]
    preprocess_kg(debug, attr='lemma', input_paths=input_paths, output_paths=output_paths)

    if 'clsb' in kg_name:
        preprocess_kg(debug, attr='lemma', input_paths=['data/prop_norm/clsb2014.en.csv'], output_paths=['data/prop_norm/clsb.lemma_spacy.csv'])
