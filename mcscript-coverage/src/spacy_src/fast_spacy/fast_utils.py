import os
import re
import pandas as pd
import spacy
from tqdm import tqdm
import functools
import operator
import math
from spacy.tokens import DocBin
from joblib import Parallel, delayed

import logging
logger = logging.getLogger("spacy")
logger.setLevel(logging.ERROR) # filtering out unnecessary warning from spacy

nlp = spacy.load('en_core_web_sm', disable=[ 'parser', 'ner'])
nlp.add_pipe("sentencizer")

limit = 0

stopwordfile = "src/spacy_src/stopwords.txt"
def get_stopwords():
    "Return a set of stopwords read in from a file."
    with open(stopwordfile) as f:
        stopwords = []
        for line in f:
            stopwords.append(line.strip("\n"))
    # Convert to set for performance
    stopwords_set = set(stopwords)
    return stopwords_set

stopwords = get_stopwords()

####### 
def lemmatize(text):
    """Perform lemmatization and stopword removal in the clean text
       Returns a list of lemmas
    """
    doc = nlp(text)
    lemma_list = [str(tok.lemma_).lower() for tok in doc
                  if tok.is_alpha and tok.text.lower() not in stopwords]
    return lemma_list

def lemmatize_pipe_rmstopwords(doc):
    lemma_list = [str(tok.lemma_).lower() for tok in doc
                  if tok.is_alpha and tok.text.lower() not in stopwords] 
    return lemma_list

def lemmatize_pipe(doc):
    '''
    note: keep stopwords in a sentence for concept matching
    '''
    lemma_list = [str(tok.lemma_).lower() for tok in doc if tok.is_alpha ] 
    return lemma_list



def cleaner(df, column_name):
    "Extract relevant text from DataFrame using a regex"
    # Regex pattern for only alphanumeric, hyphenated text with 3 or more chars
    pattern = re.compile(r"[A-Za-z0-9\-]{5,50}")
    df['clean'] = df[column_name].str.findall(pattern).str.join(' ')
    if limit > 0:
        return df.iloc[:limit, :].copy()
    else:
        return df

def chunker(iterable, total_length, chunksize):
    return (iterable[pos: pos + chunksize] for pos in range(0, total_length, chunksize))

def flatten(list_of_lists):
    "Flatten a list of lists to a combined list"
    return [item for sublist in list_of_lists for item in sublist]

def flat_list(a):
    return functools.reduce(operator.iconcat, a, [])


def lemmatize_chunk(texts, batch_size=200):
    preproc_pipe = []
    for doc in tqdm(nlp.pipe(texts, batch_size=batch_size), total = len(texts)):
        preproc_pipe.append(lemmatize_pipe(doc))
    return preproc_pipe

# def preprocess_parallel(texts, total_length, n_jobs=20, chunksize=100):
def lemmatize_parallel(texts, total_length, n_jobs=20, chunksize=100):
    ''' 
    1. chunk texts by chunksize 
    2. assign each chunk to multiple jobs
    3. each chunk is lemmatizes with a batch 
    '''
    executor = Parallel(n_jobs=n_jobs, backend='multiprocessing', prefer="processes")
    do = delayed(lemmatize_chunk)
    tasks = (do(chunk) for chunk in chunker(texts, total_length, chunksize=chunksize))
    result = executor(tasks)
    return flatten(result)


def nlp_chunk(texts, batch_size=200):
    preproc_pipe = []
    for doc in tqdm(nlp.pipe(texts, batch_size=batch_size), total = len(texts)):
        # preproc_pipe.append(doc)
        preproc_pipe.append([tok.pos_ for pos in doc])
    return preproc_pipe

def docbin_chunk(texts, save_path=None, batch_size=200):
    doc_bin = DocBin(attrs=["ORTH","POS","LEMMA"], store_user_data=True)
    # preproc_pipe = []
    for doc in tqdm(nlp.pipe(texts, batch_size=batch_size), total = len(texts)):
        # preproc_pipe.append(doc)
        doc_bin.add(doc) 

    if save_path:
        doc_bin.to_disk(save_path) 
        print(f"save serialized data {save_path}")
    return doc_bin
    # return preproc_pipe


def nlp_parallel(texts, total_length, nlp_path, n_jobs=30, batch_size=128, chunksize=100):
    ''' 
    1. chunk texts by chunksize 
    2. assign each chunk to multiple jobs
    3. each chunk is lemmatizes with a batch 
    '''
       
    # tasks = chunker(texts, total_length, chunksize=chunksize)
    # doc_bin = DocBin(attrs=["ORTH","POS","LEMMA"], store_user_data=True)
    # for doc in tqdm(nlp.pipe(texts, n_process=n_jobs, batch_size=batch_size), total = len(texts)):
    #     doc_bin.add(doc) 
    chunknum=10
    chunksize = math.ceil(total_length/chunknum)
    if nlp_path is not None:
        save_paths = []
        for i in range(0, chunknum):
            save_paths.append(nlp_path + f"{i}")
        print(save_paths)
        chunks = chunker(texts, total_length, chunksize=chunksize )
        executor = Parallel(n_jobs=n_jobs, backend='multiprocessing', prefer="processes")
        do = delayed(docbin_chunk)
        tasks = (do(chunk, save_path) for chunk,save_path in zip(chunks,save_paths))
        result = executor(tasks)
        # result = flatten(result)
        # result = flat_list(result)
    else:
        chunks = chunker(texts, total_length, chunksize=chunksize )
        executor = Parallel(n_jobs=n_jobs, backend='multiprocessing', prefer="processes")
        do = delayed(nlp_chunk)
        tasks = (do(chunk) for chunk in zip(chunks))
        result = executor(tasks)
        # result = flatten(result)
        result = flat_list(result)
# 
    # 
    # for nlp in result:
        # doc_bin.add(nlp) 
    # lambda x: doc_bin.add(x), result
    # map(lambda x: doc_bin.add(x), result)
    # doc_bin.to_disk(nlp_path)
    # print(f"save serialized data {nlp_path}")
    # return doc_bin 
    return result


def pos_lemmatize_pipe(doc):
    '''
    note: keep stopwords in a sentence for concept matching
    '''
    list = ["{}/{}/{}".format(tok.orth_, tok.pos_, tok.lemma_) for tok in doc ] 
    return list

def pos_lemmatize_chunk(texts, batch_size=200):
    preproc_pipe = []
    for doc in tqdm(nlp.pipe(texts, batch_size=batch_size), total = len(texts)):
        preproc_pipe.append(pos_lemmatize_pipe(doc))
    return preproc_pipe

def pos_lemmatize_parallel(texts, total_length, n_jobs=10, chunksize=10000):
    ''' 
    1. chunk texts by chunksize 
    2. assign each chunk to multiple jobs
    3. each chunk is lemmatizes with a batch 
    '''
    executor = Parallel(n_jobs=n_jobs, backend='multiprocessing', prefer="processes")
    do = delayed(pos_lemmatize_chunk)
    tasks = (do(chunk) for chunk in chunker(texts, total_length, chunksize=chunksize))
    result = executor(tasks)
    return flatten(result)