import os
import re 
import sys
import json,ujson
import time 
import spacy
# spacy.require_gpu()
from spacy.lang.en import English
import numpy as np 
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import Counter
from datetime import datetime
from scipy.sparse import csr_matrix
# import sparse_dot_topn.sparse_dot_topn as ct
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from multiprocessing import Pool
from spacy.matcher import Matcher
import functools
import operator
import networkx as nx
from itertools import islice

from stop_words import _STOP_WORDS, _BLACK_LIST 

nlp = spacy.load("en_core_web_sm")


# __all__ = ['create_matcher_patterns', 'ground']



def ngrams_char(string, n=3):
    '''
    # Check if this makes sense:
    # ngrams('!J INC')
    '''
    string = re.sub(r'[,-./]|\sBD',r'', string)
    ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]

def ngrams(string, n=3):
    '''
    # Check if this makes sense:
    # ngrams('!J INC')
    '''
    string = re.sub(r'[,-./]|\sBD',r'', string)
    ngrams = zip(*[string.split()[i:] for i in range(n)])

    return [' '.join(ngram) for ngram in ngrams]


########## text courpus functions #########
def read_omcs(path = 'data/omcs/omcs-sentences-more.txt'):
    '''
    read omcs sentences
    '''
    data_sent = set() 
    with open(path, 'r') as fin:
        for i, line in enumerate(tqdm(fin.readlines())):
            items = line.strip().split("\t")
            if len(items) < 7:
                continue
            if items[4] == 'en':
                data= re.split(r'!|\?|\.', items[1].lower())
                data_sent |= {*data}     # unordered, better to save for unique index
    return pd.DataFrame(list(data_sent), columns=['sentence'])


def cleaner(df, column):
    df = df[df[column].str.len() > 5].reset_index(drop=True)
    pattern = re.compile(r"[A-Za-z0-9\-]{5,50}")
    df[column] = df[column_name].str.findall(pattern).str.join(' ')
# %%
    return  df

def read_bookcorpus(archive='data/bookcorpus/', out_path=None, debug=False):
    '''
    reference: https://github.com/huggingface/datasets/blob/master/datasets/bookcorpus/bookcorpus.py
    '''
    # from datasets import list_datasets, list_metrics, load_dataset, load_metric
    t1 = datetime.now() 
    # dataset = load_dataset('bookcorpus', split='train')
   
    def _generate_examples(directory):
        files = [
            os.path.join(directory, "books_large_p1.txt"),
            os.path.join(directory, "books_large_p2.txt"),
        ]
        if debug:
            files = [os.path.join(directory, "books_large_d1.txt")]
        _id = 0
        for txt_file in files:
            with open(txt_file, mode="r", encoding="utf-8") as f:
                for line in tqdm(f, desc="Loading Bookcorpus"):
                    yield _id, {"text": line.strip().lower()}
                    _id += 1

    def _vocab_text_gen(archive):
        for _, ex in _generate_examples(archive):
            yield ex["text"]
            
    dataset = list(_vocab_text_gen(archive))

    #remove redundant lines 
    # nlp = spacy.load('en_core_web_sm', disable=['tagger', 'ner', 'parser', 'lemma', 'attr_rule'])
    # nlp.add_pipe('sentencizer')

    # for doc in tqdm(nlp.pipe(dataset, n_process=20, batch_size=256), total = len(dataset)):
    #     # data_sent |= {*data}
    #     data_sent |= {*list(doc.sents)}

    # nlp = English()
    nlp = spacy.blank('en')
    dataset = [nlp.make_doc(item) for item in tqdm(dataset, total = len(dataset), desc='making doc')]

    data_sent = set() 
    sentencizer = nlp.add_pipe("sentencizer")
    for doc in sentencizer.pipe(dataset, batch_size=512):
        data_sent |= {*list(doc.sents)}

    df = pd.DataFrame({'sentence': list(data_sent)})
    df = cleaner(df, column='sentence') #remove too short sentences
    # df['sentence'].astype(str)
    print(df.head(3))

    t = datetime.now() - t1
    print("BookCorpus: {} sentences. Cost time: {}".format( len(df.index), str(t)))

    if not out_path:
        out_path = "data/bookcorpus/clean_raw_files/books_large_p12.csv"
    df.to_csv(out_path)
    # df.to_json(out_path, orient='records', lines=True)
    print(f"save {out_path}")
    return df

def read_commoncrawl():
    path = 'data/commoncrawl/en.99.clean'
    data_sent = [] 
    t1 = datetime.now() 
    with open(path, 'r') as fin:
        lines = fin.readlines()
        for line in tqdm(lines, total=len(lines), desc="Load CommonCrawl"):
            data_sent.append(ujson.loads(line.strip())["sentence"])

    df_sent = pd.DataFrame({'sentence': data_sent})
    t = datetime.now() - t1
    print("CommonCrawl: {} sentences. Cost time: {}".format( df_sent['sentence'].count(), str(t)))
    return df_sent

def clean_commoncrawl(path, outpath, debug=False):
    '''
    clean a corpus by sentence token length 
    sort unique sentence
    '''
    t1 = datetime.now() 
    s_min = 7
    s_max = 50
    data_sent = set() 
    _id = 0
    with open(path, mode="r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading CommonCrawl"):
            line = line.strip().lower()

            line = re.split(r'!|\?|\.', line)
            line = [l for l in line if s_min < len(l.split()) < s_max  ]
            data_sent |= {*line}
            _id += 1
            if _id%1000000==0:
                print(f"Read {_id} lines")
            if debug and _id==5000000:
                break                
    
    df_sent = pd.DataFrame({'sentence': list(data_sent)})
    t = datetime.now() - t1
    print("CommonCrawl: {} sentences. Cost time: {}".format( len(df_sent.index), str(t)))

    with open(outpath, 'w') as fin:
        df_sent.to_json(fin, orient="records", lines=True)

    print(f"Save {outpath}")
    return df_sent



def generate_vocab(docs, output_path, batch_size=256, n_process=30):
    '''
    data: list
    tips: use nlp.pipe() for acceleration 
    goal: generate a vocab for a large text file
    question: 
    1. how to use Vocab
    2. how to accelerate with nlp.pipe() 


    '''
    nlp = spacy.load('en_core_web_sm')
    all_patterns={}
    
       # for doc in tqdm(nlp.pipe(docs, batch_size=batch_size, n_process=n_process, disable=["parser", "ner"]), total=len(docs), desc="lemmatizing docs"):
    for doc in tqdm(nlp.pipe(docs, batch_size=batch_size, n_process=n_process, disable=["parser", "ner"]), total=len(docs), desc="lemmatizing docs"):
        tokens  = [tok.text for tok in doc]
        lemmas = [[{"LEMMA": tok.lemma_}] for tok in doc]
        patterns = dict(zip(tokens, lemmas))
        all_patterns.update(patterns)
    
    print("Created " + str(len(all_patterns)) + " patterns.")
    with open(output_path, "w", encoding="utf8") as fout:
        json.dump(all_patterns, fout)
    print(f"Save {output_path}")

# def sentence_cleanup(docs, batch_size=256, n_process=30):
#     '''
#     1. if
#     '''
#     s_threshold = 7
#     nlp = spacy.load('en_core_web_sm')
#     all_sentences = set() 
#     all_patterns={}
#     # sid = 0
#     # s_to_id = {}
#     sentences = set()
#     for doc in tqdm(nlp.pipe(docs, batch_size=batch_size, n_process=n_process, disable=["parser", "ner"]), total=len(docs), desc="lemmatizing docs"):
#         if all([(tok.text in _STOP_WORDS or tok.lemma_ in _STOP_WORDS or tok.lemma_ in _BLACK_LIST)] for tok in doc): 
#             continue                        
#         tokens  = [tok.text for tok in doc]
#         s = " ".joint(tokens)
#         if len(tokens) < s_threshold: continue
#         sentences.add(s)
#     id_to_s = {i:s for i, s in enumerate(sentences)}


def corpus_lemmatization(docs, output_path, batch_size=256, n_process=30):
    '''
    data: list
    tips: use nlp.pipe() for acceleration 
    '''
    nlp = spacy.load('en_core_web_sm')
    all_patterns={}
    for doc in tqdm(nlp.pipe(docs, batch_size=batch_size, n_process=n_process, disable=["parser", "tagger","ner"]), total=len(docs), desc="lemmatizing docs"):
        # if all([(tok.text in _STOP_WORDS or tok.lemma_ in _STOP_WORDS or tok.lemma_ in _BLACK_LIST)] for tok in doc): 
        #     continue
        tokens  = [tok.text for tok in doc]
        lemmas = [[{"LEMMA": tok.lemma_}] for tok in doc]
        patterns = dict(zip(tokens, lemmas))
        all_patterns.update(patterns)
    
    print("Created " + str(len(all_patterns)) + " patterns.")
    with open(output_path, "w", encoding="utf8") as fout:
        json.dump(all_patterns, fout)
    print(f"Save {output_path}")

########## KG courpus functions #########
def read_triples(path, pair=True):
    triples=[]
    with open(path, 'r') as fin:
        for line in fin.readlines():
            items = line.strip().split("\t")
            if pair:
                triples.append(" ".join([items[1], items[2]]))
            else:
                triples.append(" ".join([items[1], items[0].lower(), items[2]]))
    return pd.DataFrame(triples, columns=['triple'])


def read_pair(path, pair=True):
    '''
    input format: (r,h,t,w), seperator='\t'
    '''
    pairs_seen = set()
    pairs=[]
    heads=[]
    tails=[]
    rels = []
    weights = []
    with open(path, 'r') as fin:
        for i, line in enumerate(fin.readlines()):
            items = line.strip().split("\t")
            items = [item.replace("_", " ") for item in items]

            if i ==0 and items[0] == 'rel':
                continue 
            if items[0] in ("bidirectionalassociated",):
                continue
            if items[1] in _STOP_WORDS or items[2] in _STOP_WORDS:
                continue
            if len(items[1]) == 1: # filter out single character as cue word in swow 
                continue
            if (items[1], items[2]) not in pairs_seen:
                pairs_seen.add((items[1], items[2]))
                if pair:
                    pairs.append(" ".join([items[1], items[2]]))
                else:
                    pairs.append(" ".join([items[1], items[0].lower(), items[2], items[3]]))
                heads.append(items[1])
                tails.append(items[2])
                rels.append(items[0])
                weights.append(float(items[3]))
    df = pd.DataFrame({"pair":pairs, "head": heads, "tail":tails, "rel":rels, 'weight': weights })
    print(f"Loaded {len(df.index)} pairs")
    return df 


def load_graph(path):
    '''
    triples to graph
    e.g., graph_sw =  load_graph(path +  'conceptnet.en.csv')
    search usage:
        if graph.has_edge(subj, obj):
            edge_data = graph[subj][obj]
            for idx, data  in edge_data.items():
                rel_cn = data['rel']
                weight_cn = data['weight']
    '''
    graph = nx.MultiDiGraph()
    with open(path, "r") as fo:
        for line in tqdm(fo.readlines()):
            rel, subj, obj, weight = line.strip().split("\t")
            if graph.has_edge(subj, obj):
                if rel not in graph[subj][obj]:
                    # print(rel, subj, obj)
                    graph.add_edge(subj, obj, key=rel, rel=rel, weight=weight)
            else:
                graph.add_edge(subj, obj, key=rel, rel=rel, weight=weight)
            # graph.add_edge(subj, obj, rel=rel, weight=weight)
            # graph.add_edge(obj, subj, rel="_"+rel, weight=weight)
    return graph
########## KG courpus functions #########

def get_matches_df(sparse_matrix, name_vector, column_vector,top=None):
    non_zeros = sparse_matrix.nonzero()

    sparserows = non_zeros[0]
    sparsecols = non_zeros[1]

    if top:
        nr_matches = top
    else:
        nr_matches = sparsecols.size

    left_side = np.empty([nr_matches], dtype=object)
    right_side = np.empty([nr_matches], dtype=object)
    similairity = np.zeros(nr_matches)

    for index in range(0, min(nr_matches, sparserows.size)):
        # print(index, nr_matches, len(sparserows), len(sparsecols))
        # print(sparserows[index], sparsecols[index])
        # print()
        left_side[index] = name_vector[sparserows[index]]
        right_side[index] = column_vector[sparsecols[index]]
        similairity[index] = sparse_matrix.data[index]

    return pd.DataFrame({'left_side': left_side,
                          'right_side': right_side,
                           'similairity': similairity})


# def conceptnet_to_omcs(debug, dataset):
#     # sentence_path = 'omcs-sentences-more.txt'
#     sentence_path = 'omcs-sentences-free.txt'
#     triple_path = 'train600k.txt'
#     read_pair=False

#     cosine_threshold = 0.5
#     if debug=='True':
#         print("Debugging")
#         topn=100000
#         df_sent = read_sentences(sentence_path).head(topn)
#         df_triple = read_triples(triple_path, read_pair).head(topn)
#     else:
#         df_sent = read_sentences(sentence_path)
#         df_triple = read_triples(triple_path, read_pair)
#         topn = len(df_triple)

#     print(df_sent.shape, df_sent.head(5))
#     print(df_triple.shape, df_triple.head(5))

#     # def matches(df_triple, df_sent):
#     triples = list(df_triple['triple'])
#     sents = list(df_sent['sentence'])

#     vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams)
#     tf_idf_matrix_all = vectorizer.fit_transform(triples + sents)
#     tf_idf_matrix_triples = tf_idf_matrix_all[:len(triples)]
#     tf_idf_matrix_sents  = tf_idf_matrix_all[-len(sents):]


#     t1 = time.time()
#     print("creating matches")
#     matches = awesome_cossim_top(tf_idf_matrix_triples, tf_idf_matrix_sents.transpose(), 10, cosine_threshold)
#     print("matches shape: {}".format(matches.shape))
#     t = time.time()-t1
#     print("timer :", t)

#     if debug=='True':
#         matches_df = get_matches_df(matches, triples, sents, top=100)
#     else:
#         matches_df = get_matches_df(matches, triples, sents)

#     # matches_df = get_matches_df(matches, triples, top=10)
#     matches_df = matches_df[matches_df['similairity'] < 0.99999] # Remove all exact matches
#     # print(matches_df)
#     # matches_df.sample(n=10)

#     # matches_df.sort_values(['similairity'], ascending=False).head(10)
#     matches_df.to_csv("triple_to_sentences.csv")
#     # print(matches_df)


def count_frequency(pair_count):
    '''
    count frequency by Counter() values
    '''
   
    count_number = Counter(pair_count.values())
    count_number_sorted = sorted(count_number.items(), key=lambda pair: int(pair[0]), reverse=False) # pair=[sentence number of per(h,t),  frequency of sentence number]

    return  count_number_sorted


#### Plot Functions ########
def plot_statics(data, output_path):
    '''
    data: a sorted list of list [[x,y], [x,y]], e.g., [[1, 3], [5, 1], [23, 1]]
    '''
    x, y = zip(*data)
    N=10
    if len(x) >=N:
        y_N = sum([j for i,j in zip(x,y) if i>=N])
        x, y = list(x[:N]), list(y[:N])
        y[-1] = y_N
    else:
        N=len(x)
    plot_data = tuple(zip(x,y)) 
    print("plot data: {}".format(plot_data))

    def plot_bar_chart(ax, N, x, y):
        x_tick = np.arange(1, N+1)
        ax.set_xticks(x_tick)
        ax.set_xticklabels([i for i in x]) # work only when set_xticks() is called

        ax.bar(x_tick, y)
        ax.set_yscale('log')
        # ax.set_xticklabels(labels=x)
        ax.set_xlabel("Sentence Number (xrank>10 are summed)")
        ax.set_ylabel("Count of (h,t) pairs")
        return ax
       

    def plot_pie_chart(ax, N, x, y):
        ax.axis('equal')
        patches,l_text,p_text = ax.pie(y, labels = x,autopct='%1.2f%%', textprops={'fontsize': 10})
        for t in p_text:
            t.set_size(5)

        for t in l_text: #label font size
            t.set_size(6)
        return ax

    fig, (ax1, ax2) = plt.subplots(2, 1)
    plt.title("".join(output_path.split(".")[:-1]))
    
    ax1 = plot_bar_chart(ax1, N, x, y)
    ax2 = plot_pie_chart(ax2, N, x, y) 

    fig.tight_layout()
    plt.show()
    plt.savefig(output_path)
    print("save {}".format(output_path))
    return plot_data
# %%
#### Lemmatizing  Functions ########

def load_cpnet_vocab(vocab_path):
    '''
    load vocab into a list
    '''
    with open(vocab_path, "r", encoding="utf8") as fin:
        cpnet_vocab = [l.strip() for l in fin]
    cpnet_vocab = [c.replace("_", " ") for c in cpnet_vocab]
    return cpnet_vocab

def create_pattern(nlp, doc, debug=False):
    '''
    Creating patterns for all concepts after filtering. 
    Filtering concepts 1) consisting of all stop words and 2) longer than four words. (why 2?)
    '''
    if len(doc) >= 5 or all([(token.text in _STOP_WORDS or token.lemma_ in _STOP_WORDS or token.lemma_ in _BLACK_LIST) for token in doc]):
        if debug:
            return False, doc.text
        return None  # ignore this concept as pattern

    pattern = []
    for token in doc:  # a doc is a concept containing single or multiple tokens/words
        pattern.append({"LEMMA": token.lemma_})
    if debug:
        return True, doc.text
    return pattern


def create_matcher_patterns(vocab_path, output_path, reuse=False, debug=False):
    if reuse:
        all_pattern = {}
        with open(pattern_path, "r", encoding="utf8") as fin:
            all_patterns = ujson.load(fin)
        print(f"Loaded {output_path}")
        
    else:
        cpnet_vocab = load_cpnet_vocab(vocab_path)
        nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner', 'textcat'])
        docs = nlp.pipe(cpnet_vocab)
        all_patterns = {}
    
        if debug:
            f = open("filtered_concept.txt", "w")
    
        for doc in tqdm(docs, total=len(cpnet_vocab), desc=f"lemmatizing {vocab_path}"):
    
            pattern = create_pattern(nlp, doc, debug)
            if debug:
                if not pattern[0]:
                    f.write(pattern[1] + '\n')
    
            if pattern is None:
                continue
            all_patterns["_".join(doc.text.split(" "))] = pattern
    
        print("Created " + str(len(all_patterns)) + " patterns.")
        with open(output_path, "w", encoding="utf8") as fout:
            json.dump(all_patterns, fout)
        if debug:
            f.close()
        print(f"Save {output_path}")
    return all_patterns

def lemmatize(nlp, concept):

    doc = nlp(concept.replace("_", " "))
    lcs = set()
    # for i in range(len(doc)):
    #     lemmas = []
    #     for j, token in enumerate(doc):
    #         if j == i:
    #             lemmas.append(token.lemma_)
    #         else:
    #             lemmas.append(token.text)
    #     lc = "_".join(lemmas)
    #     lcs.add(lc)
    lcs.add("_".join([token.lemma_ for token in doc]))  # all lemma
    return lcs


def load_matcher(nlp, pattern_path):
    assert os.path.exists(pattern_path), f"{pattern_path} doesn't exist"
    with open(pattern_path, "r", encoding="utf8") as fin:
        all_patterns = json.load(fin)

    matcher = Matcher(nlp.vocab)
    for concept, pattern in all_patterns.items():
        matcher.add(concept, [pattern])
    return matcher, all_patterns

def merge_matcher(nlp, pattern_paths:list):

    matcher = Matcher(nlp.vocab)
    for pattern_path in pattern_paths: 
        _, patterns = load_matcher(nlp, pattern_path)
        for concept, pattern in patterns.items():
            matcher.add(concept, [pattern])
    return matcher

def flat_list(a):
    return functools.reduce(operator.iconcat, a, [])

def flat_list_dicts(a):
    res = {}
    for d in tqdm(a, total=len(a), desc='flat list of dicts'):
        res.update(d)
    return res

def get_filename(dir):
    import os
    filenames= []
    for root, dirs, files in os.walk(dir):
        for filename in files:
            filenames.append(filename)
    return sorted(filenames)

def join_dir_filename(dir, filenames):
    filepaths = []
    for filename in filenames:
        filepaths.append(os.path.join(dir,filename))
    return filepaths

# def find_filename(dir, prefix, psuffix):
#     import os, fnmatch

#     listOfFiles = os.listdir('.')
#     pattern = suffix + suffix
#     for entry in listOfFiles:
#         if fnmatch.fnmatch(entry, pattern):
#             print (entry)


def get_path_args(dataset, corpus_name, version='v1'):
    output_path = f"data/{dataset}/hts_{corpus_name}_{version}_hit.json"
    output_path_missed = f"data/{dataset}/hts_{corpus_name}_{version}_missed.json"
    fig_path = f"data/{dataset}/{dataset}_to_{corpus_name}_{version}.png"
    return output_path, fig_path ,output_path_missed 


def dump_dic(dic, path):
    '''
    words: {word:lemma}
    '''
    with open(path, 'w') as f:
        ujson.dump(dic,f)

    lk, lv = len(dic.keys()), sum([len(v) for v in dic.values()])
    print(f"save {lk} keys {lv} values to {path}")

def check_file(path):
    # assert os.path.exists(path), f"{path} doesn't exist" 
    # assert os.path.getsize(path)!=0, f"{path} is an empty file" 
    return True if os.path.exists(path) and os.path.getsize(path)!=0 else False 

def load_dic(path):
    '''
    load pre-build inverted 
    format: {word:{doc_id:[locations]}}
    '''
    if check_file(path):
        with open(path, 'r') as f:
            data = ujson.load(f)
        print(f"Loaded inverted from {path}")
        return data 
    else:
        return {} 





def next_n_lines(file_opened, N):
    # return [x.strip() for x in islice(file_opened, N)]
    while True:
        lines = [x.strip() for x in islice(file_opened, N)]

        if not lines:
            break
        yield lines

def find_sub_list(sl,l):
    if isinstance(sl, str):
        sl=sl.split()
    sll=len(sl)

    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            if sll==1:
                return [[ind]]
            else:
                return [[ind,ind+sll-1]]
