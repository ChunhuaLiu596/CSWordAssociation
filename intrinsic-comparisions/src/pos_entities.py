import os, sys
import nltk
from nltk import Tree
import json
from tqdm import tqdm
import spacy   #python -m spacy download en_core_web_sm
import textacy #conda install -c conda-forge textacy
from collections import Counter
import matplotlib.pyplot as plt
python -u utils/kgsrc/pos_swow_cue.py
import numpy as np
import benepar #pip install benepar
from benepar.spacy_plugin import BeneparComponent
from multiprocessing import Pool
import argparse
from pos_utils import * 
import pandas as pd
import seaborn as sns
from matplotlib.ticker import ScalarFormatter,FormatStrFormatter

'''
  what information do I want to convey?
    1. SWOW has more advantge of single token connection.
    2. CN has more advantage of single-phrase connection and phrase-phrase connection.

    merge (A,B) (B,A) to (A,B)
'''


def bool_flag(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def check_path(path):
    d = os.path.dirname(path)
    if not os.path.exists(d):
        os.makedirs(d)


'''
input: concept.txt 
output: 
1) the percentage of NP/VP 
2) the distribution of phrase type for the combination of (head tail) (Unigram)
questions:
1) How to deal with multi-word expression?
'''

nlp = customized_spacy_nlp(split_hyphens=False)
nlp.add_pipe(BeneparComponent('benepar_en'))

def load_concepts(input_file, debug=False):
    # unigram=[]
    ngram=[]
    concepts = []
    with open(input_file, "r") as fr:
        lines = fr.readlines()

        for i, line in enumerate(lines):
            line=line.strip().replace("_", " ")
            concepts.append(line)

            # if len(line.split()) ==1:  ## uni-gram and bi-gram
                # unigram.append(line)
# 
            if len(line.split()) >=2:  ## uni-gram and bi-gram
                ngram.append(line)

            # elif len(line.split()) >=3:  ## N-gram 
                # trigram.append(line)
# 
            if debug is True and len(ngram)>100:
                print("Debugging {} concepts".format(i))
                break

    return concepts


def chunk_pairs_statistics(triple_path, kg_name, pos_path, chunk_path, output_path):
    '''
    triple_path: rel, head, tail, w
    chunk_path: concept concept_tag
    '''
    chunk_tags = {}
    for line in open(pos_path, 'r', encoding='utf-8'):
        chunk, chunk_tag = line.strip().split("\t")
        chunk_tags[chunk]=chunk_tag

    for line in open(chunk_path, 'r', encoding='utf-8'):
        chunk, chunk_tag, chunk_sequence = line.strip().split("\t")
        chunk_tags[chunk]=chunk_tag
        if len(eval(chunk_sequence)) >2:
            print("**", chunk, chunk_tag, chunk_sequence)

    chunk_pairs_count=Counter()
    check_path(output_path)
    with open(triple_path, "r") as fr, open(output_path, "w") as fw :
        for triple in fr.readlines():
            rel, arg1, arg2, weight = triple.strip().split("\t")

            arg1, arg2 = arg1.replace("_", " "), arg2.replace("_", " ")

            tags = (chunk_tags[arg1], chunk_tags[arg2])
            chunk_pairs_count[tags] +=1

            fw.write("{}\t{}\t{}\t{}\n".format(rel, arg1, arg2, tags))
        
        print("{} chunk pairs statistics (top20):{} ".format(kg_name, chunk_pairs_count.most_common(20)))
        chunk_pairs_norm = normalization_by_values(chunk_pairs_count)
        print("save {}".format(output_path))
        print("save top(20) {}".format(chunk_pairs_norm[:20]))
        plot_util(chunk_pairs_norm, f"analysis/{kg_name}_node_pairs_pos_phrasetype.png")
    return chunk_pairs_count, chunk_pairs_norm


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


def plot_util(doc_tagged, fig_name):
    # labels, values = zip(*doc_tagged.items())
    # labels, values = zip((x[0],x[1]) for x in doc_tagged)
    
    labels = [x[0] for x in doc_tagged]
    values = [x[1] for x in doc_tagged]
    indexes = np.arange(len(labels))
    width = 0.8

    plt.bar(indexes, values, width)
    plt.yscale('log')
    plt.xticks(indexes + width*0.1, labels, rotation=45)

    # for ax 
    plt.show()
    plt.savefig(fig_name)
    plt.clf()
    print("save {}".format(fig_name))
     

def normalization_by_values(inp_dic, sorted=False):
    '''
    input a dict, norm values, return with values sorted tuple
    '''

    total = sum(inp_dic.values())
    inp_dic_norm = [(k,round(v/total, 4)) for k,v in inp_dic.items()]
    inp_dic_norm = Counter(dict(inp_dic_norm)).most_common()
    return inp_dic_norm 

def constituents_parsing_ngram(input):
    '''
    doc = nlp('The time for action is now. It is never too late to do something.')
    for i, x in enumerate(list(sent._.children)):
        print(input, sent._.labels[-1], i, x) # notes: ngram: sent._.labels[-1]
    '''
    doc= nlp(input)
    sent = list(doc.sents)[0]
    # print(sent, sent._.labels[-1])
    # return sent, sent._.labels[-1] 
    return sent, sent._.labels[-1], sent._.labels


def get_chunk_spacy(concepts, kg_name, concept_chunk_path, debug=False):
    chunk_counter = Counter()

    # with Pool(num_processes) as p:
        # for chunk, chunk_tag in tqdm(p.imap(constituents_parsing_ngram, concepts), total=len(concepts)): 
    check_path(concept_chunk_path)
    with open (concept_chunk_path, "w") as fw:
        for concept in tqdm(concepts, total=len(concepts)): 
            chunk, chunk_tag, chunk_sequence = constituents_parsing_ngram(concept)
            chunk_counter[chunk_tag] +=1
            fw.write("{}\t{}\t{}\n".format(concept, chunk_tag, chunk_sequence))

    chunk_counter_norm = normalization_by_values(chunk_counter)
    plot_util(chunk_counter_norm, f"analysis/{kg_name}_ngram_phrasetag.png")
    # print(chunk_counter)
    # print(chunk_counter_norm)
    print("save {}".format(concept_chunk_path))
    return chunk_counter.most_common(), chunk_counter_norm


def pos2csv(args):
    csv_path = "./analysis/node_pos_cn_sw.csv"
    if args.reload:
        df=pd.read_csv(csv_path)
        df_freq= None
    else:
        concepts_cn = load_concepts(args.concept_path_cn, debug=args.debug)
        pos_cn, pos_cn_norm, phrase_cn = get_pos_spacy(concepts_cn, "conceptnet", args.concept_pos_path_cn, debug=args.debug)
    
        concepts_sw = load_concepts(args.concept_path_sw, debug=args.debug)
        pos_sw, pos_sw_norm, phrase_sw = get_pos_spacy(concepts_sw, "swow", args.concept_pos_path_sw, debug=args.debug)

        df_freq = pd.DataFrame(pos_cn, columns=['tags', 'CN'])
        df_freq['SW'] = [dict(pos_sw).get(k,0) for k in df_freq['tags']]

        df = pd.DataFrame(pos_cn_norm, columns=['tags', 'CN'])
        df['SW'] = [dict(pos_sw_norm).get(k,0) for k in df['tags']]
        df.to_csv(csv_path)


    return df_freq, df, phrase_cn, phrase_sw

def phrase2csv(args, phrase_cn=None, phrase_sw=None):
    csv_path = "./analysis/node_phrase_cn_sw.csv"
    if args.reload:
        df=pd.read_csv(csv_path)
    else:
        if phrase_cn is  None and phrase_sw is None:
            phrase_cn = load_concepts(args.concept_path_cn, debug=args.debug)
            phrase_sw = load_concepts(args.concept_path_sw, debug=args.debug)

        tag_cn, tag_cn_norm = get_chunk_spacy(phrase_cn, "conceptnet", args.concept_chunk_path_cn)
        tag_sw, tag_sw_norm = get_chunk_spacy(phrase_sw, "swow", args.concept_chunk_path_sw)
        
        df_freq = pd.DataFrame(tag_cn, columns=['tags', 'CN'])
        df_freq['SW'] = [dict(tag_sw).get(k,0) for k in df_freq['tags']]

        df = pd.DataFrame(tag_cn_norm, columns=['tags', 'CN'])
        df['SW'] = [dict(tag_sw_norm).get(k,0) for k in df['tags']]
        df.to_csv(csv_path)


    return df_freq, df


def pair_tags_to_csv(args):
    # if args.reload:
        # df = pd.read_csv('./analysis/tag_pairs_cn_sw.csv')
    # else:
    chunk_pairs_cn_count,chunk_pairs_cn = chunk_pairs_statistics(args.triple_path_cn, "conceptnet", args.concept_pos_path_cn, args.concept_chunk_path_cn, args.triple_chunk_path_cn)

    df = pd.DataFrame(all_tags, columns=['tags'])
    df['CN'] = [dict(chunk_pairs_cn).get(k,0) for k in df['tags']]
    # df = pd.DataFrame(chunk_pairs_cn, columns=['tags', 'CN'])

    chunk_pairs_sw_count, chunk_pairs_sw =  chunk_pairs_statistics(args.triple_path_sw, "swow", args.concept_pos_path_sw,args.concept_chunk_path_sw, args.triple_chunk_path_sw)
    df['SW'] = [dict(chunk_pairs_sw).get(k,0) for k in df['tags']]

    df['tag_group'] = [ group_tags[k] for k in df['tags']] 
    df['CN_count'] = [dict(chunk_pairs_cn_count).get(k,0) for k in df['tags']]
    df['SW_count'] = [dict(chunk_pairs_sw_count).get(k,0) for k in df['tags']]
    df['SWdCN'] = df['SW']/df['CN']

    print("top 20")
    print(df[:20])
    df.to_csv (r'./analysis/tag_pairs_cn_sw.csv', index = False, header=True)
    plot_pair_tags_bar_groups(df, fig_name="./analysis/tag_pairs_cn_sw.png", reload=args.reload)

    print(df.groupby(['tag_group'])['CN'].sum())
    print(df.groupby(['tag_group'])['SW'].sum())

    selected_tags=['NOUN', 'PROPN', 'NP', 'VP', 'VERB', 'ADJ', 'ADV']
    selected_tag_pairs = [ [(x,y)] for x in selected_tags for y in selected_tags ]

    y_index = [x  for x in reversed(range(len(selected_tags))) for y in reversed(range(len(selected_tags)))]
    x_index = [y  for x in reversed(range(len(selected_tags))) for y in range(len(selected_tags))]

    # df_selected = pd.DataFrame(selected_tag_pairs, columns=['tags'])
    # df_selected['y'] =  y_index
    # df_selected['x'] =  x_index
    selected_tag_pairs = pd.DataFrame(selected_tag_pairs, columns=['tags'])
    df.set_index('tags')
    df1 = df.loc[df['tags'].isin(selected_tag_pairs['tags'])]
    
    df1['y'] =  y_index
    df1['x'] =  x_index
    df1.reset_index(drop=True, inplace=False)
    df1.set_index(['tags'])
    df1.reindex(selected_tag_pairs['tags'])
    
    print(df1)
    df1.to_csv (r'./analysis/tag_pairs_cn_sw_selected.csv', index = False, header=True)
    # plot_pair_tags_bar_groups(df_selected, fig_name="./analysis/tag_pairs_cn_sw.png", reload=args.reload)

def node_tags_to_csv(args):
    df1_freq, df1, phrase_cn, phrase_sw = pos2csv(args)
    df2_freq, df2 = phrase2csv(args, phrase_cn, phrase_sw)

    df_freq = pd.concat([df1_freq, df2_freq])
    df_freq.sort_values(by=['CN', 'SW'], ascending=False, inplace=True)

    csv_path = "./analysis/tag_node_pos_phrase_cn_sw.csv"
    df_freq.to_csv(csv_path)
    print("save {}".format(csv_path))
    print(df_freq)

    plot_pos_phrase_tags(df1, df2)
    plot_phrase_tags(df2)


def plot_pos_phrase_tags(df1, df2, topN=10):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10,5))

    df1 = df1[:topN]
    N = np.arange(len(df1['tags']))
    labels = df1['tags']
    ax1.set_xticks(N)
    ax1.set_xticklabels(labels, rotation=45)
    ax1.set_xlabel('Category of POS tags')

    ax1.set_ylim(ymin=0.0001, ymax=1)
    ax1.set_yscale('log')
    ax1.set_ylabel('Ratio')
    # ax1.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))

    ax1.bar(N + 0.00, df1['CN'], color = 'C0', width = 0.25, label='ConceptNet')
    ax1.bar(N + 0.25, df1['SW'], color = 'C1', width = 0.25, label='SWOW')
    ax1.legend()
    ax1.set_title("POS Tag Distribution")

    df2 = df2[:topN]
    N = np.arange(len(df2['tags']))
    labels = df2['tags']
    # ax2.invert_xaxis()
    ax2.set_xticks(N)
    ax2.set_xticklabels(labels, rotation=45)
    ax2.set_xlabel('Category of phrase tags')
    ax2.set_ylim(0.0001, 1.0)
    ax2.set_yscale('log')
    ax2.set_ylabel('Ratio')
    # ax2.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))

    ax2.bar(N + 0.00, df2['CN'], color = 'C0', width = 0.25, label='ConceptNet')
    ax2.bar(N + 0.25, df2['SW'], color = 'C1', width = 0.25, label='SWOW')
    ax2.legend()
    ax2.set_title("Phrase Tag Distribution")

    # for axis in [ax.xaxis, ax.yaxis]:
        # axis.set_major_formatter(FormatStrFormatter('%.2f'))

    # plt.legend()
    plt.tight_layout()
    plt.show()
    fig_name = "./analysis/node_pos_phrase_cn_sw.png"
    plt.savefig(fig_name)
    plt.clf()
    print(f"save {fig_name}")

    print(df1)
    print()
    print(df2)

def add_annoation(text,ax, x, y):
    for i, txt in enumerate(text):
        ax.annotate(s=txt, xy=(x[i], y[i]),  xytext=(x[i], y[i]), fontsize=14, ha='left', va='bottom')

def plot_phrase_tags(df, topN=10):
    df = df[:topN]
    fig, ax = plt.subplots(figsize=(5,5))
    N = np.arange(len(df['tags']))
    labels = df['tags']
    ax.set_xticks(N)
    ax.set_xticklabels(labels)
    ax.set_yscale('log')

    ax.bar(N + 0.00, df['CN'], color = 'C0', width = 0.25, label='ConceptNet')
    ax.bar(N + 0.25, df['SW'], color = 'C1', width = 0.25, label='SWOW')

    plt.legend()
    plt.gca().invert_xaxis()
    plt.tight_layout()

    plt.show()
    fig_name = "./analysis/node_phrase_cn_sw.png"
    plt.savefig(fig_name)
    plt.clf()
    print(f"save {fig_name}")


def plot_pair_tags_bar_groups(df, fig_name, topN=20, reload=False):
    
    labels = df['tags']
    indexes = np.arange(len(labels))
    width = 0.8
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(6,6))

    df1= df.loc[df['tag_group']=='PosPos'][:topN]
    N = np.arange(len(df1['tags']))
    labels = df1['tags']
    ax1.set_xticks(N)
    ax1.set_xticklabels(labels, rotation=90)
    ax1.set_yscale('log')
    # ax1.tick_params(labelsize=12)

    ax1.bar(N + 0.00, df1['CN'], color = 'C0', width = 0.25, label='ConceptNet')
    ax1.bar(N + 0.25, df1['SW'], color = 'C1', width = 0.25, label='SWOW')

    df2= df.loc[df['tag_group']=='PosPhrase'][:topN]
    # print()
    # print(df2)
    N = np.arange(len(df2['tags']))
    labels = df2['tags']
    ax2.set_xticks(N)
    ax2.set_xticklabels(labels, rotation=90)
    ax2.set_yscale('log')
    # ax2.tick_params(labelsize=12)
    ax2.bar(N + 0.00, df2['CN'], color = 'C0', width = 0.25, label='ConceptNet')
    ax2.bar(N + 0.25, df2['SW'], color = 'C1', width = 0.25, label='SWOW')

    df3= df.loc[df['tag_group']=='PhrasePhrase'][:topN]
    N = np.arange(len(df3['tags']))
    labels = df3['tags']
    ax3.set_xticks(N)
    ax3.set_xticklabels(labels, rotation=90)
    ax3.set_yscale('log')
    # plt.tick_params(labelsize=12)
    ax3.bar(N + 0.00, df3['CN'], color = 'C0', width = 0.25, label='ConceptNet')
    ax3.bar(N + 0.25, df3['SW'], color = 'C1', width = 0.25, label='SWOW')


    plt.tight_layout()
    plt.legend()
    # plt.yscale('log')
    # plt.xticks(indexes + width*0.1, labels, rotation=45)

    # for ax 
    plt.show()
    plt.savefig(fig_name)
    plt.clf()
    print("save {}".format(fig_name))


def main(args):
    print("Chunking concepts in ConceptNet")
    # doc = load_concepts(args.concept_path_cn, debug=args.debug)
    # get_pos_spacy(doc[0], "conceptnet", args.concept_pos_path_cn, debug=args.debug)
    # get_chunk_spacy(doc[1], "conceptnet", args.concept_chunk_path_cn)
    # chunk_pairs_statistics(args.triple_path_cn, "conceptnet", args.concept_pos_path_cn, args.concept_chunk_path_cn, args.triple_chunk_path_cn)
#  
    # print("Chunking concepts in SWOW")
    # doc = load_concepts(args.concept_path_sw, debug=args.debug)
    # get_pos_spacy(doc[0], "swow", args.concept_pos_path_sw, debug=args.debug)
    # get_chunk_spacy(doc[1], "swow", args.concept_chunk_path_sw)
    # chunk_pairs_statistics(args.triple_path_sw, "swow", args.concept_pos_path_sw,args.concept_chunk_path_sw, args.triple_chunk_path_sw)
    # sys.exit()

    # node_tags_to_csv(args)
    #pair_tags_to_csv(args)



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--concept_path_sw', type=str, default='data/swow/concept.txt')
    parser.add_argument('--triple_path_sw', type=str, default='data/swow/conceptnet.en.csv')
    parser.add_argument('--concept_pos_path_sw', type=str, default='data/analysis/swow/concept_pos.txt')
    parser.add_argument('--concept_chunk_path_sw', type=str, default='data/analysis/swow/concept_chunk.txt')
    parser.add_argument('--triple_chunk_path_sw', type=str, default='data/analysis/swow/triple_chunk.txt')

    parser.add_argument('--concept_path_cn', type=str, default='data/cpnet47rel/concept.txt')
    parser.add_argument('--triple_path_cn', type=str, default='data/cpnet47rel/conceptnet.en.csv')
    parser.add_argument('--concept_pos_path_cn', type=str, default='data/analysis/cpnet47rel/concept_pos.txt')
    parser.add_argument('--concept_chunk_path_cn', type=str, default='data/analysis/cpnet47rel/concept_chunk.txt')
    parser.add_argument('--triple_chunk_path_cn', type=str, default='data/analysis/cpnet47rel/triple_chunk.txt')

    parser.add_argument('--debug', type= bool_flag, default=False, nargs='?', help='small data debug')
    parser.add_argument('--reload', type= bool_flag, default=False, nargs='?', help='reload from a file for plotting')

    args = parser.parse_args()
    # args.debug = sys.argv[1] 

   # test_chunk1()
    print("debug={}".format(args.debug))
    main(args)
    
    # get_chunk()
    # test_chunk()


  
