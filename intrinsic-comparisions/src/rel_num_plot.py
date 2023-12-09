import os, sys
import nltk
from nltk import Tree
import json
from tqdm import tqdm
import spacy   #python -m spacy download en_core_web_sm
import textacy #conda install -c conda-forge textacy
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import benepar #pip install benepar
from benepar.spacy_plugin import BeneparComponent
from multiprocessing import Pool
import argparse
from pos_utils import * 
import pandas as pd
import seaborn as sns

def plot_ablation_relation_num():
    path1= "./analysis/ablation_relation_num_cn.csv"
    path2= "./analysis/ablation_relation_num_sw.csv"

    df1 = pd.read_csv(path1).T
    df2 = pd.read_csv(path2).T
    plot(df1, df2)

def plot(df1, df2, topN=10):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10,5))

    labels = ["17","7","1","0"] 
    N = np.arange(len(labels))
    ax1.set_xticks(N)
    ax1.set_xticklabels(labels)
    ax1.set_xlabel('The number of relation types on ConceptNet')

    ax1.set_ylim(ymin=70, ymax=78)
    # ax1.set_yscale('log')
    ax1.set_ylabel('Accuracy')
    # ax1.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))

    # xlabels= df1['Models']
    # print(df1[0][0])
    ax1.bar(N + 0.00, df1[0][1:], color = 'C0', width = 0.25, label="RN")
    ax1.bar(N + 0.25, df1[1][1:], color = 'C1', width = 0.25, label="PG-Full")
    ax1.legend()
    ax1.set_title("Ablation on the number of relationt types.")


    labels = ["2","1","0"] 
    N = np.arange(len(labels))
    ax2.set_xticks(N)
    ax2.set_xticklabels(labels )
    ax2.set_xlabel('The number of relation types on SWOW')

    ax2.set_ylim(ymin=70, ymax=78)
    ax2.set_ylabel('Accuracy')
    # ax2.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))

    ax2.bar(N + 0.00, df2[0][1:], color = 'C0', width = 0.25, label="RN")
    ax2.bar(N + 0.25, df2[1][1:], color = 'C1', width = 0.25, label="PG-Full")
    ax2.legend()
    ax2.set_title("Ablation on the number of relationt types.")
    # for axis in [ax.xaxis, ax.yaxis]:
        # axis.set_major_formatter(FormatStrFormatter('%.2f'))

    # plt.legend()
    plt.tight_layout()
    plt.show()
    fig_name = "./analysis/ablation_rel_num_cn_sw.png"
    plt.savefig(fig_name)
    plt.clf()
    print(f"save {fig_name}")

    print(df1)
    print()
    print(df2)


plot_ablation_relation_num()