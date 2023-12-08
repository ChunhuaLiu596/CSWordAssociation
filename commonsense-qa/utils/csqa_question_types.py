import torch
import pickle
from tqdm import tqdm
import json
import argparse
import os
# from paths import * 
import random 
import networkx as nx

try:
    from .utils import check_path
except:
    from utils import check_path

def inhouse_train_qids():
    path = "./data/csqa/inhouse_split_qids.txt"
    global train_qid
    with open(path, 'r') as fin:
        train_qid = [line.strip() for line in fin.readlines()]
    return set(train_qid)
         

def recognize_question_types(qa_file: str, output_file: str, graph_file:str, ans_pos: bool=False):
    '''
    Params:
    qa_file: statement file 
    output_file: json dict{q_id, q_type}
    '''
    global cpnet,train_qid
    # , cpnet_simple, concept2id, id2concept, relation2id, id2relation
    cpnet = load_graph(graph_file)
    train_qid=inhouse_train_qids()

    print(f'identifying {qa_file} question types ...')
    nrow = sum(1 for _ in open(qa_file, 'r'))

    with open(output_file, 'w') as output_handle, open(qa_file, 'r') as qa_handle:
        # print("Writing to {} from {}".format(output_file, qa_file))
        for line in tqdm(qa_handle, total=nrow):
            json_line = json.loads(line)
            output_dict = qajson_to_type(json_line)
            if output_dict["id"] not in train_qid:
                output_handle.write(json.dumps(output_dict))
                output_handle.write("\n")
    print(f'converted statements saved to {output_file}')
    print()

def qajson_to_type(qa_json: dict):
    stem = qa_json["question"]["stem"]
    question_concept = qa_json["question"]["question_concept"]
    answerKey= qa_json.get("answerKey", "A")
    choices = qa_json["question"]["choices"]
    for cho in choices:
        if cho["label"]== answerKey:
            answer_text = cho["text"]

    question_concept = "_".join(question_concept.split(" "))
    answer_concept= "_".join(answer_text.split(" "))
    relation = find_rels_qa_concept_pair(question_concept, answer_concept )
    output={
        "id": qa_json.get("id"),
        "type": relation,
        "question_concept":question_concept, 
        "answer_concept":answer_concept,
        "stem": stem
    }
    return output

def load_graph(path):
    graph = nx.MultiDiGraph()
    with open(path, "r") as fo:
        for line in tqdm(fo.readlines()):
            rel, subj, obj, weight = line.strip().split("\t")
            if graph.has_edge(subj, obj):
                if rel not in graph[subj][obj]:
                    # print(rel, subj, obj)
                    graph.add_edge(subj, obj, key=rel, rel=rel, weight=weight)
            else:
                graph.add_edge(subj, obj, key=rel, rel=rel,weight=weight)
            # graph.add_edge(subj, obj, rel=rel, weight=weight)
            # graph.add_edge(obj, subj, rel="_"+rel, weight=weight)
    return graph


def find_rels_qa_concept_pair(source: str, target: str, ifprint=True):
    """
    find paths for a (question concept, answer concept) pair
    source and target is text
    """
   
    global cpnet

    if source not in cpnet.nodes() or target not in cpnet.nodes():
        return
    if cpnet.has_edge(source, target):
        rel_list = cpnet[source][target]
        print(rel_list)
        if len(rel_list)>1:
            rel = list(random.sample(list(rel_list), 1))[0]
            print(rel)
        else:
            rel = list(rel_list.keys())[0]
    else:
        rel=None
    return rel

def main():
    graph_path="./data/cpnet/conceptnet.en.csv"
    train_test_qa_json = "./data/csqa/train_rand_split.jsonl"
    outout_qa_json = "data/csqa/lin_test_split_qtype.jsonl"
    recognize_question_types(train_test_qa_json, outout_qa_json, graph_path)

if __name__=='__main__':
    main()



