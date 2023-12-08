import sys
import json
from  tqdm  import tqdm
from numpy.random import seed
from numpy.random import randn
from scipy.stats import ttest_ind, ttest_rel
import numpy as np
import pandas as pd
from collections import Counter
import copy

# seed the random number generator
seed(1)
# generate two independent samples
data1 = 5 * randn(100) + 50
data2 = 5 * randn(100) + 51


def load_pred_gold(test_pred_path, qid2type):
    preds = []
    labels = []
    qtype2count=Counter()
    qtype2correct=Counter()
    with open(test_pred_path, 'r', encoding='utf-8') as fin:
        for line in fin.readlines():
            js = json.loads(line.strip(), encoding='utf-8')
            labels.append(js['label'] )
            preds.append(js['pred_label'] )

            if js['pred_label']==js['label']:
                qtype2correct[qid2type.get(js['qid'])] +=1
            qtype2count[qid2type.get(js['qid'])] +=1

    qtype2acc = [(k, qtype2correct.get(k,0)/v, v) for k,v in qtype2count.most_common()]
    # qtypes_count = copy.deepcopy(qtype2count).most_common()
    qtype2acc = list(zip(*qtype2acc))
    # qtypes = list(qtype2count.keys())
    # qtype2acc = [round(qtype2correct.get(qtype,0)/qtype2count(qtype),2) for qtype in qtypes ]

    acc = sum(qtype2correct.values())/sum(qtype2count.values())
    print("Overall acc: {}".format(acc))
    # print("Qtype acc: {} ".format(qtype2acc))
    df = pd.DataFrame({
        "qtype": qtype2acc[0], "accuracy": qtype2acc[1],"count":qtype2acc[2]}, columns=["qtype", "accuracy","count"])
    print(df, sep=",")
    print()
    # compute_acc_for_types(df)
    # return dis 
    #return pred_one_hot, label_one_hot

def compute_acc_for_types(df):
    df1 = df.groupby(["qtype", "correct"]).count()/df.groupby('qtype').sum()
    # df1 = df.loc[df["correct"]==1]
    # col_list = df1.columns.values.tolist()
    # print(col_list)
    # df2 = df1/df1.groupby('qtype').sum()
    # print(df1.loc("correct"==1).sum())
    # print(df2)
    df1.to_csv("./analysis/test_qtype_pred.csv")

def load_question_type(path):
    # global qid2type
    qid2type = {}
    with open(path, 'r') as fin:
        for line in fin.readlines():
            qidjson = json.loads(line)
            qid2type[qidjson.get("id")]= qidjson.get("type")
    return qid2type

def compute_qtype_accuracy(cn_prediction, qid2type):
    load_pred_gold(cn_prediction, qid2type)
    # sw_dis = load_pred_gold(sw_prediction)

    # print("compare cn and sw")
    # t, p=compare_samples(cn_dis, sw_dis)
    # return t, p
   

def csqa_predictions():
    sw_predictions = [
        './saved_models/csqa/albert-xxlarge-v2_elr1e-5_dlr1e-3_d0.1_b16_s19286_g0_pg_full_aatt_pool_swow_entroberta_p1.0_0/predictions_test.json', 
        './saved_models/csqa/albert-xxlarge-v2_elr1e-5_dlr1e-3_d0.1_b16_s11010_g0_pg_full_aatt_pool_swow_entroberta_p1.0_0/predictions_test.json', 
        './saved_models/csqa/albert-xxlarge-v2_elr1e-5_dlr1e-3_d0.1_b16_s0_g0_pg_full_aatt_pool_swow_entroberta_p1.0_0/predictions_test.json', 
    ]
        # './saved_models/csqa/albert-xxlarge-v2_elr1e-5_dlr1e-3_d0.1_b16_s21527_g0_pg_full_aatt_pool_swow_entroberta_p1.0_1/predictions_test.json'
        # './saved_models/csqa/albert-xxlarge-v2_elr1e-5_dlr1e-3_d0.1_b16_s11010_g0_pg_full_aatt_pool_swow_entroberta_p1.0_1/predictions_test.json', 


    sw_predictions_dev = [
        './saved_models/csqa/albert-xxlarge-v2_elr1e-5_dlr1e-3_d0.1_b16_s19286_g0_pg_full_aatt_pool_swow_entroberta_p1.0_0/predictions_dev.json', 
        './saved_models/csqa/albert-xxlarge-v2_elr1e-5_dlr1e-3_d0.1_b16_s11010_g0_pg_full_aatt_pool_swow_entroberta_p1.0_0/predictions_dev.json', 
        './saved_models/csqa/albert-xxlarge-v2_elr1e-5_dlr1e-3_d0.1_b16_s0_g0_pg_full_aatt_pool_swow_entroberta_p1.0_0/predictions_dev.json', 
    ]
        # './saved_models/csqa/albert-xxlarge-v2_elr1e-5_dlr1e-3_d0.1_b16_s11010_g0_pg_full_aatt_pool_swow_entroberta_p1.0_1/predictions_dev.json', 
        # './saved_models/csqa/albert-xxlarge-v2_elr1e-5_dlr1e-3_d0.1_b16_s21527_g0_pg_full_aatt_pool_swow_entroberta_p1.0_1/predictions_dev.json'


    cn_predictions=[
        './saved_models/csqa/albert-xxlarge-v2_elr1e-5_dlr1e-3_d0.1_b16_s0_g0_pg_full_aatt_pool_cpnet_entroberta_p1.0_0/predictions_test.json',
        './saved_models/csqa/albert-xxlarge-v2_elr1e-5_dlr1e-3_d0.1_b16_s23528_g0_pg_full_aatt_pool_cpnet_entroberta_p1.0_3/predictions_test.json',
        './saved_models/csqa/albert-xxlarge-v2_elr1e-5_dlr1e-3_d0.1_b16_s24524_g0_pg_full_aatt_pool_cpnet_entroberta_p1.0_2/predictions_test.json',
    ]

    cn_predictions_dev=[
        './saved_models/csqa/albert-xxlarge-v2_elr1e-5_dlr1e-3_d0.1_b16_s0_g0_pg_full_aatt_pool_cpnet_entroberta_p1.0_0/predictions_dev.json',
        './saved_models/csqa/albert-xxlarge-v2_elr1e-5_dlr1e-3_d0.1_b16_s23528_g0_pg_full_aatt_pool_cpnet_entroberta_p1.0_3/predictions_dev.json', 
        './saved_models/csqa/albert-xxlarge-v2_elr1e-5_dlr1e-3_d0.1_b16_s24524_g0_pg_full_aatt_pool_cpnet_entroberta_p1.0_2/predictions_dev.json',
    ]
    return cn_predictions, sw_predictions, cn_predictions_dev, sw_predictions_dev



if __name__=='__main__':
    # dataset=sys.argv[1]
    # print(dataset)

    qtype_path="./data/csqa/lin_test_split_qtype.jsonl"
    qid2type = load_question_type(qtype_path)

    cn_predictions, sw_predictions, cn_predictions_dev, sw_predictions_dev = csqa_predictions()

    # results=[]
    print("cn")
    for i, x in enumerate(cn_predictions):
        compute_qtype_accuracy(x, qid2type)

    print("sw")
    for i, x in enumerate(sw_predictions):
        compute_qtype_accuracy(x, qid2type)
    
    # df = pd.DataFrame(results, columns=["cn_model", "sw_model", "t", "p"])
    # print(df)
    #compute_qtype_accuracy(cn_predictions[0], sw_predictions[0])
    #compute_qtype_accuracy(cn_predictions[1], sw_predictions[1])
    #compute_qtype_accuracy(cn_predictions[2], sw_predictions[2])
