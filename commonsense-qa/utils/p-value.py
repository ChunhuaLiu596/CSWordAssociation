#Student's t-test
import sys
import json
from numpy.random import seed
from numpy.random import randn
from scipy.stats import ttest_ind, ttest_rel
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
# seed the random number generator
seed(1)
# generate two independent samples
data1 = 5 * randn(100) + 50
data2 = 5 * randn(100) + 51

def compare_samples(data1, data2, verbose=False):
    stat, p = ttest_rel(data1, data2)
    if verbose:
        print('Statistics={}, p={}'.format(stat, p))

    # interpret: H0 denotes two models' predictions are the same
    alpha = 0.05
    if p > alpha:
        if verbose:
            print('Same distributions (fail to reject H0)+\n')
        rejectH0=False
    else:
        if verbose:
            print('Different distributions (reject H0) +\n')
        rejectH0=True
    return stat, p, rejectH0


def load_pred_gold(test_pred_path):
    preds = []
    labels = []
    dis =[]
    with open(test_pred_path, 'r', encoding='utf-8') as fin:
        for line in fin.readlines():
            js = json.loads(line.strip(), encoding='utf-8')
            labels.append(js['label'] )
            preds.append(js['pred_label'] )
            if js['pred_label']==js['label']:
                dis.append(1)
            else:
                dis.append(0)

    preds = np.array(preds)
    labels = np.array(labels)
    pred_one_hot = one_hot_label(preds)
    label_one_hot= one_hot_label(labels)
    dis = np.array(dis)
    acc = accuracy_score(labels, preds)
    print(f"accuracy: {acc} | {test_pred_path}")
    return dis 
    #return pred_one_hot, label_one_hot

def one_hot_label(a):
    b = np.zeros((a.size, a.max()+1))
    b[np.arange(a.size),a] = 1

    return b

def compute_p_value(cn_prediction, sw_prediction):
    cn_dis = load_pred_gold(cn_prediction)
    sw_dis = load_pred_gold(sw_prediction)
   
    # print("compare cn and sw")
    t, p, rejectH0=compare_samples(cn_dis, sw_dis)
    return t, p, rejectH0
   

def csqa_predictions():
    lm_predictions = [
        './saved_models/csqa/albert-xxlarge-v2_elr1e-5_dlr1e-3_d0.1_b16_s0_g0_None_aatt_pool_swow_entroberta_p1.0_0/predictions_test.json'
    ]
    cn_predictions=[
        './saved_models/csqa/ensemble-models/albert-xxlarge-v2_elr1e-5_dlr1e-3_d0.1_b16_s0_g0_pg_full_aatt_pool_cpnet_entroberta_p1.0_0/predictions_test.json',
        './saved_models/csqa/ensemble-models/albert-xxlarge-v2_elr1e-5_dlr1e-3_d0.1_b16_s23528_g0_pg_full_aatt_pool_cpnet_entroberta_p1.0_3/predictions_test.json',
        './saved_models/csqa/ensemble-models/albert-xxlarge-v2_elr1e-5_dlr1e-3_d0.1_b16_s24524_g0_pg_full_aatt_pool_cpnet_entroberta_p1.0_2/predictions_test.json',
    ]

    cn_predictions_dev=[
        './saved_models/csqa/ensemble-models/albert-xxlarge-v2_elr1e-5_dlr1e-3_d0.1_b16_s0_g0_pg_full_aatt_pool_cpnet_entroberta_p1.0_0/predictions_dev.json',
        './saved_models/csqa/ensemble-models/albert-xxlarge-v2_elr1e-5_dlr1e-3_d0.1_b16_s23528_g0_pg_full_aatt_pool_cpnet_entroberta_p1.0_3/predictions_dev.json', 
        './saved_models/csqa/ensemble-models/albert-xxlarge-v2_elr1e-5_dlr1e-3_d0.1_b16_s24524_g0_pg_full_aatt_pool_cpnet_entroberta_p1.0_2/predictions_dev.json',
    ]

    sw_predictions = [
        './saved_models/csqa/ensemble-models/albert-xxlarge-v2_elr1e-5_dlr1e-3_d0.1_b16_s19286_g0_pg_full_aatt_pool_swow_entroberta_p1.0_0/predictions_test.json', 
        './saved_models/csqa/ensemble-models/albert-xxlarge-v2_elr1e-5_dlr1e-3_d0.1_b16_s11010_g0_pg_full_aatt_pool_swow_entroberta_p1.0_0/predictions_test.json', 
        './saved_models/csqa/ensemble-models/albert-xxlarge-v2_elr1e-5_dlr1e-3_d0.1_b16_s0_g0_pg_full_aatt_pool_swow_entroberta_p1.0_0/predictions_test.json', 
    ]
        # './saved_models/csqa/ensemble-models/albert-xxlarge-v2_elr1e-5_dlr1e-3_d0.1_b16_s21527_g0_pg_full_aatt_pool_swow_entroberta_p1.0_1/predictions_test.json'
        # './saved_models/csqa/ensemble-models/albert-xxlarge-v2_elr1e-5_dlr1e-3_d0.1_b16_s11010_g0_pg_full_aatt_pool_swow_entroberta_p1.0_1/predictions_test.json', 


    sw_predictions_dev = [
        './saved_models/csqa/ensemble-models/albert-xxlarge-v2_elr1e-5_dlr1e-3_d0.1_b16_s19286_g0_pg_full_aatt_pool_swow_entroberta_p1.0_0/predictions_dev.json', 
        './saved_models/csqa/ensemble-models/albert-xxlarge-v2_elr1e-5_dlr1e-3_d0.1_b16_s11010_g0_pg_full_aatt_pool_swow_entroberta_p1.0_0/predictions_dev.json', 
        './saved_models/csqa/ensemble-models/albert-xxlarge-v2_elr1e-5_dlr1e-3_d0.1_b16_s0_g0_pg_full_aatt_pool_swow_entroberta_p1.0_0/predictions_dev.json', 
    ]
        # './saved_models/csqa/ensemble-models/albert-xxlarge-v2_elr1e-5_dlr1e-3_d0.1_b16_s11010_g0_pg_full_aatt_pool_swow_entroberta_p1.0_1/predictions_dev.json', 
        # './saved_models/csqa/ensemble-models/albert-xxlarge-v2_elr1e-5_dlr1e-3_d0.1_b16_s21527_g0_pg_full_aatt_pool_swow_entroberta_p1.0_1/predictions_dev.json'



    return lm_predictions, cn_predictions, sw_predictions, cn_predictions_dev, sw_predictions_dev

def obqa_predictions():
    lm_predictions=[
        './saved_models/obqa/albert-xxlarge-v2_elr1e-5_dlr3e-4_d0.2_b32_s31415_g0_None_ano_rel_swow_entroberta_p1.0_0/predictions_test.json',
        './saved_models/obqa/albert-xxlarge-v2_elr1e-5_dlr3e-4_d0.2_b32_s0_g0_None_ano_rel_swow_entroberta_p1.0_0/predictions_test.json',
        './saved_models/obqa/albert-xxlarge-v2_elr1e-5_dlr3e-4_d0.2_b32_s4989_g0_None_ano_rel_swow_entroberta_p1.0_1/predictions_test.json',
    ]
    sw_predictions = [
        './saved_models/obqa/ensemble-models/albert-xxlarge-v2_elr1e-5_dlr3e-4_d0.2_b32_s10806_g0_pg_full_aatt_pool_swow_entroberta_p1.0_3/predictions_test.json',
        './saved_models/obqa/ensemble-models/albert-xxlarge-v2_elr1e-5_dlr3e-4_d0.2_b32_s17036_g0_pg_full_aatt_pool_swow_entroberta_p1.0_5/predictions_test.json',
        './saved_models/obqa/ensemble-models/albert-xxlarge-v2_elr1e-5_dlr3e-4_d0.2_b32_s3469_g0_pg_full_aatt_pool_swow_entroberta_p1.0_0/predictions_test.json',
    ]

    sw_predictions_dev = [
        './saved_models/obqa/ensemble-models/albert-xxlarge-v2_elr1e-5_dlr3e-4_d0.2_b32_s10806_g0_pg_full_aatt_pool_swow_entroberta_p1.0_3/predictions_dev.json',
        './saved_models/obqa/ensemble-models/albert-xxlarge-v2_elr1e-5_dlr3e-4_d0.2_b32_s17036_g0_pg_full_aatt_pool_swow_entroberta_p1.0_5/predictions_dev.json',
        './saved_models/obqa/ensemble-models/albert-xxlarge-v2_elr1e-5_dlr3e-4_d0.2_b32_s3469_g0_pg_full_aatt_pool_swow_entroberta_p1.0_0/predictions_dev.json'
    ]
    
    cn_predictions = [
       './saved_models/obqa/ensemble-models/albert-xxlarge-v2_elr1e-5_dlr3e-4_d0.2_b32_s8625_g0_pg_full_aatt_pool_cpnet_entroberta_p1.0_1/predictions_test.json',
       './saved_models/obqa/ensemble-models/albert-xxlarge-v2_elr1e-5_dlr3e-4_d0.2_b32_s14846_g0_pg_full_aatt_pool_cpnet_entroberta_p1.0_5/predictions_test.json',
       './saved_models/obqa/ensemble-models/albert-xxlarge-v2_elr1e-5_dlr3e-4_d0.2_b32_s5863_g0_pg_full_aatt_pool_cpnet_entroberta_p1.0_4/predictions_test.json'
    ]

    cn_predictions_dev = [
        './saved_models/obqa/ensemble-models/albert-xxlarge-v2_elr1e-5_dlr3e-4_d0.2_b32_s8625_g0_pg_full_aatt_pool_cpnet_entroberta_p1.0_1/predictions_dev.json',
        './saved_models/obqa/ensemble-models/albert-xxlarge-v2_elr1e-5_dlr3e-4_d0.2_b32_s14846_g0_pg_full_aatt_pool_cpnet_entroberta_p1.0_5/predictions_dev.json',
        './saved_models/obqa/ensemble-models/albert-xxlarge-v2_elr1e-5_dlr3e-4_d0.2_b32_s5863_g0_pg_full_aatt_pool_cpnet_entroberta_p1.0_4/predictions_dev.json',
    ]

    return lm_predictions, cn_predictions, sw_predictions, cn_predictions_dev, sw_predictions_dev

def mcscript_predictions():
    lm_predictions = [
        './saved_models/mcscript/albert-xxlarge-v2_elr1e-5_dlr1e-3_d0.1_b16_s0_g0_None_aatt_pool_swow_entroberta_p1.0_0/predictions_test.json',
    ]
    sw_predictions = [] 
    cn_predictions = []


def model_pvalue(prediction1, prediction2, model1, model2):
    '''
    H0 (null hypothesis): two models are similar 
    H1: two models are different 
    '''
    print(f"Comparing {model1} with {model2}")
    results=[]
    for i, x in enumerate(prediction1):
        for j, y in enumerate(prediction2):
            t, p, r=compute_p_value(x, y)
            results.append([i, j, t, p, r])

    df = pd.DataFrame(results, columns=[f"{model1}", f"{model2}", "t", "p", "rejectedH0"])

    rejectH0_rate = len(df.query('rejectedH0==True').index) / len(df.index)
    print(df)
    print(f"Reject rate: {rejectH0_rate}" )
    return df

def get_best_model(predictions):

    for i, pred in enumerate(predictions):
        print(i)
        load_pred_gold(pred) 

if __name__=='__main__':
    dataset=sys.argv[1]
    output_path = sys.argv[2]

    print(dataset)
    if dataset=='csqa':
        lm_predictions, cn_predictions, sw_predictions, cn_predictions_dev, sw_predictions_dev = csqa_predictions()
        # get_best_model(lm_predictions)
        # get_best_model(cn_predictions)
        # get_best_model(sw_predictions)

        df = model_pvalue([lm_predictions[0]], [cn_predictions[0]], 'albert','albert-cn')
        df = model_pvalue([lm_predictions[0]], [sw_predictions[0]], 'albert','albert-sw')
        df = model_pvalue([cn_predictions[0]], [sw_predictions[0]], 'albert-cn', 'albert-sw')

    elif dataset=='obqa':
        lm_predictions, cn_predictions, sw_predictions, cn_predictions_dev, sw_predictions_dev = obqa_predictions()

        # get_best_model(lm_predictions)
        # get_best_model(cn_predictions)
        # get_best_model(sw_predictions)
        # print("debug", cn_predictions[2])
        df = model_pvalue([lm_predictions[2]], [cn_predictions[2]], 'albert','albert-cn')
        df = model_pvalue([lm_predictions[2]], [sw_predictions[1]], 'albert','albert-sw')
        df = model_pvalue([cn_predictions[2]], [sw_predictions[1]], 'albert-cn', 'albert-sw')

    # df.to_csv(output_path)
    # print(f"save results to {output_path}")
    #compute_p_value(cn_predictions[0], sw_predictions[0])
    #compute_p_value(cn_predictions[1], sw_predictions[1])
    #compute_p_value(cn_predictions[2], sw_predictions[2])
