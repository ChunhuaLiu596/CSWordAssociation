# from main import *
import json
import numpy as np
import sys

def obqa_predictions():

    sw_predictions = [
        './saved_models/obqa/ensemble-models/albert-xxlarge-v2_elr1e-5_dlr3e-4_d0.2_b32_s10806_g0_pg_full_aatt_pool_swow_entroberta_p1.0_3/predictions_test.json',
        './saved_models/obqa/ensemble-models/albert-xxlarge-v2_elr1e-5_dlr3e-4_d0.2_b32_s17036_g0_pg_full_aatt_pool_swow_entroberta_p1.0_5/predictions_test.json',
        './saved_models/obqa/ensemble-models/albert-xxlarge-v2_elr1e-5_dlr3e-4_d0.2_b32_s3469_g0_pg_full_aatt_pool_swow_entroberta_p1.0_0/predictions_test.json',
    ]

    sw_predictions_dev = [
        './saved_models/obqa/ensemble-models/albert-xxlarge-v2_elr1e-5_dlr3e-4_d0.2_b32_s10806_g0_pg_full_aatt_pool_swow_entroberta_p1.0_3/predictions_dev.json',
        './saved_models/obqa/ensemble-models/albert-xxlarge-v2_elr1e-5_dlr3e-4_d0.2_b32_s17036_g0_pg_full_aatt_pool_swow_entroberta_p1.0_5/predictions_dev.json',
        './saved_models/obqa/ensemble-models/albert-xxlarge-v2_elr1e-5_dlr3e-4_d0.2_b32_s3469_g0_pg_full_aatt_pool_swow_entroberta_p1.0_0/predictions_dev.json',
    ]
    
    cn_predictions = [
       './saved_models/obqa/ensemble-models/albert-xxlarge-v2_elr1e-5_dlr3e-4_d0.2_b32_s8625_g0_pg_full_aatt_pool_cpnet_entroberta_p1.0_1/predictions_test.json',
       './saved_models/obqa/ensemble-models/albert-xxlarge-v2_elr1e-5_dlr3e-4_d0.2_b32_s14846_g0_pg_full_aatt_pool_cpnet_entroberta_p1.0_5/predictions_test.json',
       './saved_models/obqa/ensemble-models/albert-xxlarge-v2_elr1e-5_dlr3e-4_d0.2_b32_s5863_g0_pg_full_aatt_pool_cpnet_entroberta_p1.0_4/predictions_test.json',
    ]

    cn_predictions_dev = [
        './saved_models/obqa/ensemble-models/albert-xxlarge-v2_elr1e-5_dlr3e-4_d0.2_b32_s8625_g0_pg_full_aatt_pool_cpnet_entroberta_p1.0_1/predictions_dev.json',
        './saved_models/obqa/ensemble-models/albert-xxlarge-v2_elr1e-5_dlr3e-4_d0.2_b32_s14846_g0_pg_full_aatt_pool_cpnet_entroberta_p1.0_5/predictions_dev.json',
        './saved_models/obqa/ensemble-models/albert-xxlarge-v2_elr1e-5_dlr3e-4_d0.2_b32_s5863_g0_pg_full_aatt_pool_cpnet_entroberta_p1.0_4/predictions_dev.json',
    ]

    return cn_predictions, sw_predictions, cn_predictions_dev, sw_predictions_dev

def csqa_predictions():
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
    return cn_predictions, sw_predictions, cn_predictions_dev, sw_predictions_dev

def ensemble_logits(pred_paths):
    all_logits = []
    for path in pred_paths:
        logits, labels = load_logits(path)
        all_logits.append(logits)
    avg_logits = average_logits(all_logits)
    acc = evaluate_accuracy(labels, avg_logits)
    print("ensemble acc= {:.4f}".format(acc))

def load_logits(test_pred_path):
    logits = []
    labels = []
    with open(test_pred_path, 'r', encoding='utf-8') as fin:
        for line in fin.readlines():
            # js = json.loads(line.strip(), encoding='utf-8')
            js = json.loads(line.strip())
            labels.append(js['label'] )
            logits_line = np.array(js['logits'])
            logits.append(logits_line)
    return np.array(logits), labels

def evaluate_accuracy(labels, logits):
    n_samples, n_correct = 0, 0
    n_correct += (logits.argmax(1) == labels).sum().item()
    n_samples += len(labels)
    return n_correct / n_samples

def average_logits(all_logits):
    # all_logits = np.array(np.array([logits for logits in all_logits])) #(3, l_q, 5)
    # print(all_logits)
    all_logits = np.array(all_logits)
    # print(all_logits.shape)
    all_logits = np.swapaxes(all_logits,0,1) #(l_q, 3, 5)
    avg_logits = np.average(all_logits, axis=1)
    return avg_logits

if __name__=='__main__':
    dataset=sys.argv[1]
    if dataset=='csqa':
        cn_predictions, sw_predictions, cn_predictions_dev, sw_predictions_dev = csqa_predictions()

    elif dataset=='obqa':
        cn_predictions, sw_predictions, cn_predictions_dev, sw_predictions_dev = obqa_predictions()

    print("Ensemble on dev (CN3, SW3, CN3+SW3)")
    ensemble_logits(cn_predictions_dev)
    ensemble_logits(sw_predictions_dev)
    ensemble_logits(cn_predictions_dev + sw_predictions_dev)

    print()
    print("Ensemble on test (CN3, SW3, CN3+SW3)")
    ensemble_logits(cn_predictions)
    ensemble_logits(sw_predictions)
    ensemble_logits(cn_predictions + sw_predictions)

    # print("Ensemble CN3+SW1 (seed=0)")
    # ensemble_logits(cn_predictions_dev + [sw_predictions_dev[2]])
    # ensemble_logits(cn_predictions + [sw_predictions[2]])

    # print("Ensemble CN3+SW1 (seed=19286)")
    # ensemble_logits(cn_predictions_dev + [sw_predictions_dev[0]])
    # ensemble_logits(cn_predictions + [sw_predictions[0]])

    # print("Ensemble CN3+SW1 (seed=11010)")
    # ensemble_logits(cn_predictions_dev + [sw_predictions_dev[1]])
    # ensemble_logits(cn_predictions + [sw_predictions[1]])


    # print("Ensemble CN1+SW1 (seed=11010)")
    # ensemble_logits([cn_predictions_dev[0]] + [sw_predictions_dev[0]])
    # ensemble_logits([cn_predictions[0]] + [sw_predictions[0]])
