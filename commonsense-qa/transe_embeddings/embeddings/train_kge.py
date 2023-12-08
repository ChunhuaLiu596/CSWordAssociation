import os
os.environ['CUDA_VISIBLE_DEVICES'] = "5"
from OpenKE.config import Config
from OpenKE import models
import numpy as np
import tensorflow as tf
import json
import argparse
import pathlib

parser = argparse.ArgumentParser()
parser.add_argument('model', default='TransE', help='specify model for training')
parser.add_argument('opt_method', help='SGD/Adagrad/...')
parser.add_argument('pretrain', help='0/1', type=int)
parser.add_argument('in_path', help='deliver input data path')
parser.add_argument('kg_name', default='conceptnet', help='specify the knowledg graph')
parser.add_argument('--epoch', type=int, default=1000, help='how many epoch to train')
args = parser.parse_args()


model_mapping = {
    "TransE": models.TransE,
    "TransH": models.TransH,
    "HolE": models.HolE,
    "DistMult":models.DistMult,
}

lr_mapping={
    "TransE":0.001,
    "TransH":0.001,
    "HolE":0.1,
    "DistMult":0.1,
}
margin_maping={
     "TransE": 1.0,
    "TransH":1.0,
    "HolE":0.2,
    "DistMult":0.1,
}

def run():

    opt_method = args.opt_method
    model = args.model
    int_pretrain = args.pretrain
    if int_pretrain == 1:
        pretrain = True
    elif int_pretrain == 0:
        pretrain = False
    else:
        raise ValueError('arg "pretrain" must be 0 or 1')


    # Download and preprocess ConcepNet

    config = Config()
    config.set_in_path(args.in_path)
    # config.set_test_link_prediction(True)
    # config.set_test_triple_classification(True)
    config.set_log_on(1)  # set to 1 to print the loss

    config.set_work_threads(30)
    config.set_train_times(args.epoch)  # number of iterations
    config.set_nbatches(512)  # batch size
    config.set_alpha(lr_mapping[model])  # learning rate

    config.set_bern(0)
    config.set_dimension(100)
    if model !='DistMult':
        config.set_margin(margin_maping[model])
    config.set_ent_neg_rate(1)
    config.set_rel_neg_rate(0)
    config.set_opt_method(opt_method)

    '''revision starts'''
    config.set_pretrain(pretrain)

    # Save the graph embedding every {number} iterations

    # OUTPUT_PATH = "./openke_data/embs/glove_initialized/"
    if pretrain:
        OUTPUT_PATH = "data/{}/embs/glove_initialized/".format(args.kg_name)
    pathlib.Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
    print("The output path is: {}".format(OUTPUT_PATH))
    '''revision ends'''

    # Model parameters will be exported via torch.save() automatically.
    # Model parameters will be exported to json files automatically.
    # (Might cause IOError if the file is too large)
    output_file = OUTPUT_PATH + f"glove.{model}."+opt_method+".pt"
    config.set_export_files(output_file)

    output_file = OUTPUT_PATH + f"glove.{model}."+opt_method+".vec.json"
    config.set_out_files(output_file)

    print("Model: %s" % model)
    print("Opt-method: %s" % opt_method)
    print("Pretrain: %d" % pretrain)
    config.init()
    print("set_model")
    #config.set_model(models.{}.format(args.model), args.kg_name)
    config.set_model(model_mapping[model], args.kg_name)

    print(f"Begin training {model}")

    config.run()
    print(f"model will be saved at {config.out_path}")
    # config.test()


if __name__ == "__main__":


    run()
    #init_predict(2, 3, 5)
