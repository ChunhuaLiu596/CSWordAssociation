import os
os.environ['CUDA_VISIBLE_DEVICES'] = "4"
from OpenKE.config import Config
from OpenKE import models
#from OpenKE.openke.module import model as models
import numpy as np
import tensorflow as tf
import json
import argparse
import pathlib

parser = argparse.ArgumentParser()
parser.add_argument('opt_method', help='SGD/Adagrad/...')
parser.add_argument('pretrain', help='0/1', type=int)
parser.add_argument('in_path', help='deliver input data path')
parser.add_argument('kg_name', default='conceptnet', help='specify the knowledg graph')
parser.add_argument('--epoch', type=int, default=1000, help='how many epoch to train')
args = parser.parse_args()



def run():

    opt_method = args.opt_method
    int_pretrain = args.pretrain
    if int_pretrain == 1:
        pretrain = True
    elif int_pretrain == 0:
        pretrain = False
    else:
        raise ValueError('arg "pretrain" must be 0 or 1')


    # Download and preprocess ConcepNet

    config = Config()
    #config.set_in_path("./openke_data/{}/".format(args.kg_name))

    #config.set_in_path("embeddings/OpenKE/benchmarks/swow_3rel_freq1/")
    config.set_in_path(args.in_path)
    # config.set_test_link_prediction(True)
    # config.set_test_triple_classification(True)
    config.set_log_on(1)  # set to 1 to print the loss

    config.set_work_threads(30)
    config.set_train_times(args.epoch)  # number of iterations
    config.set_nbatches(512)  # batch size
    config.set_alpha(0.001)  # learning rate

    config.set_bern(0)
    config.set_dimension(100)
    config.set_margin(1.0)
    config.set_ent_neg_rate(1)
    config.set_rel_neg_rate(0)
    config.set_opt_method(opt_method)

    '''revision starts'''
    config.set_pretrain(pretrain)

    # Save the graph embedding every {number} iterations

    # OUTPUT_PATH = "./openke_data/embs/glove_initialized/"
    kg_names = args.kg_name.split("_")
    print("kg_names {}".format(kg_names))
    if pretrain:
        #if "swow"==kg_names[0]:
        #    OUTPUT_PATH = "../data/{}/{}/embs/glove_initialized/glove.".format(args.kg_name.split("_")[0], "_".join(kg_names[1:]))
        #else:
        OUTPUT_PATH = "data/{}/embs/glove_initialized/".format(args.kg_name)
    #else:
    #    if "swow" == kg_names[0]:
    #        OUTPUT_PATH = "../data/{}/{}/embs/xavier_initialized/".format(args.kg_name.split("_")[0], "_".join(kg_names[1:]))
    pathlib.Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
    print("The output path is: {}".format(OUTPUT_PATH))
    '''revision ends'''

    # Model parameters will be exported via torch.save() automatically.
    # Model parameters will be exported to json files automatically.
    # (Might cause IOError if the file is too large)
    config.set_out_files(OUTPUT_PATH + "glove.transe."+opt_method+".vec.json")

    print("Opt-method: %s" % opt_method)
    print("Pretrain: %d" % pretrain)
    config.init()
    print("set_model")
    config.set_model(models.TransE, args.kg_name)

    print("Begin training TransE")

    config.run()
    print(f"model will be saved at {config.out_path}")
    # config.test()


def init_predict(hs, ts, rs):

    '''
    # (1) Set import files and OpenKE will automatically load models via tf.Saver().
    con = Config()


    # con.set_in_path("OpenKE/benchmarks/FB15K/")
    con.set_in_path("openke_data/")
    # con.set_test_link_prediction(True)
    con.set_test_triple_classification(True)
    con.set_work_threads(8)
    con.set_dimension(100)


    # con.set_import_files("OpenKE/res/model.vec.tf")
    con.set_import_files("openke_data/embs/glove_initialized/glove.transe.SGD.pt")
    con.init()
    con.set_model(models.TransE)
    con.test()

    con.predict_triple(hs, ts, rs)

    # con.show_link_prediction(2,1)
    # con.show_triple_classification(2,1,3)
    '''

    # (2) Read model parameters from json files and manually load parameters.
    con = Config()
    #con.set_in_path("./openke_data/")
    con.set_test_triple_classification(True)
    con.set_test_link_prediction(True)
    con.set_work_threads(8)
    con.set_dimension(100)
    con.init()
    con.set_model(models.TransE)
    #f = open("./openke_data/embs/glove_initialized/glove.transe.SGD.vec.json", "r")
    f = open("./openke_data/swow/3rel_freq1/embs/glove_initialized/glove.transe.SGD.vec.json","r")
    content = json.loads(f.read())
    f.close()
    print("Loaded the pre-trained entity embeddings")
    con.set_parameters(content)
    print("Parameters are settled.")
    con.test()
    print("Finish test")

    # (3) Manually load models via tf.Saver().
    # con = config.Config()
    # con.set_in_path("./benchmarks/FB15K/")
    # con.set_test_flag(True)
    # con.set_work_threads(4)
    # con.set_dimension(50)
    # con.init()
    # con.set_model(models.TransE)
    # con.import_variables("./res/model.vec.tf")
    # con.test()


if __name__ == "__main__":


    run()
    #init_predict(2, 3, 5)
