import random 
from itertools import chain
import os

class KCrossFold(object):
    def __init__(self, dataset, k=5):
        self.dataset = dataset
        self.kfolds = self.create_kfolds(self.dataset, k)

    def generate_721_5fold(self, out_dir):
        out_dir= out_dir + "/{}".format("721_5fold")
        if not os.path.exists(out_dir):
                os.mkdir(out_dir)

        test_valid_idxs = [[0,1,2,3],[1,2,3,4],[2,3,4,0],[3,4,0,1],[4,0,1,2]]
        train_idxs =[[4],[0],[1],[2],[3]]
        for i in range(5):
            out_path= out_dir + "/{}".format(i+1)
            if not os.path.exists(out_path):
                print("Creating fold {} ..".format(i+1))
                os.mkdir(out_path)
            else:
                print("Generating fold {} ..".format(i+1))

            train_idx = train_idxs[i] 
            train = list(chain.from_iterable([self.kfolds[i] for i in train_idx]))

            test_valid_idx = test_valid_idxs[i] 
            test_valid = list(chain.from_iterable([self.kfolds[i] for i in test_valid_idx]))
            test = test_valid[:-int(len(test_valid)/8)]
            valid = test_valid[-int(len(test_valid)/8):]
            self.write_splits(test, train, valid, out_path)
            print("Aligned entities: test: {} train: {} valid: {}".format(
                                len(test), len(train), len(valid)))
    def generate_271_5fold(self, out_dir):
        out_dir= out_dir + "/{}".format("271_5fold")
        if not os.path.exists(out_dir):
                os.mkdir(out_dir)

        train_valid_idxs = [[0,1,2,3],[1,2,3,4],[2,3,4,0],[3,4,0,1],[4,0,1,2]]
        test_idxs =[[4],[0],[1],[2],[3]]
        for i in range(5):
            out_path= out_dir + "/{}".format(i+1)
            if not os.path.exists(out_path):
                print("Creating fold {} ..".format(i+1))
                os.mkdir(out_path)
            else:
                print("Generating fold {} ..".format(i+1))

            test_idx = test_idxs[i] 
            test = list(chain.from_iterable([self.kfolds[i] for i in test_idx]))

            train_valid_idx = train_valid_idxs[i] 
            train_valid = list(chain.from_iterable([self.kfolds[i] for i in train_valid_idx]))
            train = train_valid[:-int(len(train_valid)/8)]
            valid = train_valid[-int(len(train_valid)/8):]
            
            self.write_splits(test, train, valid, out_path)
            print("Aligned entities: test: {} train: {} valid: {}".format(
                                len(test), len(train), len(valid)))

    def write_splits(self, test, train, valid, out_path):
        self.write_tuples(test, out_path+'/test_links')
        self.write_tuples(train, out_path+'/train_links')
        self.write_tuples(valid, out_path+'/valid_links')


    def write_tuples(self, inp, out_path):
        print("Write {}".format(out_path))
        with open(out_path,"w") as f:
            for x in inp:
                out_line="{}\t{}\n".format(x[0],x[1])
                f.write(out_line)
        
    def create_kfolds(self, dataset, k=5):
        """ 
        get from:https://raw.githubusercontent.com/narenkmanoharan/K-Cross-fold-Validation/master/k_fold.py
        Gets the dataset and generates the training and testing data splices using the K-fold cross validation
    
        :param dataset:
            dataset - list[list]: Dataset which contains the attributes and classes.
    
                Example: [[0.23, 0.34, 0.33, 0.12, 0.45, 0.68, 'cp'], [0.13, 0.35, 0.01, 0.72, 0.25, 0.08, 'pp'], .... ]
    
        :param k:
            k - int: numbers of fold in the K-Fold cross validation (default value = 10)
    
        Yields:
            train_test_split - array[list[list]]: Contains k arrays of training and test data splices of dataset
    
                Example: [[[0.23, 0.34, 0.33, 0.12, 0.45, 0.68], [0.13, 0.35, 0.01, 0.72, 0.25, 0.08], ....] , .... ,
                        [[0.12, 0.45, 0.23, 0.64, 0.67, 0.98], [0.20, 0.50, 0.23, 0.12, 0.32, 0.88], ....]]
        """
        train_test_split = []
        size = len(dataset)
        num_of_elements = int(size / k)
        for i in range(k):
            new_sample = random.sample(dataset, num_of_elements)
            train_test_split.append(new_sample)
            for row in new_sample:
                dataset.remove(row)
        if len(dataset) != 0:
            for rows in range(len(dataset)):
                train_test_split[rows].append(dataset[rows])
            dataset.clear()
        return train_test_split
