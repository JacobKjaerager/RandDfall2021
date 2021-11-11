import pandas as pd
import scipy.io as spio
import numpy as np
from keras.utils import np_utils
from math import floor, ceil
from hyperopt_models import get_hyper_opt_conf
from mapper import compile_and_train_models, get_train_set
from control_file import Control_dict
import Models
import data_management
import ast

if __name__ == '__main__':
<<<<<<< HEAD
    [X_train, y_train, X_cv, y_cv] = get_train_set(Control_dict)
    hyper_opt_models = get_hyper_opt_conf(train_shape=X_train[0].shape)
    compile_and_train_models(hyperopt_confs=hyper_opt_models,
                             X_train=X_train,
                             y_train=y_train,
                             Control_dict=Control_dict)
=======
    train_set = spio.loadmat("../deepFold_train", squeeze_me=True)
    test_set = spio.loadmat("../deepFold_test", squeeze_me=True)
    X_train = train_set["x_train"][:ceil(Control_dict["cross_validation_size"]* len(train_set["x_train"]))]
    y_train = pd.DataFrame(train_set["y_train"])[[0]].iloc[:ceil(Control_dict["cross_validation_size"] * len(train_set["y_train"]))]
    X_cv = train_set["x_train"][ceil(Control_dict["cross_validation_size"] * len(train_set["x_train"])):]
    y_cv = pd.DataFrame(train_set["y_train"])[[0]].iloc[
           ceil(Control_dict["cross_validation_size"] * len(train_set["y_train"])):]
    X_test = test_set["x_test"]
    y_test = pd.DataFrame(test_set["y_test"])[[0]]
    # To enable CNN and softmax
    y_test = np_utils.to_categorical(y_test - 1, 3)
    y_train = np_utils.to_categorical(y_train - 1, 3)
    X_train = X_train.reshape(len(X_train), 10, 40, 1)
    X_test = X_test.reshape(len(X_test), 10, 40, 1)
    # 
    hyper_opt_models = get_hyper_opt_conf(train_shape=X_train[0].shape)
    models = compile_and_train_models(hyperopt_confs=hyper_opt_models,
                                      X_train=X_train,
                                      y_train=y_train,
                                      X_test=X_test,
                                      y_test=y_test,
                                      Control_dict=Control_dict)

>>>>>>> 69c092ea9fce827ee6ded204d7525d7e60db6005
    print("eawf")

