import numpy as np
import seaborn as sn
import tensorflow as tf
import pandas as pd
import json
import scipy.io as spio
from pathlib import Path
from math import floor, ceil
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras import backend as K
from sklearn.metrics import classification_report, confusion_matrix
import plotly
import matplotlib.pyplot as plt
from datetime import datetime as dt
from hyperopt_models import get_hyper_opt_conf
from mapper import compile_and_train_models
from control_file import Control_dict
import ast

if __name__ == '__main__':

    train_set = spio.loadmat("../deepFold_train", squeeze_me=True)
    test_set = spio.loadmat("../deepFold_test", squeeze_me=True)
    X_train = train_set["x_train"][:ceil(Control_dict["cross_validation_size"]* len(train_set["x_train"]))]
    y_train = pd.DataFrame(train_set["y_train"])[[0]].iloc[:ceil(Control_dict["cross_validation_size"] * len(train_set["y_train"]))]

    X_cv = train_set["x_train"][ceil(Control_dict["cross_validation_size"] * len(train_set["x_train"])):]
    y_cv = pd.DataFrame(train_set["y_train"])[[0]].iloc[
           ceil(Control_dict["cross_validation_size"] * len(train_set["y_train"])):]

    X_test = test_set["x_test"]
    y_test = pd.DataFrame(test_set["y_test"])[[0]]
    hyper_opt_models = get_hyper_opt_conf(train_shape=X_train[0].shape)
    models = compile_and_train_models(hyperopt_confs=hyper_opt_models,
                                      X_train=X_train,
                                      y_train=y_train,
                                      Control_dict=Control_dict)

    print("eawf")

    #
    # print("fae")

