import pandas as pd
import scipy.io as spio
import numpy as np
from keras.utils import np_utils
from math import floor, ceil
from hyperopt_models import get_hyper_opt_conf
from mapper import compile_and_train_models
from control_file import Control_dict
from data_management import *
from sklearn.decomposition import PCA
import Models
import data_management
import ast
import tensorflow as tf
#from frontend.webapp import start_webserver
from mapper import *
import time
import plotly.graph_objs as go

def manually_add_graphics(model_path):
    save_folder = model_path #"./saved_model//25-11-21 10-49-40_fitted_on_80_EPOCHS"
    df_pred_and_real = pd.read_csv("{}/predictions.csv".format(save_folder))
    hist = pd.read_csv("{}/history.csv".format(save_folder))
    save_html_based_plots(df_pred_and_real=df_pred_and_real,
                          hist=hist,
                          save_folder=save_folder)
    
def static_train_test_data():
    [train_data, test_data] = read_data(base_path="../BenchmarkDatasets/",
                                        auction=False,
                                        normalization="Zscore",
                                        fold=7,
                                        combine_test=True)
    return [train_data, test_data]


if __name__ == '__main__':
    [train_data, test_data] = static_train_test_data()

    print("wae")
    # start_webserver()
    # [train_data, test_data] = read_data(base_path="../BenchmarkDatasets/",
    #                                     auction=False,
    #                                     normalization="Zscore",
    #                                     fold=7,
    #                                     combine_test=True)

    start = time.time()
    train_data = pd.read_csv("../pickle_files/training_data.csv")
    test_data = pd.read_csv("../pickle_files/test_data.csv")
    end1 = time.time()
    hyper_opt_models = get_hyper_opt_conf()
    # X_train = X_train.reshape(len(X_train), 10, 40, 1)
    # X_test = X_test.reshape(len(X_test), 10, 40, 1)
    compile_and_train_models(hyperopt_confs=hyper_opt_models,
                             train_data=train_data,
                             test_data=test_data,
                             Control_dict=Control_dict)


    # train_set = spio.loadmat("../deepFold_train", squeeze_me=True)
    # test_set = spio.loadmat("../deepFold_test", squeeze_me=True)
    # X_train = train_set["x_train"][:ceil(Control_dict["cross_validation_size"]* len(train_set["x_train"]))]
    # y_train = pd.DataFrame(train_set["y_train"])[[0]].iloc[:ceil(Control_dict["cross_validation_size"] * len(train_set["y_train"]))]
    # X_cv = train_set["x_train"][ceil(Control_dict["cross_validation_size"] * len(train_set["x_train"])):]
    # y_cv = pd.DataFrame(train_set["y_train"])[[0]].iloc[
    #        ceil(Control_dict["cross_validation_size"] * len(train_set["y_train"])):]
    # X_test = test_set["x_test"]
    # y_test = pd.DataFrame(test_set["y_test"])[[0]]
    # # To enable CNN and softmax

    # #
    # hyper_opt_models = get_hyper_opt_conf(train_shape=X_train[0].shape)
    # compile_and_train_models(hyperopt_confs=hyper_opt_models,
    #                          X_train=X_train,
    #                          y_train=y_train,
    #                          X_test=X_test,
    #                          y_test=y_test,
    #                          Control_dict=Control_dict)
    # print("eawf")
    #
