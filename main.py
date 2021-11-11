import pandas as pd
import scipy.io as spio
from math import floor, ceil
from hyperopt_models import get_hyper_opt_conf
from mapper import compile_and_train_models, get_train_set
from control_file import Control_dict
import ast

if __name__ == '__main__':
    [X_train, y_train, X_cv, y_cv] = get_train_set(Control_dict)
    hyper_opt_models = get_hyper_opt_conf(train_shape=X_train[0].shape)
    compile_and_train_models(hyperopt_confs=hyper_opt_models,
                             X_train=X_train,
                             y_train=y_train,
                             Control_dict=Control_dict)
    print("eawf")

