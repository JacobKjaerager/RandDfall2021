from keras.models import Sequential
from keras.layers import Dense, LSTM
from datetime import datetime as dt
import pandas as pd
import json
import scipy.io as spio
from math import floor, ceil


def compile_and_train_models(hyperopt_confs, X_train, y_train, Control_dict):

<<<<<<< HEAD
=======
def compile_and_train_models(hyperopt_confs, X_train, y_train, X_test, y_test, Control_dict) -> list:
    return_list = []
>>>>>>> 69c092ea9fce827ee6ded204d7525d7e60db6005
    for model_params in hyperopt_confs: #Individual model
        if model_params["enabled_for_run"]:
            print("New model fitting started at: {}".format(dt.now().strftime("%H:%M:%S %d-%m-%y")))
            model = model_params["model"]
            for layer in model_params["layers"]: #Individual layer
                model.add(
                    layer=layer["layer_type"](**layer["layer_arguments"])
                )
            model.compile(optimizer=model_params["optimizer"],
<<<<<<< HEAD
                          loss=model_params["loss_function"])
            model.fit(x=X_train,
                      y=y_train,
                      epochs=model_params["EPOCHS"],
                      verbose=0,
                      shuffle=False)
=======
                          loss=model_params["loss_function"],
                          metrics=['accuracy'])
            history = model.fit(x=X_train,
                                y=y_train,
                                epochs=model_params["EPOCHS"],
                                verbose=0,
                                validation_data=(X_test, y_test),
                                shuffle=False)
            return_list.append(model)
>>>>>>> 69c092ea9fce827ee6ded204d7525d7e60db6005
            save_folder = '{}\\{}_fitted_on_{}_EPOCHS'.format(Control_dict["models_save_folder"],
                                                              dt.now().strftime("%d-%m-%y %H-%M-%S"),
                                                              model_params["EPOCHS"])
            model.save(save_folder)
            pd.DataFrame.from_dict(history.history).to_csv(path_or_buf="{}\\history.csv".format(save_folder))
            pd.DataFrame.from_dict(model_params).to_csv(path_or_buf="{}\\hyperparameters.csv".format(save_folder))
            [X_test, y_test] = get_test_set()
            y_pred = model.predict(x=X_test,
                                   verbose=0)
            df = pd.DataFrame(columns=["predicted", "real"])
            df["predicted"] = pd.Series(y_pred.squeeze())
            df["real"] = y_test[0]
            df.to_csv(path_or_buf="{}\\predictions.csv".format(save_folder))


def get_train_set(Control_dict) -> list:
    train_set = spio.loadmat("../deepFold_train", squeeze_me=True)
    X_train = train_set["x_train"][:ceil(Control_dict["cross_validation_size"]* len(train_set["x_train"]))]
    y_train = pd.DataFrame(train_set["y_train"])[[0]].iloc[:ceil(Control_dict["cross_validation_size"] * len(train_set["y_train"]))]
    X_cv = train_set["x_train"][ceil(Control_dict["cross_validation_size"] * len(train_set["x_train"])):]
    y_cv = pd.DataFrame(train_set["y_train"])[[0]].iloc[
           ceil(Control_dict["cross_validation_size"] * len(train_set["y_train"])):]
    return [X_train, y_train, X_cv, y_cv]

def get_test_set() -> list:
    test_set = spio.loadmat("../deepFold_test", squeeze_me=True)
    X_test = test_set["x_test"]
    y_test = pd.DataFrame(test_set["y_test"])[[0]]

    return [X_test, y_test]

