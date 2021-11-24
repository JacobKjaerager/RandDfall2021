from keras.models import Sequential
from keras.layers import Dense, LSTM
from datetime import datetime as dt
import pandas as pd
import json
from keras.utils import np_utils
import tensorflow as tf
import scipy.io as spio
from math import floor, ceil
from data_management import get_model_data
import pandas as pd


def compile_and_train_models(hyperopt_confs: dict,
                             train_data: pd.DataFrame,
                             test_data: pd.DataFrame,
                             Control_dict: dict) -> None:
    for model_params in hyperopt_confs: #Individual model
        if model_params["enabled_for_run"]:
            reshaper_train = []
            reshaper_test = []
            print("New model fitting started at: {}".format(dt.now().strftime("%H:%M:%S %d-%m-%y")))

            [X_train, y_train] = get_model_data(data=train_data,
                                                sample_size=model_params["input_shape_sample"],
                                                feature_num=model_params["input_shape_features"],
                                                target_num=model_params["target_numbers"])
            [X_test, y_test] = get_model_data(data=test_data,
                                              sample_size=model_params["input_shape_sample"],
                                              feature_num=model_params["input_shape_features"],
                                              target_num=model_params["target_numbers"])
            for i in range(model_params["data_dimensions"]):
                reshaper_train.append(X_train.shape[i])
                reshaper_test.append(X_test.shape[i])
            X_train = X_train.reshape(reshaper_train)
            X_test = X_test.reshape(reshaper_test)

            print("New model fitting started at: {}".format(dt.now().strftime("%H:%M:%S %d-%m-%y")))
            model = model_params["model"]
            for layer in model_params["layers"]: #Individual layer'
                if list(layer.keys()).__contains__("layer_arguments"):
                    model.add(
                        layer=layer["layer_type"](**layer["layer_arguments"])
                    )
                else:
                    model.add(
                        layer=layer["layer_type"]()
                    )
            model.compile(optimizer=model_params["optimizer"],
                          loss=model_params["loss_function"],
                          metrics=['accuracy'])
            history = model.fit(x=X_train,
                                y=y_train,
                                epochs=model_params["EPOCHS"],
                                verbose=1,
                                shuffle=False,
                                validation_data=(X_test, y_test))

            save_folder = '{}\\{}_fitted_on_{}_EPOCHS'.format(Control_dict["models_save_folder"],
                                                              dt.now().strftime("%d-%m-%y %H-%M-%S"),
                                                              model_params["EPOCHS"])
            model.save(save_folder)
            pd.DataFrame.from_dict(history.history).to_csv(path_or_buf="{}\\history.csv".format(save_folder))
            #pd.DataFrame.from_dict(model_params).to_csv(path_or_buf="{}\\hyperparameters.csv".format(save_folder))
            y_pred = model.predict(x=X_test, verbose=0)
            df = pd.DataFrame(columns=["predicted", "real"])
            df["predicted"] = pd.Series(y_pred.argmax(axis=1))
            df["real"] = pd.Series(y_test.argmax(axis=1))
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

