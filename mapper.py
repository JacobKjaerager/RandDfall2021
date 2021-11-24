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
import plotly.graph_objs as go
from map_to_object import map_2_obj

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
                                                feature_num=model_params["input_shape_features"])
            [X_test, y_test] = get_model_data(data=test_data,
                                              sample_size=model_params["input_shape_sample"],
                                              feature_num=model_params["input_shape_features"])
            for i in range(model_params["data_dimensions"]):
                reshaper_train.append(X_train.shape[i])
                reshaper_test.append(X_test.shape[i])
            X_train = X_train.reshape(reshaper_train)
            X_test = X_test.reshape(reshaper_test)
            model = map_2_obj(model_params["model"])
            for layer in model_params["layers"]: #Individual layer'
                if list(layer.keys()).__contains__("layer_arguments"):
                    model.add(
                        layer=map_2_obj(layer["layer_type"])(**layer["layer_arguments"])
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
            hist = pd.DataFrame.from_dict(history.history)
            hist.to_csv(path_or_buf="{}\\history.csv".format(save_folder))
            y_pred = model.predict(x=X_test, verbose=0)

            df_output = pd.DataFrame(y_pred)
            df_output = df_output.rename(columns={0:"1", 1:"2", 2:"3"})
            df_output.to_csv(path_or_buf="{}\\softmax.csv".format(save_folder))

            df_pred_and_real = pd.DataFrame(columns=["predicted", "real"])
            df_pred_and_real["predicted"] = pd.Series(y_pred.argmax(axis=1))
            df_pred_and_real["real"] = pd.Series(y_test.argmax(axis=1))
            df_pred_and_real.to_csv(path_or_buf="{}\\predictions.csv".format(save_folder))
            pd.DataFrame.from_dict(model_params).to_csv(path_or_buf="{}\\hyperparameters.csv".format(save_folder))
            save_html_based_plots(df_pred_and_real=df,
                                  hist=hist,
                                  save_folder=save_folder)


    df["predicted"] = pd.Series(y_pred.argmax(axis=1))
    df["real"] = pd.Series(y_test.argmax(axis=1))
    return df


def save_html_based_plots(df_pred_and_real, hist, save_folder):
    make_and_save_epoch_dev_plot(hist=hist, save_folder=save_folder)
    make_and_save_binned_pred_and_true(df_pred_and_real=df_pred_and_real, save_folder=save_folder)


def make_and_save_epoch_dev_plot(hist, save_folder):
    data = []
    hist.index = hist.index + 1
    data.append(
        go.Scatter(
            x=hist.index,
            y=hist.accuracy,
            name="Training Accuracy"
        )
    )
    data.append(
        go.Scatter(
            x=hist.index,
            y=hist.val_accuracy,
            name="Validation Accuracy"
        )
    )
    fig = go.Figure(
        data=data
    )
    fig.update_layout(title='Accuracy over Epochs',
                      xaxis_title='Epochs',
                      yaxis_title='Accuracy')
    fig.write_html("{}\\epoch_dev.html".format(save_folder))


def get_groups(df: pd.DataFrame) -> list:
    real_groups = df.groupby("real")
    t1 = real_groups.get_group(0).groupby("predicted").count()
    t2 = real_groups.get_group(1).groupby("predicted").count()
    t3 = real_groups.get_group(2).groupby("predicted").count()
    first_data_bar = [t1.real.loc[0], t2.real.loc[0], t3.real.loc[0]]
    second_data_bar = [t1.real.loc[1], t2.real.loc[1], t3.real.loc[1]]
    third_data_bar = [t1.real.loc[2], t2.real.loc[2], t3.real.loc[2]]
    return [first_data_bar, second_data_bar, third_data_bar]


def make_and_save_binned_pred_and_true(df_pred_and_real, save_folder):
    [correct_label0, correct_label1, correct_label2] = get_groups(df_pred_and_real)
    x = [1,2,3]
    data = [
        go.Bar(
            x=x,
            y=correct_label0,
            name="Classified label 0"
        ),
        go.Bar(
            x=x,
            y=correct_label1,
            name="Classified label 1"
        ),
        go.Bar(
            x=x,
            y=correct_label2,
            name="Classified label 2"
        ),
    ]
    fig = go.Figure(
        data=data
    )
    fig.update_layout(
        barmode='group',
        title='Amount of correctly classified labels and what they are mapped to',
        xaxis_title='y_real',
        yaxis_title='No of samples in y_pred')
    fig.write_html("{}\\predict_vs_real.html".format(save_folder))


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

