from keras.models import Sequential
from keras.layers import Dense, LSTM
from datetime import datetime as dt
import pandas as pd
import json

def compile_and_train_models(hyperopt_confs, X_train, y_train, X_test, y_test, Control_dict) -> list:
    return_list = []
    for model_params in hyperopt_confs: #Individual model
        print("New model fitting started at: {}".format(dt.now().strftime("%H:%M:%S %d-%m-%y")))
        if model_params["enabled_for_run"]:
            model = model_params["model"]
            for layer in model_params["layers"]: #Individual layer
                model.add(
                    layer=layer["layer_type"](**layer["layer_arguments"])
                )
            model.compile(optimizer=model_params["optimizer"],
                          loss=model_params["loss_function"],
                          metrics=['accuracy'])
            history = model.fit(x=X_train,
                                y=y_train,
                                epochs=model_params["EPOCHS"],
                                verbose=0,
                                validation_data=(X_test, y_test),
                                shuffle=False)
            return_list.append(model)
            save_folder = '{}\\{}_fitted_on_{}_EPOCHS'.format(Control_dict["models_save_folder"],
                                                              dt.now().strftime("%d-%m-%y %H-%M-%S"),
                                                              model_params["EPOCHS"])
            model.save(save_folder)
            pd.DataFrame.from_dict(history.history).to_csv(path_or_buf="{}\\history.csv".format(save_folder))
            pd.DataFrame.from_dict(model_params).to_csv(path_or_buf="{}\\hyperparameters.csv".format(save_folder))


    return return_list
