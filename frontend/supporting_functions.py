from pathlib import Path
import pandas as pd
import os
import numpy as np
class DataObject:
    def __init__(self):
        print("DataObject made")

    def get_data(self, column):
        collector = []
        preds = []
        model_folder = "../saved_model/"
        pred_length = 10
        for i in os.listdir(Path(model_folder)):
            #hyper_parameters = pd.read_csv(current_model_path + "hyperparameters.csv").drop(columns=["Unnamed: 0"])
            #collector.append(pd.read_csv(current_model_path + "history.csv").drop(columns=["Unnamed: 0"]).rename(columns={column: i})[i])
            current_df = pd.read_csv(model_folder + i + "/predictions.csv")
            preds.append(current_df.rename(columns={"predicted": i})[i])
            preds.append(current_df["real"])
            # if any(hyper_parameters["model"].values) == "deeplob":
            #     execute_deeblob_logic(hyper_parameters)
            # # if hyper_parameters["model"].values == "tabl":
            # #     execute_tabl_logic(hyper_parameters)
            #  else:
            #      execute_layered_logic(hyper_parameters)
            #
            # current_model_path = model_folder + i
            # print("faweeawf")
        df = pd.concat(preds, axis=1).loc[:, ~pd.concat(preds, axis=1).columns.duplicated()]
        df.loc[:,"emsemble_pred"] = np.NaN
        for col in df.columns[df.columns != "real"]:
            df.loc[:, col + "_weight"] = np.NaN
            df.loc[0:pred_length, col + "_weight"] = 1

        for index, row in df.iterrows():
            new_row = row[~row.index.str.contains("weight") & ~row.index.str.contains("real")]
            correct_preds = new_row[new_row == row["real"]]
            wrong_preds = new_row[new_row != row["real"]]

            for i in new_row == row["real"]:
                if row[i] == row["real"]:
                    print("eafw")
        return pd.concat(preds, axis=1)


def execute_tabl_logic(hyper_parameters):
    trimmed = hyper_parameters.drop(columns=["enabled_for_run",
                                             "input_shape_sample",
                                             "input_shape_features",
                                             "data_dimensions"])
    return trimmed


def execute_deeblob_logic(hyper_parameters):
    print("aefwawe")

def execute_layered_logic(hyper_parameters):
    print("awfe")