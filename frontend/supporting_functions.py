from pathlib import Path
import pandas as pd
import os
import numpy as np

from sklearn.metrics import accuracy_score
class DataObject:
    def __init__(self):
        print("DataObject made")

    def get_data(self, column):
        collector = []
        preds = []
        model_folder = "../saved_model/"
        pred_length = 10
        deduction_factor = 0.02
        for i in os.listdir(Path(model_folder)):
            #hyper_parameters = pd.read_csv(current_model_path + "hyperparameters.csv").drop(columns=["Unnamed: 0"])
            #collector.append(pd.read_csv(current_model_path + "history.csv").drop(columns=["Unnamed: 0"]).rename(columns={column: i})[i])
            current_df = pd.read_csv(model_folder + i + "/predictions.csv")
            preds.append(current_df.rename(columns={"predicted": i})[i])
            preds.append(current_df["real"])

        df = pd.concat(preds, axis=1).loc[:, ~pd.concat(preds, axis=1).columns.duplicated()]

        for col in df.columns[df.columns != "real"]:
            df.loc[:, col + "_weight"] = np.NaN
            df.loc[0:pred_length, col + "_weight"] = 1
        df.loc[:, "ensemble_pred"] = np.NaN
        for index, row in df.iloc[df.shape[0] - pred_length].iterrows():
            if index%1000 == 0:
                print(index)

            new_row = row[~row.index.str.contains("weight") & ~row.index.str.contains("real") & ~row.index.str.contains("ensemble_pred")]
            correct_preds = new_row[new_row == row["real"]]
            wrong_preds = new_row[new_row != row["real"]]
            if correct_preds.shape[0] > 0:
                b = df.loc[index + pred_length-1, correct_preds.index + "_weight"] + wrong_preds.shape[0]*deduction_factor / correct_preds.shape[0]
                b[b > 2] = 2
                df.loc[index + pred_length, correct_preds.index + "_weight"] = b
            if wrong_preds.shape[0] > 0:
                a = df.loc[index + pred_length - 1, wrong_preds.index + "_weight"] - deduction_factor
                a[a < 0] = 0
                df.loc[index + pred_length, wrong_preds.index + "_weight"] = a

            votes_for_zero = df.iloc[index][new_row[new_row == 0].index + "_weight"].sum()
            votes_for_one = df.iloc[index][new_row[new_row == 1].index + "_weight"].sum()
            votes_for_two = df.iloc[index][new_row[new_row == 2].index + "_weight"].sum()
            if(votes_for_zero > votes_for_one) & (votes_for_zero >= votes_for_two):
                df.loc[index, "ensemble_pred"] = 0
            elif(votes_for_one >= votes_for_zero) & (votes_for_one >= votes_for_two):
                df.loc[index, "ensemble_pred"] = 1
            elif(votes_for_two > votes_for_zero) & (votes_for_two > votes_for_one):
                df.loc[index, "ensemble_pred"] = 2


        return df

    def run_ensemble_data(self):
        collector = []
        preds = []
        model_folder = "../saved_model/"
        pred_length = 10
        deduction_factor = 0.02
        for i in os.listdir(Path(model_folder)):
            #hyper_parameters = pd.read_csv(current_model_path + "hyperparameters.csv").drop(columns=["Unnamed: 0"])
            #collector.append(pd.read_csv(current_model_path + "history.csv").drop(columns=["Unnamed: 0"]).rename(columns={column: i})[i])
            current_df = pd.read_csv(model_folder + i + "/predictions.csv")
            preds.append(current_df.rename(columns={"predicted": i})[i])
            preds.append(current_df["real"])

        df = pd.concat(preds, axis=1).loc[:, ~pd.concat(preds, axis=1).columns.duplicated()]

        for col in df.columns[df.columns != "real"]:
            df.loc[:, col + "_weight"] = np.NaN
            df.loc[0:pred_length, col + "_weight"] = 1
        df.loc[:, "ensemble_pred"] = np.NaN
        for index, row in df.iloc[0:df.shape[0] - pred_length].iterrows():
            if index%100 == 0:
                print(index)

            new_row = row[~row.index.str.contains("weight") & ~row.index.str.contains("real") & ~row.index.str.contains("ensemble_pred")]
            correct_preds = new_row[new_row == row["real"]]
            wrong_preds = new_row[new_row != row["real"]]
            if correct_preds.shape[0] > 0:
                b = df.loc[index + pred_length-1, correct_preds.index + "_weight"] + wrong_preds.shape[0]*deduction_factor / correct_preds.shape[0]
                b[b > 2] = 2
                df.loc[index + pred_length, correct_preds.index + "_weight"] = b
            if wrong_preds.shape[0] > 0:
                a = df.loc[index + pred_length - 1, wrong_preds.index + "_weight"] - deduction_factor
                a[a < 0] = 0
                df.loc[index + pred_length, wrong_preds.index + "_weight"] = a

            votes_for_zero = df.iloc[index][new_row[new_row == 0].index + "_weight"].sum()
            votes_for_one = df.iloc[index][new_row[new_row == 1].index + "_weight"].sum()
            votes_for_two = df.iloc[index][new_row[new_row == 2].index + "_weight"].sum()
            if(votes_for_zero > votes_for_one) & (votes_for_zero >= votes_for_two):
                df.loc[index, "ensemble_pred"] = 0
            elif(votes_for_one >= votes_for_zero) & (votes_for_one >= votes_for_two):
                df.loc[index, "ensemble_pred"] = 1
            elif(votes_for_two > votes_for_zero) & (votes_for_two > votes_for_one):
                df.loc[index, "ensemble_pred"] = 2


        return df

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