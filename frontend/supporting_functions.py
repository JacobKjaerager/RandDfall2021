from pathlib import Path
import pandas as pd
import os

class DataObject:
    def __init__(self):
        print("DataObject made")

    def get_data(self):
        model_folder = "../saved_model/"
        for i in os.listdir(Path(model_folder)):
            current_model_path = model_folder + i + "/"
            hyper_parameters = pd.read_csv(current_model_path + "hyperparameters.csv")
            if hyper_parameters["model"] == "deeplob":
                execute_deeblob_logic(hyper_parameters)
            if hyper_parameters["model"] == "tabl":
                execute_tabl_logic(hyper_parameters)
            else:
                execute:layered_logic(hyper_parameters)
            current_model_path = model_folder + i
            print("faweeawf")
