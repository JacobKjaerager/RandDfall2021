from pathlib import Path
import pandas as pd
import os

class DataObject:
    def __init__(self):
        print("DataObject made")

    def get_data(self):
        model_folder = "../saved_model/"
        for i in os.listdir(Path(model_folder)):
            current_model_path = model_folder + i

            print("faweeawf")
