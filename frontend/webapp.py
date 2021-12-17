import dash
from pathlib import Path
#from frontend.callbacks import init_callback
#from frontend.layout import layout
from supporting_functions import DataObject
#import socket
import plotly.graph_objs as go
import os
import pandas as pd

def save_to_graph(Do, column):
    data = []
    Do.index = Do.index + 1
    for i in Do.columns:
        data.append(
            go.Scatter(
                x=Do.index,
                y=Do[i],
                name="{}".format(i)
            )
        )
    fig = go.Figure(
        data=data
    )
    fig.update_layout(title='{} over Epochs'.format(column),
                      xaxis_title='Epochs',
                      yaxis_title='{}'.format(column))
    fig.write_html("{}epoch_dev_{}.html".format("../global_saves/", column))



def start_webserver(local_host: bool=True):
    current_column = "accuracy"
    model_folder = "../saved_model/"
    preds=[]
    trace = "accuracy"
    for i in os.listdir(Path(model_folder)):
        # hyper_parameters = pd.read_csv(current_model_path + "hyperparameters.csv").drop(columns=["Unnamed: 0"])
        # collector.append(pd.read_csv(current_model_path + "history.csv").drop(columns=["Unnamed: 0"]).rename(columns={column: i})[i])
        current_df = pd.read_csv(model_folder + i + "/history.csv")
        preds.append(current_df.rename(columns={"accuracy": i})[i])
        #preds.append(current_df["real"])
    Do = DataObject().run_ensemble_data()
    print("ewaf")
    save_to_graph(Do, current_column)
    app = dash.Dash(
        __name__,
        assets_folder="{}\styling".format(Path(__file__).parent)
    )
    app = layout(app)
    #init_callback(app, data = Do)
    if local_host:
        app.run_server(debug=True)
    else:
        app.run_server(host=socket.gethostbyname(socket.gethostname()), debug=True, port=4000)


if __name__ == '__main__':
    print("faew")
    #start_webserver()
    Do = DataObject().run_ensemble_data()

#     import plotly.graph_objects as go
#
#     models = ["11-12-21-02-47-40_fitted_on_100_EPOCHS",
#                "11-12-21-05-49-30_fitted_on_100_EPOCHS",
#                "11-12-21-09-23-59_fitted_on_100_EPOCHS"]
#     fig = go.Figure(data=[
#     go.Bar(name='Percentage predicted 3 when true is 1',
#            x=models,
#            y=[(6506 / (6506 + 5327 + 37132)),
#               (6637 / (6637 + 5423 + 36691)),
#               (6739 / (6739 + 5732 + 37076)),
#            ]),
#     go.Bar(name='Percentage predicted 2 when true is 1',
#            x=models,
#            y=[(5327 / (6506 + 5327 + 37132)),
#               (5423 / (6637 + 5423 + 36691)),
#               (5732 / (6739 + 5732 + 37076)),
#            ]),
#     go.Bar(name='Percentage predicted 1 when true is 3',
#            x=models,
#            y=[(7617 / (7617 + 5215 + 33943)),
#               (7788 / (7788 + 5192 + 33681)),
#               (7523 / (7523 + 5799 + 33578)),
#            ]),
#     go.Bar(name='Percentage predicted 2 when true is 3',
#            x=models,
#            y=[(5215 / (7617 + 5215 + 33943)),
#               (5192 / (7788 + 5192 + 33681)),
#               (5799 / (7523 + 5799 + 33578)),
#            ]),
#     ])
#     # Change the bar mode
#     fig.update_layout(barmode='group',
#                       xaxis_title = 'Model',
#                       yaxis_title = 'Bad predictions(%)')
#     fig.write_html("{}bad_predictions_deeplob_models.html".format("../global_saves/"))
#
#     ###EPOCH development
#     preds = []
#     model_folder = "../saved_model/"
#     trace = "val_accuracy" #val_accuracy
#     for i in os.listdir(Path(model_folder)):
#         current_df = pd.read_csv(model_folder + i + "/history.csv")
#         preds.append(current_df.rename(columns={trace: i})[i])
#     df = pd.concat(preds, axis=1)[["11-12-21-02-47-40_fitted_on_100_EPOCHS",
#                                    "11-12-21-05-49-30_fitted_on_100_EPOCHS",
#                                    "11-12-21-09-23-59_fitted_on_100_EPOCHS"]]
#     q = df.iloc[0:df.shape[0]-100] #100 EPOCHS
#     data = []
#     for i in q.columns:
#         data.append(
#             go.Scatter(
#                 x=q[i].index,
#                 y=q[i].values,
#                 name=i,
#             )
#     )
#     fig = go.Figure(
#         data=data
#     )
#     # Change the bar mode
#     fig.update_layout(xaxis_title='Epochs',
#                       yaxis_title='{}'.format(trace))
#     fig.write_html("{}{}Epoch_dev_DeepLOB.html".format("../global_saves/", trace))
#
# ###

