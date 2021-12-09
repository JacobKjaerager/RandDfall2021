import dash
from pathlib import Path
from frontend.callbacks import init_callback
from frontend.layout import layout
from frontend.supporting_functions import DataObject
import socket
import plotly.graph_objs as go


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
    start_webserver(local_host=True)

    print("weaf")

