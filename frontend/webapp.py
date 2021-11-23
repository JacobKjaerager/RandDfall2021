import dash
from pathlib import Path
from frontend.callbacks import init_callback
from frontend.layout import layout
from frontend.supporting_functions import DataObject
import socket

def start_webserver(local_host: bool):
    Do = DataObject().get_data()
    app = dash.Dash(
        __name__,
        assets_folder="{}\styling".format(Path(__file__).parent)
    )
    app = layout(app)
    init_callback(app)
    if local_host:
        app.run_server(debug=True)
    else:
        app.run_server(host=socket.gethostbyname(socket.gethostname()), debug=True, port=4000)



if __name__ == '__main__':
    start_webserver(local_host=True)
    print("weaf")

