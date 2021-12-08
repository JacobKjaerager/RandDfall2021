# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 20:06:24 2020

@author: Jacob Kjaerager
"""

import dash
import pandas as pd
import plotly.graph_objs as go
from dash import dcc, html, dash_table


def layout(app, data):
    app.layout = \
        html.Div(
            className="Full_view",
            children=[
                # 500 API calls pr month = update interval on 1.488 hour(rounded to 2 hours) (math: (31*24)/api_calls = 1.488
                html.Div(
                    className="right_side_of_view",
                    children=[
                        html.Div(
                            className="card-datatable-ttb",
                            children=[
                                html.Div(
                                    className="datatable",
                                    children=[
                                        dash_table.DataTable(
                                            id='datatable',
                                            style_as_list_view=True,
                                            fixed_rows={
                                                'headers': True
                                            },
                                            style_table={
                                                'height': 400,
                                                'overflowX': 'auto',
                                            },
                                            style_cell={
                                                'textAlign': 'center',
                                                'overflow': 'hidden',
                                                'minWidth': '80px',
                                                'width': '80px',
                                                'maxWidth': '180px',
                                            }
                                        )
                                    ],
                                ),

                            ]
                        ),
                        html.Div(
                            className="graph-card",
                            children=[
                                html.Div(
                                    className="historical-graph-odds-development-hometeam",
                                    children=[
                                        dcc.Graph(
                                            id="full_round_bar_chart"
                                        )
                                    ]
                                )
                            ]
                        )
                    ],
                ),
            ]
        )
    return app