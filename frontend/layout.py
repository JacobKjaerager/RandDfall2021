# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 20:06:24 2020

@author: Jacob Kjaerager
"""

import dash
import dash_table
import pandas as pd
import plotly.graph_objs as go
import dash_core_components as dcc
import dash_html_components as html

def layout(app):
    app.layout = \
    html.Div(
        className="Full_view",
        children=[
            html.Div(
                className="left_half_of_view",
                children=[
                    html.Div(
                        className="card-filter_main_component",
                        children=[
                            html.Div(
                                className="all-model-selector",
                                children=[
                                    dcc.Dropdown(
                                        id='all_model_selector',
                                        clearable=False,
                                        multi=True
                                    )    
                                ],
                            ),
                        ]
                    ),
                    html.Div(
                        className="card-game-tweaking",
                        children=[
                            html.Div(
                                className="game-tweaking",
                                children=[
                                    dcc.RangeSlider(
                                        count=1,
                                        min=-5,
                                        max=10,
                                        step=0.5,
                                        value=[-3, 7]
                                    )
                                ],
                            ),
                        ]
                    ),
                ]
            ),
            html.Div(
                className="right_side_of_view",
                children=[
                    html.Div(
                        className="graph-card",
                        children=[
                            html.Div(
                                className="game-evaluation-graph",
                                children=[
                                    dcc.Graph(
                                        id="full_round_bar_chart"
                                    )
                                ]
                            )
                        ]
                    ),
                    html.Div(
                        className="card-datatable-hyperparameters",
                        children=[
                            html.Div(
                                className="datatable",
                                children=[
                                    dash_table.DataTable(
                                        id='datatable',
                                        style_as_list_view=True,
                                        # fixed_rows={
                                        #     'headers': True
                                        # },
                                        # style_table={
                                        #     'height': 400,
                                        #     'overflowX': 'auto',
                                        # },
                                        # style_cell={
                                        #     'textAlign': 'center',
                                        #     'overflow': 'hidden',
                                        #     'minWidth': '80px',
                                        #     'width': '80px',
                                        #     'maxWidth': '180px',
                                        # }
                                    )
                                ],
                            ),
                        ]
                    ),
                ],
            ),
        ]
    )
    return app