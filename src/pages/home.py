# -*- coding: utf-8 -*-
"""
Created on Sun May  5 11:57:57 2024

@author: Gebruiker
"""
import pandas as pd
import numpy as np
import datetime

import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import plots_generator
import metrics_generator
from config import colors_config, card_config

# =============================================================================
# This is the first page the user will see.
# It contains an overview of the results of the running year where nested dbc Rows and Cols are used to control the placement of the components.
# Starting on the left we have a block of dbc Cards which contain interactive information based on the time period the user is selecting.
# In the middle there is a large interactive graph where on the top the user can choose the time period either through a date picker or a button.
# On the top right there is a column with a brief explanation of the different strategies one can choose and below it is a target graph.
# The bottom row than has graphs per month and a multiplot with different statistics.
# 
# =============================================================================

# Required so the dash multipager will take this page on. As it is the first page we have to add "path='/'"
dash.register_page(__name__, path='/')

# downloading data containing all individual stock trades for the running year
fname = 'DC_2024trades.csv'
df = pd.read_csv(f'../{fname}', parse_dates = ['date'], index_col = 'date')

# some definitions for readability
legend_labels = {'pnl_u': 'All-in', 'pnl_c': 'Conditional', 'pnl_cl': 'Leveraged'}

background_img = 'linear-gradient(to left, rgba(39,83,81,0.75), rgba(0,0,0,1))'


# Here we start with the page layout
layout = html.Div(
            style={
                'background-image': background_img,  # Specify the path to your image file
                'background-size': 'cover',  # Cover the entire container
                'background-position': 'center',  # Center the background image
                'height': '100vh',  # Set the height to full viewport height
            },
    children = [
    dbc.Card(
        dbc.CardBody([
            # ROW 1
            dbc.Row([
                # COLUMN 1
                dbc.Col([
                html.Div([
                    html.Div([
                        dbc.Card(
                            dbc.CardBody(
                                [
                                html.H4('Portfolio', className='card-title', style=card_config['cardtitle']),
                                html.P(id='pnl', className='card-text', style=card_config['cardtext']),
                                ]
                                ),
                            style=card_config['cardstyle']),
                        ], className = 'w-100'),
                    html.Div([
                        dbc.Card(
                            dbc.CardBody(
                                    [
                                    html.H4('Sharpe Ratio', className='card-title', style=card_config['cardtitle']),
                                    html.P(id='sharp', className='card-text', style=card_config['cardtext'])
                                    ]
                                    ),
                                style=card_config['cardstyle']),
                            ], className = 'w-100'),                                
                    html.Div([
                    dbc.Card(
                            dbc.CardBody(
                                    [
                                    html.H4('DrawDown', className='card-title', style=card_config['cardtitle']),
                                    html.P(id='maxdd', className='card-text', style=card_config['cardtext'])
                                    ]
                                    ),
                                style=card_config['cardstyle']), 
                            ], className = 'w-100'),
                    html.Div([
                        dbc.Card(
                            dbc.CardBody(
                                    [
                                    html.H4('WinRate', className='card-title', style=card_config['cardtitle']),
                                    html.P(id='winrate', className='card-text', style=card_config['cardtext'])
                                    ]
                                    ),
                                style=card_config['cardstyle']),
                            ], className = 'w-100'),
                    html.Div([
                        dbc.Card(
                            dbc.CardBody(
                                    [
                                    html.H4('ProfitRatio', className='card-title', style=card_config['cardtitle']),
                                    html.P(id='pr', className='card-text', style=card_config['cardtext'])
                                    ]
                                    ),
                                style=card_config['cardstyle']),
                            ], className = 'w-100'),
                    ], className = 'vstack gap-3'),
                    ], width=2),
                # COLUMN 2
                dbc.Col([
                    dcc.DatePickerRange(
                        id='date-range-picker',
                        min_date_allowed=df.index.min()-pd.Timedelta(days=1),
                        max_date_allowed=df.index.max(),
                        start_date=df.index.min() - pd.Timedelta(days=1),
                        end_date=df.index.max()
                    ),
                    dbc.Button('MTD', id='mtd', n_clicks=0, style={'margin-left': '40px'}),
                    dbc.Button('QTD', id='qtd', n_clicks=0, style={'margin-left': '12px'}),
                    dbc.Button('YTD', id='ytd', n_clicks=0, style={'margin-left': '12px'}),
                    
                    html.Br(),
                    html.Div(style={'height': '15px'}),
                    dcc.Graph(
                        id='ytd_plot',
                        figure={},
                        style={'height':'46vh', 'border-radius': '15px', 'border':'4px solid #C0C0C0'}
                    ),
                ], width=6),
                #COLUMN 3
                dbc.Col([
                    dbc.Row([
                        html.Div([
                            dbc.Card(
                                dbc.CardBody([
                                    html.H4('Legend Explanation', className='card-title', style={'font-size':'16px', 'font-weight':'bold', 'color':"#000000"}),
                                    html.P("'All-in': selected stocks trade each day as per algorithmic prediction", className='card-text', style={'color':colors_config['colors']['palet'][0]}),
                                    html.P("'Conditional': selected stock portfolio is divided into smaller clusters. A cluster will become active only when certain daily criteria are checked.", className='card-text', style={'color':colors_config['colors']['palet'][1]}),
                                    html.P("'Leveraged': like 'conditional', however when a cluster is inactive more weight is given to active clusters", className='card-text', style={'color':colors_config['colors']['palet'][2]})
                                    ]),
                                style={
                                       'margin-left':'0px',
                                       'border-radius': '10px',
                                       'border':'4px solid #ddd',
                                       'font-size':'14px',
                                       'background-color': '#FFFFFF'
                                       }
                                ),
                            ]),
                        ]),
                    html.Br(),
                    dbc.Row([
                        html.Div([
                            dcc.Graph(
                                id= 'targetplot',
                                figure={},
                                style={'height':'27vh', 'border-radius': '15px', 'border':'4px solid #C0C0C0'}
                                ),
                            ]),
                        ]),
                    ], width=4,   # Adjust the margin between columns
                    ),
                ], style={"margin-right": "15px", "margin-left": "15px"}),
            html.Br(),
            # ROW 2
            dbc.Row([
                # COLUMN 1
                dbc.Col([
                    dcc.Graph(
                        figure = plots_generator.generate_month_bars(),
                        responsive=True,
                        style={'height':'40vh', 'border-radius': '10px', 'border':'4px solid #ddd'},
                        ),               
                    ], width = 5),
                # COLUMN 2
                dbc.Col([
                    html.Div([
                        dcc.Graph(
                            id='multiplot',
                            figure = {},
                            style={'height':'40vh','border-radius': '10px', 'border':'4px solid #ddd'},    
                            )
                        ]),
                    ], width=6),                       
                ], style={"margin-right": "15px", "margin-left": "15px"}),  # CLOSING ROW 2
            ], style = {'background-image': background_img,}  # Specify the path to your image file
            ) # CLOSING CARDBODY
        ), # CLOSING CARD
    ] #CLOSING children
    ) #CLOSING DIV

# =============================================================================
# # The Outputs contain the interactive elements, this page has 8 interactive elements,
# # with the multiplot containing another 8 elements
# # The Inputs are the choices the User can make, which on this page is only time period related,
# # either through a drop picker or a button.
# 
# =============================================================================
@callback(
    [
    Output('ytd_plot', 'figure'),
    Output('targetplot', 'figure'),
    Output('pnl', 'children'),
    Output('sharp', 'children'),
    Output('maxdd', 'children'),
    Output('winrate', 'children'),
    Output('pr', 'children'),
    Output('multiplot', 'figure')
    ],
    [
    Input('date-range-picker', 'start_date'),
    Input('date-range-picker', 'end_date'),
    Input('mtd', 'n_clicks'),
    Input('qtd', 'n_clicks'),
    Input('ytd', 'n_clicks'),
    ],
)

def update_page1(start_date, end_date, mtd, qtd, ytd):
    
    #update the dates according to the preferred button click
    #also updating the titles to show the user the graphs are updating to their selection in the titles
    # target updates for the target graph is chosen to be (a bit random) 70% of the average performance over the last 6 years
    ctx = dash.callback_context
    button_id = None
    start_date = df.index[0]
    if ctx.triggered:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if button_id == 'mtd':
        start_date = pd.Timestamp.now().to_period('M').start_time
        figln_title = 'MTD'
        target = 0.7*0.0158
    elif button_id == 'qtd':
        start_date =  pd.Timestamp.now().to_period('Q').start_time
        figln_title = 'QTD'
        target = 0.7 * 0.0705 
    elif button_id == 'ytd':
        start_date = pd.Timestamp.now().to_period('Y').start_time
        figln_title = 'YTD'
        target = 0.7 * 0.3543  
    else:
        figln_title = 'YTD'
        target = 0.7 * 0.3543
    
    # Graphs are all created on a helper page plots_generator. All other metrics are calculated on a helper page metrics_generator.
    figln, fig_target = plots_generator.generate_perf_plot1(start_date, end_date, figln_title, target) 
    pnl, sharp, maxdd, winrate, pr = metrics_generator.generate_metrics(start_date, end_date)
    multiplot = plots_generator.generate_multi_barplot(start_date, end_date)
    
    # the order of returns should be the same as the order of Output in the callbacks.
    return figln, fig_target, pnl, sharp, maxdd, winrate, pr, multiplot