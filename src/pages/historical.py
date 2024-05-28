# -*- coding: utf-8 -*-
"""
Created on Sun May  5 11:56:58 2024

@author: Gebruiker
"""

import pandas as pd
import numpy as np
import datetime

import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plots_generator
import metrics_generator
from config import colors_config, card_config

# =============================================================================
# This page contains historical information before the running year.
# It looks very similar to the opening page, interactivity coming from choosing the time period.
# The only addition is a pie graph showing the average performance over a quarter or a month (interactive).
# Also the multiplot is showing 2 extra graphs.
# 
# The page is built up out of 2 rows. 
# The first row containing 3 columns for 1) info cards 2) interactive buttons and a linegraph 3) dropdown for the piegraph below
# The second row containing 2 columns for 1) a non-interactive barchart and 2) an interactive multiplot barchart 
# 
# =============================================================================

#making this page to be read
dash.register_page(__name__)

# setting the background color in an image
background_img = 'linear-gradient(to left, rgba(39,83,81,0.75), rgba(0,0,0,1))'
# defining the choices for dropdown of piechart
dropdown_values = {
    'choose_window': ['Monthly','Quarterly']
    }

# here the layout of the page starts
layout = html.Div(
            style={
                  'background-image': background_img,  # Specify the path to your image file
                  'background-size': 'cover',  # Cover the entire container
                  'background-position': 'center',  # Center the background image
                  'height': '100vh',  # Set the height to full viewport height
                  'padding': '0px'  # Add some padding for better visibility of content
                  },
            
    children = [
    dbc.Card(
        dbc.CardBody([
            #ROW 1
            dbc.Row([
                # COLUMN 1,  contains 5 cards with info: PnL, Sharpe, Drawdown, %Winners, avg Winning pnl vs avg Losing pnl
                dbc.Col([
                html.Div([
                    html.Div([
                        dbc.Card(
                            dbc.CardBody(
                                [
                                html.H4('Portfolio', className='card-title', style=card_config['cardtitle']),
                                html.P(id='pnlh', className='card-text', style=card_config['cardtext']),
                                ]
                                ),
                            style=card_config['cardstyle']),
                        ], className = 'w-100'),
                    html.Div([
                        dbc.Card(
                            dbc.CardBody(
                                [
                                html.H4('Sharpe Ratio', className='card-title', style=card_config['cardtitle']),
                                html.P(id='sharph', className='card-text', style=card_config['cardtext'])
                                ]
                                ),
                            style=card_config['cardstyle']),
                        ], className = 'w-100'),
                    html.Div([
                        dbc.Card(
                            dbc.CardBody(
                                [
                                html.H4('DrawDown', className='card-title', style=card_config['cardtitle']),
                                html.P(id='maxddh', className='card-text', style=card_config['cardtext'])
                                ]
                                ),
                            style=card_config['cardstyle']),
                        ], className = 'w-100'),
                    html.Div([
                        dbc.Card(
                            dbc.CardBody(
                                [
                                html.H4('WinRate', className='card-title', style=card_config['cardtitle']),
                                html.P(id='winrateh', className='card-text', style=card_config['cardtext'])
                                ]
                                ),
                            style=card_config['cardstyle']),
                        ], className = 'w-100'),
                    html.Div([
                        dbc.Card(
                            dbc.CardBody(
                                [
                                html.H4('ProfitRatio', className='card-title', style=card_config['cardtitle']),
                                html.P(id='prh', className='card-text', style=card_config['cardtext'])
                                ]
                                ),
                            style=card_config['cardstyle']),
                        ], className = 'w-100'),
                    ], className = 'vstack gap-3'),
                    ], width=2),
                # COLUMN 2, contains on a horizontal row the time period buttons the user can choose, and under it the line plot for the performance
                dbc.Col([
                    dbc.Button('2018', id='y2018', n_clicks=0, style={'margin-left': '0px'}),
                    dbc.Button('2019', id='y2019', n_clicks=0, style={'margin-left': '12px'}),
                    dbc.Button('2020', id='y2020', n_clicks=0, style={'margin-left': '12px'}),
                    dbc.Button('2021', id='y2021', n_clicks=0, style={'margin-left': '12px'}),
                    dbc.Button('2022', id='y2022', n_clicks=0, style={'margin-left': '12px'}),
                    dbc.Button('2023', id='y2023', n_clicks=0, style={'margin-left': '12px'}),
                    dbc.Button('Total', id='yall', n_clicks=0, style={'margin-left': '12px'}),
                    
                    html.Div(style={'height': '10px'}),
    
                    dcc.Graph(
                        id='historical_plot',
                        figure={},
                        style={'height':'48vh', 'border-radius': '15px', 'border':'4px solid #C0C0C0'}
                        ),
                    html.Br(),
                    ], width=6),
                # COLUMN 3, contains a dropdown menu for the piechart and the piechart showing the cyclical performance    
                dbc.Col([
                    dcc.Dropdown(
                        id='tf_dd',
                        options=[{'label':period, 'value': period} for period in dropdown_values['choose_window'] ],
                        value='Quarterly'
                        ),
                    html.Div(style={'height': '10px'}),
                    dcc.Graph(
                        id='donut',
                        figure = {},
                        style={'height':'48vh','border-radius': '15px', 'border':'4px solid #ddd'}
                        )            
                    ], width=4),
                ], style={"margin-right": "15px", "margin-left": "15px"}  # Adjust the margin between columns
                ),
            #ROW 2    
            dbc.Row([
                # COLUMN 1, contains the non-interactive barchart for annual performances
                dbc.Col([
                    dcc.Graph(
                        figure = plots_generator.generate_annual_bars(),
                        responsive=True,
                        style={'height':'36vh', 'border-radius': '10px', 'border':'4px solid #ddd'},
                        ),               
                    ], width = 4),
                # COLUMN 2, contain the interactive multi barcharts if metrics of interest
                dbc.Col([
                    dcc.Graph(
                        id = 'multi_histplot',
                        figure = {},
                        style={'height':'36vh', 'border-radius': '10px', 'border':'4px solid #ddd'},
                        )
                    ], width=8),
                ], style={"margin-right": "15px", "margin-left": "15px"}),
            ], style = {'background-image': background_img}),  # END CARDBODY
        ),  # END CARD
        ],  # END CHILDREN          
    )   #END Div

# =============================================================================
# # The Outputs contain the interactive elements, this page has 8 interactive elements,
# # with the multiplot containing another 10 elements
# # The Inputs are the choices the User can make, which on this page is only time period related,
# # either through a button or a dropdown menu.
# 
# =============================================================================

@callback(
    [
     Output('historical_plot', 'figure'),
     Output('pnlh', 'children'),
     Output('sharph', 'children'),
     Output('maxddh', 'children'),
     Output('winrateh', 'children'),
     Output('prh', 'children'),
     Output('multi_histplot', 'figure'),
     Output('donut', 'figure'),
     ], 
    [
     Input('y2018', 'n_clicks'),
     Input('y2019', 'n_clicks'),
     Input('y2020', 'n_clicks'),
     Input('y2021', 'n_clicks'),
     Input('y2022', 'n_clicks'),
     Input('y2023', 'n_clicks'),
     Input('yall', 'n_clicks'),
     Input('tf_dd', 'value'),
     ],
    )
# function to update interactive elements, make sure the arguments are all Input elements from the callback
def update_page2(y2018, y2019, y2020, y2021, y2022, y2023, yall, tf_dd):
    ctx = dash.callback_context
    button_id = None
    if ctx.triggered:
   #     return dash.no_update
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    # Determine date range based on button click
    if button_id == 'y2018':
        start_date = '01-01-2018'
        end_date = '31-12-2018'
        plot_title = '2018'
    elif button_id == 'y2019':
        start_date = '01-01-2019'
        end_date = '31-12-2019'
        plot_title = '2019'
    elif button_id == 'y2020':
        start_date = '01-01-2020'
        end_date = '31-12-2020'
        plot_title = '2020'
    elif button_id == 'y2021':
        start_date = '01-01-2021'
        end_date = '31-12-2021'
        plot_title = '2021'
    elif button_id == 'y2022':
        start_date = '01-01-2022'
        end_date = '31-12-2022'
        plot_title = '2022'
    elif button_id == 'y2023':
        start_date = '01-01-2023'
        end_date = '31-12-2023'
        plot_title = '2023'
    elif button_id == 'yall':
        start_date = '01-01-2018'
        end_date = '31-12-2023'
        plot_title = 'Total'
    else:
        start_date = '01-01-2018'
        end_date = '31-12-2023'
        plot_title = 'Total' 
    
    hist_plot = plots_generator.generate_histperf(start_date, end_date, plot_title)
    pnlh, sharph, maxddh, winrateh, profitrateh = metrics_generator.generate_hist_metrics(start_date, end_date)
    multi_plot2 = plots_generator.generate_multi_hist_bars(start_date, end_date)
    donut = plots_generator.generate_donut(tf_dd)
    
    # make sure the returns have the same order as the callbacks
    return hist_plot, pnlh, sharph, maxddh, winrateh, profitrateh,  multi_plot2, donut