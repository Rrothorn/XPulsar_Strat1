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

dash.register_page(__name__)

background_img = 'linear-gradient(to left, rgba(39,83,81,0.5), rgba(39,83,81,1))'
#pie_dict = {'Monthly':total_monthlies, 'Quarterly':total_quarterlies}

dropdown_values = {
    'choose_window': ['Monthly','Quarterly']
    }


layout = html.Div(
            style={
                'background-image': background_img,  # Specify the path to your image file
                'background-size': 'cover',  # Cover the entire container
                'background-position': 'center',  # Center the background image
                'height': '100vh',  # Set the height to full viewport height
                'padding': '30px'  # Add some padding for better visibility of content
            },

    children=[
        dbc.Row([
            dbc.Col([
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H4('Portfolio', className='card-title', style=card_config['cardtitle']),
                            html.P(id='pnlh', className='card-text', style=card_config['cardtext']),
                        ]
                        ),
                        style=card_config['cardstyle']
                    ),
                html.Div(style={'height': '10px'}),
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H4('Sharpe Ratio', className='card-title', style=card_config['cardtitle']),
                            html.P(id='sharph', className='card-text', style=card_config['cardtext'])
                        ]
                        ),
                        style=card_config['cardstyle']
                    ),                                
                html.Div(style={'height': '10px'}),
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H4('DrawDown', className='card-title', style=card_config['cardtitle']),
                            html.P(id='maxddh', className='card-text', style=card_config['cardtext'])
                        ]
                        ),
                        style=card_config['cardstyle']
                    ),  
                html.Div(style={'height': '10px'}),
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H4('WinRate', className='card-title', style=card_config['cardtitle']),
                            html.P(id='winrateh', className='card-text', style=card_config['cardtext'])
                        ]
                        ),
                        style=card_config['cardstyle']
                    ),  
                html.Div(style={'height': '10px'}),
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H4('ProfitRatio', className='card-title', style=card_config['cardtitle']),
                            html.P(id='prh', className='card-text', style=card_config['cardtext'])
                        ]
                        ),
                        style=card_config['cardstyle']
                    ),                 
            ], width=2),
            
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
                    style={'height':'40vh', 'border-radius': '15px', 'border':'4px solid #C0C0C0'}
                ),
                html.Br(),
            ], width=5),
            
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
                    style={'height':'40vh', 'border-radius': '15px', 'border':'4px solid #ddd'}
                    )            
            ], width=3),
        ], style={"margin-right": "15px", "margin-left": "15px"}  # Adjust the margin between columns

        ),
        dbc.Row([
            html.Div(style={'height': '15px'}),
            dbc.Col([
                dcc.Graph(
                    figure = plots_generator.generate_annual_bars(),
                    responsive=True,
                    style={'height':'34vh', 'border-radius': '10px', 'border':'4px solid #ddd'},
                    ),               
                ], width = 4),
            dbc.Col([
                dcc.Graph(
                    id='sharph_bar',
                    figure={},
                    responsive=True,
                    style={'height':'17vh', 'width':'13rem', 'border-radius': '10px', 'border':'4px solid #ddd'}
                    ),
                html.Div(style={'height': '5px'}),
                dcc.Graph(
                    id='toh_bar',
                    figure={},
                    style={'height':'17vh', 'width':'13rem', 'border-radius': '10px', 'border':'4px solid #ddd'}
                    ), 
                ], width=2),
            dbc.Col([
                dcc.Graph(
                    id='ddh_bar',
                    figure={},
                    responsive=True,
                    style={'height':'17vh', 'width':'13rem', 'border-radius': '10px', 'border':'4px solid #ddd'}
                    ),
                html.Div(style={'height': '5px'}),
                dcc.Graph(
                    id='mtoh_bar',
                    figure={},
                    style={'height':'17vh', 'width':'13rem', 'border-radius': '10px', 'border':'4px solid #ddd'}
                    ),                                
                ], width=2),
            dbc.Col([
                dcc.Graph(
                    id='winh_bar',
                    figure={},
                    responsive=True,
                    style={'height':'17vh', 'width':'13rem', 'border-radius': '10px', 'border':'4px solid #ddd'}
                    ),
                html.Div(style={'height': '5px'}),
                dcc.Graph(
                    id='prh_bar',
                    figure={},
                    style={'height':'17vh', 'width':'13rem', 'border-radius': '10px', 'border':'4px solid #ddd'}
                    ),
                ], width=2),
            dbc.Col([
                dcc.Graph(
                    id='windh_bar',
                    figure={},
                    responsive=True,
                    style={'height':'17vh', 'width':'13rem', 'border-radius': '10px', 'border':'4px solid #ddd'}
                    ),
                html.Div(style={'height': '5px'}),
                dcc.Graph(
                    id='winmh_bar',
                    figure={},
                    style={'height':'17vh', 'width':'13rem', 'border-radius': '10px', 'border':'4px solid #ddd'}
                    ),
                ], width=2),                        

                        
            ], style={"margin-right": "15px", "margin-left": "15px"} 
        )
    ]
)



@callback(
    [
     Output('historical_plot', 'figure'),
     Output('pnlh', 'children'),
     Output('sharph', 'children'),
     Output('maxddh', 'children'),
     Output('winrateh', 'children'),
     Output('prh', 'children'),
     Output('sharph_bar', 'figure'),
     Output('toh_bar', 'figure'),
     Output('ddh_bar', 'figure'),
     Output('mtoh_bar', 'figure'),
     Output('winh_bar', 'figure'),
     Output('prh_bar', 'figure'),
     Output('windh_bar', 'figure'),
     Output('winmh_bar', 'figure'),
     Output('donut', 'figure'),
     ], 
    [
#     Input('date-range-picker', 'start_date'), 
#     Input('date-range-picker', 'end_date'),  
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
        
    bar_title = plot_title
    
    hist_plot = plots_generator.generate_histperf(start_date, end_date, plot_title)
    pnlh, sharph, maxddh, winrateh, profitrateh = metrics_generator.generate_hist_metrics(start_date, end_date)
    donut = plots_generator.generate_donut(tf_dd)
    bar_sharph, bar_toh, bar_ddh, bar_mtoh, bar_winh, bar_prh, bar_windh, bar_winmh = plots_generator.generate_hist_metrics_bars(start_date, end_date, bar_title)
    
    return hist_plot, pnlh, sharph, maxddh, winrateh, profitrateh,  bar_sharph, bar_toh, bar_ddh, bar_mtoh, bar_winh, bar_prh, bar_windh, bar_winmh, donut