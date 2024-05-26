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

dash.register_page(__name__, path='/')

# donwloading data
fname = 'DC_2024trades.csv'
#df = pd.read_csv(f'C:/Users/Gebruiker/Documents/Trading/DC_reports/{fname}', parse_dates = ['date'], index_col = 'date')
df = pd.read_csv(f'../{fname}', parse_dates = ['date'], index_col = 'date')
legend_labels = {'pnl_u': 'All-in', 'pnl_c': 'Conditional', 'pnl_cl': 'Leveraged'}

background_img = 'linear-gradient(to left, rgba(39,83,81,0.75), rgba(0,0,0,1))'
colors = {
    'bg_figs': '#FFFFFF',
    "surround_figs": '#5B706F',
    'text': '#7FDBFF',
    'font': 'Verdana',
    'palet': ['#294867', '#98691E', '#672967', '#1C778A', '#C0C0C0']
    }


layout = html.Div(
            style={
                'background-image': background_img,  # Specify the path to your image file
                'background-size': 'cover',  # Cover the entire container
                'background-position': 'center',  # Center the background image
                'height': '100vh',  # Set the height to full viewport height
  #              'padding': '30px'  # Add some padding for better visibility of content
            },
    children = [
    dbc.Card(
        dbc.CardBody([
            dbc.Row([
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
            
            dbc.Col([
                dbc.Row([
                    html.Div([
                        dbc.Card(
                            dbc.CardBody([
                                html.H4('Legend Explanation', className='card-title', style={'font-size':'16px', 'font-weight':'bold', 'color':"#000000"}),
                                html.P("'All-in': selected stocks trade each day as per algorithmic prediction", className='card-text', style={'color':colors['palet'][0]}),
                                html.P("'Conditional': selected stock portfolio is divided into smaller clusters. A cluster will become active only when certain daily criteria are checked.", className='card-text', style={'color':colors['palet'][1]}),
                                html.P("'Leveraged': like 'conditional', however when a cluster is inactive more weight is given to active clusters", className='card-text', style={'color':colors['palet'][2]})
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
        
        dbc.Row([
#            html.Div(style={'height': '15px'}),
            dbc.Col([
                dcc.Graph(
                    figure = plots_generator.generate_month_bars(),
                    responsive=True,
                    style={'height':'40vh', 'border-radius': '10px', 'border':'4px solid #ddd'},
                    ),               
                ], width = 5),
            dbc.Col([
                html.Div([
                    dcc.Graph(
                        id='multiplot',
                        figure = {},
                        style={'height':'40vh','border-radius': '10px', 'border':'4px solid #ddd'},    
                        )
                    ]),
                ], width=6),                       
            ], style={"margin-right": "15px", "margin-left": "15px"}),
        ], style = {'background-image': background_img,}  # Specify the path to your image file
        )
        ),
    ])



@callback(
    [
    Output('ytd_plot', 'figure'),
    Output('targetplot', 'figure'),
    Output('pnl', 'children'),
    Output('sharp', 'children'),
    Output('maxdd', 'children'),
    Output('winrate', 'children'),
    Output('pr', 'children'),
#    Output('sharp_bar', 'figure'),
#    Output('dd_bar', 'figure'),
#    Output('win_bar', 'figure'),
#    Output('to_bar', 'figure'),
#    Output('mto_bar', 'figure'),
#    Output('pr_bar', 'figure'),
#    Output('windays_bar', 'figure'),
#    Output('winmonths_bar', 'figure'),
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
    ctx = dash.callback_context
    button_id = None
    start_date = df.index[0]
    if ctx.triggered:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if button_id == 'mtd':
        start_date = pd.Timestamp.now().to_period('M').start_time
        figln_title = 'MTD'
        target = 0.7*0.0222
    elif button_id == 'qtd':
        start_date =  pd.Timestamp.now().to_period('Q').start_time
        figln_title = 'QTD'
        target = 0.7 * 0.0307 
    elif button_id == 'ytd':
        start_date = pd.Timestamp.now().to_period('Y').start_time
        figln_title = 'YTD'
        target = 0.7 * 0.3543  
    else:
        figln_title = 'YTD'
        target = 0.7 * 0.3543
    bar_title = figln_title
    target_title = figln_title
    
    figln, fig_target = plots_generator.generate_perf_plot1(start_date, end_date, figln_title, target) 
    pnl, sharp, maxdd, winrate, pr = metrics_generator.generate_metrics(start_date, end_date)
#    bar_sharp, bar_dd, bar_win, bar_to, bar_mto, bar_pr, bar_wind, bar_winm = plots_generator.generate_metrics_bars(start_date, end_date, bar_title)
    multiplot = plots_generator.generate_multi_barplot(start_date, end_date, bar_title)
    
    return figln, fig_target, pnl, sharp, maxdd, winrate, pr, multiplot