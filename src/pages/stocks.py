# -*- coding: utf-8 -*-
"""
Created on Mon May  6 20:23:33 2024

@author: Gebruiker
"""

import pandas as pd
import numpy as np
import datetime

import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import dash_table
from dash.dash_table import DataTable, FormatTemplate
from dash.dash_table.Format import Format, Group

import plotly.express as px

import plots_generator
import metrics_generator
from config import colors_config, card_config

# donwloading data
fname = 'DC_2024trades.csv'
df = pd.read_csv(f'../{fname}', parse_dates = ['date'], index_col = 'date')

# dataprep for top15 table on stocks page
dftop15 = df.groupby('ticker').pnl_cl.sum().sort_values(ascending=False).head(15)
dftop15 = dftop15.reset_index()
dftop15 = dftop15.rename(columns={'ticker':'Top15', 'pnl_cl':'Top_PnL'})

dfworst15 = df.groupby('ticker').pnl_cl.sum().sort_values(ascending=True).head(15)
dfworst15 = dfworst15.reset_index()
dfworst15 = dfworst15.rename(columns={'ticker':'Worst15', 'pnl_cl':'Worst_PnL'})

dft = round(pd.concat([dftop15, dfworst15], axis=1), 4)
percentage = FormatTemplate.percentage(2)
table1_columns = [
        dict(id='Top15', name='Top 22'),
        dict(id='Top_PnL', name='P/L', type='numeric', format=percentage),
        dict(id='Worst15', name='Worst 22'),
        dict(id='Worst_PnL', name='P/L', type='numeric', format=percentage),
      ]


dash.register_page(__name__)

background_img = 'linear-gradient(to left, rgba(39,83,81,0.5), rgba(39,83,81,1))'

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
                dbc.Button('MTD', outline=False, color='primary', id='mtd', n_clicks=0),
                html.Div(style={'height': '10px'}),                
                dbc.Button('QTD', outline=False, color='primary', id='qtd', n_clicks=0),
                html.Div(style={'height': '10px'}),
                dbc.Button('YTD', outline=False, color='primary', id='ytd', n_clicks=0),
                ], width = 1),
            
            
            dbc.Col([
                dcc.Dropdown(id='my_dpdn',
                              multi=False, 
                              value='BBWI', 
                              options=[{'label':x, 'value':x} for x in sorted(df['ticker'].unique())],
                              style={'border-radius': '15px'}
                              ),
                html.Div(style={'height': '10px'}),
                dcc.Graph(id='bar-fig',
                          figure={},
                          style={'height':'50vh', 'border-radius': '15px', 'border':'4px solid #ddd'}
                          ),
                html.Br(),
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(id='best_bar',
                                  figure = {},
                                  style={'height':'32vh','border-radius': '15px', 'border':'4px solid #ddd'}
                                  )
                        ], width = 6),
                    dbc.Col([
                        dcc.Graph(id='worst_bar',
                                  figure = {},
                                  style={'height':'32vh','border-radius': '15px', 'border':'4px solid #ddd'}
                                  )
                        ], width = 6),
                    ]),
                ], width = 6),
             
            dbc.Col(width = 1),
            dbc.Col([
                dcc.Dropdown(id='table_dpdn',
                             multi=False,
                             value= 'Leveraged',
                             options = ['Leveraged', 'All-in', 'Conditional'],
                             style ={'border-radius': '15px'}
                             ),
                html.Br(),
                html.Div([dash_table.DataTable(
                        id='table',
                        data=dft.to_dict('records'),
                        columns = table1_columns,
                       # columns=[{'name': col, 'id': col, 'format':FormatTemplate.percentage(2)} for col in dft.columns],
                        style_header={'backgroundColor': colors_config['colors']['surround_figs'],
                                      'color': colors_config['colors']['text']},
                        style_table = {'borderRadius': '10px', 'border':'4px solid #ddd', 'overflow': 'hidden'},
                        style_cell = {'color': '#000000',
                                      'font_family':'bold',
                                      }
                        )
                    ])
                ], width = 4)
        ]),    
        ]
    )

                

@callback(
    [
     Output('bar-fig', 'figure'),
     Output('table', 'data'),
     Output('best_bar', 'figure'),
     Output('worst_bar', 'figure'),
     ],
    [
     Input('mtd', 'n_clicks'),
     Input('qtd', 'n_clicks'),
     Input('ytd', 'n_clicks'),
     Input('my_dpdn', 'value'),
     Input('table_dpdn', 'value')
     ],
    )

def update_stockspage(mtd, qtd, ytd, selected_stock, selected_strat):
    ctx = dash.callback_context
    button_id = None
    if ctx.triggered:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    # Determine date range based on button click
    if button_id == 'ytd':
        start_date = pd.Timestamp.now().to_period('Y').start_time
    elif button_id == 'qtd':
        start_date = pd.Timestamp.now().to_period('Q').start_time
    elif button_id == 'mtd':
        start_date = pd.Timestamp.now().to_period('M').start_time
    else:
        start_date = df.index[0]
    
    if selected_strat == 'All-in':
        strat = 'pnl_u'
    elif selected_strat == 'Conditional':
        strat = 'pnl_c'
    elif selected_strat == 'Leveraged':
        strat = 'pnl_cl'
    
    figln = plots_generator.generate_individual_stock_graph(selected_stock, start_date)
    
    # # dataprep for top15 table
    dfc = df[df.index >= start_date]
    dftop15 = dfc.groupby('ticker')[strat].sum().sort_values(ascending=False).head(22)
    dftop15 = dftop15.reset_index()
    dftop15 = dftop15.rename(columns={'ticker':'Top15', strat:'Top_PnL'})
    
    dfworst15 = dfc.groupby('ticker')[strat].sum().sort_values(ascending=True).head(22)
    dfworst15 = dfworst15.reset_index()
    dfworst15 = dfworst15.rename(columns={'ticker':'Worst15', strat:'Worst_PnL'})
    
    dft = round(pd.concat([dftop15, dfworst15], axis=1), 4)
    
    figbar, figbar2 = plots_generator.generate_bestinhistory_bar(strat)

    return figln, dft.to_dict('records'), figbar, figbar2
