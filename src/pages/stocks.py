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
import plotly.graph_objects as go

import plots_generator
import metrics_generator
from config import colors_config, card_config

# =============================================================================
# This page contains information about the individual performance of stocks active
# within this running year.
# Interactivitiy comes from selected stock (dropdown),the time periods MTD, QTD, YTD (buttons) 
# and selected strategy (dropdown).
#
# It has 1) a Line Graph showing the performance of individual stocks 
#        2) a Table showing the top12 and worst12 stocks of the selected time period and strategy
#        3) a Card showing fundamental and market data info and 2 graphs,
#           one a barchart with all time performance and one a gauge chart to show sentiment
#        4) 2 Barcharts showing the top 10 and worst 10 stocks performance of all time
# 
# The page has 1 row only containing 3 columns
# Column 1 has time period buttons
# Column 2 has dropdown for stocks, the line graph, dropdown for strategy and the table
# Column 3 has the Card and 2 barcharts 
# 
# =============================================================================

#helper functions for coloring and formatting
def get_fill_color(sentiment):
    if sentiment == 'STRONG SELL':
        fill_color = '#7C291D'
    elif sentiment == 'SELL':
        fill_color = '#C84835'
    elif sentiment == 'STRONG BUY':
        fill_color = '#1C5A2E'
    elif sentiment == 'BUY':
        fill_color =  '#32904D'
    elif sentiment == 'NEUTRAL':
        fill_color = 'lightblue'
    elif sentiment == 'None':
        fill_color = 'gray'
    return fill_color

def get_change_color(last_change):
    if last_change < 0:
        change_color = "red"
    elif last_change > 0:
        change_color = 'green'
    else:
        change_color = 'gray'
    return change_color

def format_change(last_change):
    sign = "+" if last_change > 0 else ""
    triangle = "▲" if last_change > 0 else "▼"
    last_change = round(100 * last_change, 2)
    return f"{triangle}{sign}{last_change}%"

# donwloading data
fname = 'DC_2024trades.csv'
df = pd.read_csv(f'../{fname}', parse_dates = ['date'], index_col = 'date')

metaname = 'financial_metadata.csv'
dfmeta = pd.read_csv(f'../{metaname}')

# dataprep for top12 table on stocks page
dftop15 = df.groupby('ticker').pnl_cl.sum().sort_values(ascending=False).head(12)
dftop15 = dftop15.reset_index()
dftop15 = dftop15.rename(columns={'ticker':'Top12', 'pnl_cl':'Top_PnL'})

dfworst15 = df.groupby('ticker').pnl_cl.sum().sort_values(ascending=True).head(12)
dfworst15 = dfworst15.reset_index()
dfworst15 = dfworst15.rename(columns={'ticker':'Worst12', 'pnl_cl':'Worst_PnL'})

dft = round(pd.concat([dftop15, dfworst15], axis=1), 4)
percentage = FormatTemplate.percentage(2)
table1_columns = [
        dict(id='Top12', name='Top 12'),
        dict(id='Top_PnL', name='P/L', type='numeric', format=percentage),
        dict(id='Worst12', name='Worst 12'),
        dict(id='Worst_PnL', name='P/L', type='numeric', format=percentage),
      ]

#making this page to be read
dash.register_page(__name__)

# setting the background color of the page in an image
background_img = 'linear-gradient(to left, rgba(39,83,81,0.75), rgba(0,0,0,1))'

# here the layout of the page starts
layout = html.Div(
            style={
                'background-image': background_img,  # Specify the path to your image file
                'background-size': 'cover',  # Cover the entire container
                'background-position': 'center',  # Center the background image
                'height': '100vh',  # Set the height to full viewport height
                'padding': '30px'  # Add some padding for better visibility of content
            },

    children=[
        # ROW 1
        dbc.Row([
            # COLUMN 1 for the Buttons
            dbc.Col([
                dbc.Button('MTD', outline=False, color='primary', id='mtd', n_clicks=0),
                html.Div(style={'height': '10px'}),                
                dbc.Button('QTD', outline=False, color='primary', id='qtd', n_clicks=0),
                html.Div(style={'height': '10px'}),
                dbc.Button('YTD', outline=False, color='primary', id='ytd', n_clicks=0),
                ], width = 1),
            # COLUMN 2 with Dropdown for stocks, linegraph for stocks, with Dropdown for strategy and interactive table
            dbc.Col([
                dcc.Dropdown(id='my_dpdn',
                              multi=False, 
                              value='RBA', 
                              options=[{'label':x, 'value':x} for x in sorted(df['ticker'].unique())],
                              style={'border-radius': '15px'}
                              ),
                html.Div(style={'height': '10px'}),
                dcc.Graph(id='bar-fig',
                          figure={},
                          style={'height':'40vh', 'border-radius': '15px', 'border':'4px solid #ddd'}
                          ),
                html.Br(),
                dcc.Dropdown(id='table_dpdn',
                             multi=False,
                             value= 'Leveraged',
                             options = ['Leveraged', 'All-in', 'Conditional'],
                             style ={'border-radius': '15px'}
                             ),
                html.Div(style={'height': '10px'}),
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
                        ),
                        ]),
                ], width = 5),
            # COLUMN 3 with CARD and 2 barcharts
            dbc.Col([
                dbc.Card(
                    dbc.CardBody(id='financial-info-card'),
                    className='mt-3',
                    style={'backgroundColor': 'black', 'color': 'white'}
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
            ]),   
        ]
    )

# =============================================================================
# # The Outputs contain the interactive elements, this page has 5 interactive elements,
# # with the card containing another 2 elements
# # The Inputs are the choices the User can make, which on this page is time period related through buttons,
# # and 2 dropdown menus for stock and strategy selection
# 
# =============================================================================
                
@callback(
    [
     Output('bar-fig', 'figure'),
     Output('table', 'data'),
     Output('best_bar', 'figure'),
     Output('worst_bar', 'figure'),
     Output('financial-info-card', 'children'),
     ],
    [
     Input('mtd', 'n_clicks'),
     Input('qtd', 'n_clicks'),
     Input('ytd', 'n_clicks'),
     Input('my_dpdn', 'value'),
     Input('table_dpdn', 'value')
     ],
    )

# function to update interactive elements, make sure the arguments are all Input elements from the callback
def update_stockspage(mtd, qtd, ytd, selected_stock, selected_strat):
    
    # based on button selection set the time period dates
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
    
    # create the line graph
    figln = plots_generator.generate_individual_stock_graph(selected_stock, start_date)

    # dataprep for top12 table
    dfc = df[df.index >= start_date]
    dftop15 = dfc.groupby('ticker')[strat].sum().sort_values(ascending=False).head(12)
    dftop15 = dftop15.reset_index()
    dftop15 = dftop15.rename(columns={'ticker':'Top12', strat:'Top_PnL'})
    
    dfworst15 = dfc.groupby('ticker')[strat].sum().sort_values(ascending=True).head(12)
    dfworst15 = dfworst15.reset_index()
    dfworst15 = dfworst15.rename(columns={'ticker':'Worst12', strat:'Worst_PnL'})
    
    dft = round(pd.concat([dftop15, dfworst15], axis=1), 4)  #this dataframe has to be returned as a dictionary for the dash table to interpret it correctly
    
    # create the 2 barcharts 
    figbar, figbar2 = plots_generator.generate_bestinhistory_bar(strat)
    
    # dataprep for Card info
    row = dfmeta[dfmeta['ticker'] == selected_stock].iloc[0]
    
    status_color = "green" if row['Status'] == 'Active' else "red"
    if row['Status'] == 'Inactive':
        row['Sentiment'] = 'None'
    fill_color = get_fill_color(row['Sentiment'])
    change_color = get_change_color(row['Last Change'])
    ytd_color = get_change_color(row['YTD'])
    
    # create the Card, including 2 graphs
    card_content = [
        html.H4(f"{selected_stock}, {row['Name']}", className="card-title", style={'color': '#36D0B7'}),
        html.H6(
            [
            f"Last Close: ${row['Last Close']}  ,  ",
            html.Span(
                format_change(row['Last Change']),
                style={'color': change_color}
            ),
            ",  YTD: ",
            html.Span(
                format_change(row['YTD']),
                style = {'color': ytd_color},
                ),
            ],
            className="card-subtitle"
        ),
        html.Hr(style={'borderTop': '1px dashed white'}),
        dbc.Row([
            dbc.Col([
                html.H6("Fundamentals", className="card-subtitle", style={'color':colors_config['colors']['text'], 'font-weight': 'underline'}),
                html.P(f"Sector: {row['Sector']}"),
                html.P(f"Market Cap: {row['Market Cap']}"),
                html.P(f"EPS: {row['EPS']}"),
                html.P(f"Net Income: {row['Net Income']}"),
                html.P(f"P/E Ratio: {row['P/E Ratio']}"),
                html.P(f"Div Yield: {row['Div Yield']}"),
                html.Hr(),
                dcc.Graph(figure=plots_generator.generate_individual_stock_histbar(dfmeta[dfmeta['ticker'] == selected_stock][['All-Time AI', 'All-Time Cond', 'All-Time Lvg']]))
            ], width=6),
            dbc.Col([
                html.H6("Market Quantitatives", className="card-subtitle", style={'color':colors_config['colors']['text'], 'font-weight': 'underline'}),
                html.P(f"Beta: {row['Beta']}"),
                html.P(f"10d Volatility: {row['Vola_10d']}"),
                html.P(f"3M Volatility: {row['Vola_3M']}"),
                html.P(f"Avg Daily Volume: {row['Avg_Daily_Volume']}"),
                html.Hr(),
                html.H6(f"Sentiment {row['Next Day']}", className="card-subtitle"),
                html.Span(row['Sentiment'], className = "card-subtitle", style={'color': fill_color, 'font-weight': 'bold'}),
                html.Hr(),
                dcc.Graph(figure=plots_generator.generate_gauge_nextpred(row['Prediction'], fill_color), config={'displayModeBar': False})
            ], width=6)
        ]),
        html.Div([
            html.Span(row['Status'], style={"color": status_color, "font-weight": "bold"}),
            html.Br(),
            html.Span(f"Last Updated: {row['Last Updated']}", style={"font-size": "smaller"}),
            ], style={"position": "absolute", "top": "10px", "right": "10px"}
            ),
        ]    

    # make sure the returns have the same order as the callbacks
    return figln, dft.to_dict('records'), figbar, figbar2, card_content
