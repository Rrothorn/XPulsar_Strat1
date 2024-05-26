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

def get_fill_color(sentiment):
    if sentiment == 'STRONG SELL':
        fill_color = '#7C291D'
    elif sentiment == 'SELL':
        fill_color = '#C84835'
    elif sentiment == 'STRONG BUY':
        fill_color = '#1C5A2E'
    elif sentiment == 'BUY':
        fill_color ==  '#32904D'
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

# # Function to create the gauge plot
def create_gauge(prediction, fill_color):
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prediction,
        number={'valueformat': '.2%'},
        gauge={
            'axis': {'range': [-1, 1], 'tickformat':',.0%'},
            'bar': {'color': fill_color},
            'bar': {'thickness': 0},            
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [-1, -0.3], 'color': '#F5A490' },
                {'range': [-0.3, 0.3], 'color': '#F5ED90'},
                {'range': [0.3, 1], 'color': '#A7F590'},
                {'range': [0, prediction], 'color': fill_color, 'thickness': 0.8},
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': prediction
            }
        },
        domain={'x': [0, 1], 'y': [0, 1]},
    ))
    fig.update_layout(margin=dict(l=20, r=20, t=20, b=20), 
                      height=150,
                      paper_bgcolor = '#000000',
                      font_color = colors_config['colors']['text'],
                      )
    return fig

# donwloading data
fname = 'DC_2024trades.csv'
df = pd.read_csv(f'../{fname}', parse_dates = ['date'], index_col = 'date')

metaname = 'financial_metadata.csv'
dfmeta = pd.read_csv(f'../{metaname}')

# dataprep for top15 table on stocks page
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


dash.register_page(__name__)

background_img = 'linear-gradient(to left, rgba(39,83,81,0.75), rgba(0,0,0,1))'

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
    dftop15 = dfc.groupby('ticker')[strat].sum().sort_values(ascending=False).head(12)
    dftop15 = dftop15.reset_index()
    dftop15 = dftop15.rename(columns={'ticker':'Top12', strat:'Top_PnL'})
    
    dfworst15 = dfc.groupby('ticker')[strat].sum().sort_values(ascending=True).head(12)
    dfworst15 = dfworst15.reset_index()
    dfworst15 = dfworst15.rename(columns={'ticker':'Worst12', strat:'Worst_PnL'})
    
    dft = round(pd.concat([dftop15, dfworst15], axis=1), 4)
    
    figbar, figbar2 = plots_generator.generate_bestinhistory_bar(strat)
    
    row = dfmeta[dfmeta['ticker'] == selected_stock].iloc[0]
    
    status_color = "green" if row['Status'] == 'Active' else "red"
    if row['Status'] == 'Inactive':
        row['Sentiment'] = 'None'
    fill_color = get_fill_color(row['Sentiment'])
    change_color = 'green'
#    change_color = get_change_color(row['Last Change'])
    
    card_content = [
        html.H4(f"{selected_stock}, {row['Name']}", className="card-title"),
        html.H6(
            [
                f"Last Close: ${row['Last Close']}  ,",
                html.Span(
                    row['Last Change'],
                    style={'color': change_color}
                ),
                f",  YTD: {row['YTD']}"
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
                html.P(f"Div Yield: {row['Div Yield']}")
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
                dcc.Graph(figure=create_gauge(row['Prediction'], fill_color), config={'displayModeBar': False})
            ], width=6)
        ]),
        html.Div([
            html.Span(row['Status'], style={"color": status_color, "font-weight": "bold"}),
            html.Br(),
            html.Span(f"Last Updated: {row['Last Updated']}", style={"font-size": "smaller"}),
            ], style={"position": "absolute", "top": "10px", "right": "10px"}
            ),
        ]
    

    return figln, dft.to_dict('records'), figbar, figbar2, card_content
