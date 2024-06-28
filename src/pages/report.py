# -*- coding: utf-8 -*-
"""
Created on Mon May  6 20:23:33 2024

@author: Gebruiker
"""

import pandas as pd
import numpy as np
import datetime
from datetime import date
import base64

import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
from dash import dash_table
from dash.dash_table import DataTable, FormatTemplate
from dash.dash_table.Format import Format, Group

import plotly.express as px

import plots_generator
import metrics_generator
from config import colors_config, card_config
from report_creator import generate_report

cash = 1000000
# donwloading data
fname = 'DC_reports24.csv'
df = pd.read_csv(f'../{fname}', parse_dates = ['date'], index_col = 'date')
df[['trade_open', 'trade_close']] = round(df[['trade_open', 'trade_close']], 2)
df.betsize_cl = df.betsize_cl / cash
# table dataframe defined here to determine last row for styling
dft = round(df[['ticker',
                'buysell',
                'betsize_cl',
                'trade_open',
                'trade_close',
                'pnl_cl']][(df.index == df.index[-1]) & (df.pnl_cl != 0)]  , 5)
dftotal = dft[-1:]
dftotal.ticker = 'TOTAL'
dftotal.buysell = ''
dftotal.betsize_cl = dft.betsize_cl.sum()
dftotal.trade_open = ''
dftotal.trade_close = ''
dftotal.pnl_cl = dft.pnl_cl.sum()
dft = pd.concat([dft, dftotal])

# build table layout
percentage = FormatTemplate.percentage(3)
table1_columns = [
        dict(id='ticker', name='TICKER'),
        dict(id='buysell', name='B/S'),
        dict(id='betsize_cl', name='WEIGHT', type='numeric', format=percentage),
        dict(id='trade_open', name='trade_OPEN'),
        dict(id='trade_close', name='trade_CLOSE'),
        dict(id='pnl_cl', name='P/L', type='numeric', format=percentage),
      ]


dash.register_page(__name__)

background_img = 'linear-gradient(to left, rgba(39,83,81,0.5), rgba(0,0,0,1))'

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
                dcc.DatePickerSingle(
                    id='my-date-picker-single',
                    min_date_allowed=df.index[0],
                    max_date_allowed=df.index[-1],
                    initial_visible_month=df.index[-1],
                    date=df.index[-1].date()
                ),
                html.Div(id='output-container-date-picker-single'),
                
                
                dbc.Button('Download Daily Report', outline=False, color='primary', id='dwnl-report', n_clicks=0),
                html.A('', id='pdf-link', href=''),
                ], width = 3),
            
            dbc.Col([
                html.Div([dash_table.DataTable(
                        id='report_table',
  #                      data=dft.to_dict('records'),
                        data = [],
                        columns = table1_columns,
                        style_header={'backgroundColor': colors_config['colors']['surround_figs'],
                                      'color': colors_config['colors']['text'],
                                      "fontWeight": "bold",
                                      },
                        style_table = {'borderRadius': '10px', 'border':'4px solid #ddd', 'overflow': 'hidden'},
                        style_cell = {'color': colors_config['colors']['palet'][2],
                                      'font_family':'bold',
                                      },
                        style_data_conditional=[],
                        page_size = 22,
                        )
                    ]),
                ], width=8),
            ]),                 
        ]
    )

@callback(
    [Output('report_table', 'data'),
     Output('report_table', 'style_data_conditional'),
     Output('pdf-link', 'data')],
    [Input('my-date-picker-single', 'date'),
     Input('dwnl-report', 'n_clicks')]
)
# def update_page5(date_value, n_clicks):
    
#     dft = round(df[['ticker',
#                     'buysell',
#                     'betsize_cl',
#                     'trade_open',
#                     'trade_close',
#                     'pnl_cl']][(df.index == date_value) & (df.pnl_cl != 0)]  , 5)
#     dftotal = dft[-1:]
#     dftotal.ticker = 'TOTAL'
#     dftotal.buysell = ''
#     dftotal.betsize_cl = dft.betsize_cl.sum()
#     dftotal.trade_open = ''
#     dftotal.trade_close = ''
#     dftotal.pnl_cl = dft.pnl_cl.sum()
#     dft = pd.concat([dft, dftotal])
    
#     if dft.empty:
#         return [], [], None
    
#     last_row = len(dft.to_dict('records')) - 1
#     style_last_row = [
#         {
#             "if": {"row_index": last_row},
#             'backgroundColor': colors_config['colors']['palet'][2],
#             'color': '#FFFFFF',
#             "fontWeight": "bold",
            
#         },
#         ]
    
#     if n_clicks is None or n_clicks == 0:
#         return dft.to_dict('records'), None

#     pdf_report = generate_report(df, date_value)
#     pdf_report_base64 = base64.b64encode(pdf_report).decode('utf-8')
    
#     return [dft.to_dict('records'), style_last_row, pdf_report_base64]
def update_page5(date_value, n_clicks):
    # Assume df is already defined in your context
    dft = round(df[['ticker',
                    'buysell',
                    'betsize_cl',
                    'trade_open',
                    'trade_close',
                    'pnl_cl']][(df.index == date_value) & (df.pnl_cl != 0)], 5)
    
    if dft.empty:
        return [], [], None
    
    dftotal = dft[-1:].copy()
    dftotal['ticker'] = 'TOTAL'
    dftotal['buysell'] = ''
    dftotal['betsize_cl'] = dft['betsize_cl'].sum()
    dftotal['trade_open'] = ''
    dftotal['trade_close'] = ''
    dftotal['pnl_cl'] = dft['pnl_cl'].sum()
    dft = pd.concat([dft, dftotal])
    
    style_last_row = [
        {
            "if": {"filter_query": f"{{ticker}} = 'TOTAL'"},
            'backgroundColor': colors_config['colors']['palet'][2],
            'color': '#FFFFFF',
            "fontWeight": "bold",
        },
    ]
    
    if n_clicks is None or n_clicks == 0:
        return dft.to_dict('records'), style_last_row, None

    pdf_report = generate_report(df, date_value)
    pdf_report_base64 = base64.b64encode(pdf_report).decode('utf-8')
    
    return dft.to_dict('records'), style_last_row, pdf_report_base64