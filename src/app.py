# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 17:23:18 2024

@author: Gebruiker
"""

import dash
import dash_table
from dash.dash_table import DataTable, FormatTemplate
from dash.dash_table.Format import Format, Group
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Output, Input, State
import plotly.express as px
import plotly.graph_objs as go

from sklearn.metrics import accuracy_score, f1_score  #for Classification

import pandas as pd
import numpy as np
import datetime

def Sharpe(pnl):
    """ Calculate annualised Sharpe Ratio """
    sharpe = (252 * pnl.sum()/ len(pnl)) / (pnl.std() * 252**0.5)
    return sharpe

def TurnOver(betsize):
    """ Calculate the average daily cash in play """
    avg_to = betsize.sum()/len(betsize)
    return avg_to
def WinRate(pnl):
    """ Calculate the winners vs the losers """
    winrate = len(pnl[pnl > 0]) / len(pnl[pnl != 0])
    return winrate
def ProfitRatio(pnl):
    """ Calculate the average profitable trades vs the average losing trade """
    profitratio = -pnl[pnl > 0].mean() / pnl[pnl < 0].mean()
    return profitratio
def DrawDown(pnl):
    """Calculate drawdown, or the max losing streak, given a return series."""
    wealth_index = 1000 * (1 + pnl).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdownser = (wealth_index - previous_peaks) / previous_peaks
    drawdown = drawdownser.min()
    return drawdown

def generate_individual_stock_graph(ticker, start_date):
    """ Creates a lineplot with performance of individual stocks """
    
    dfu = df[df.index >= start_date]
    dff = dfu[dfu['ticker'] == ticker]
    pnlcols = ['pnl_u', 'pnl_c', 'pnl_cl']
    dfcum = dff[pnlcols].cumsum()
    figln = px.line(dfcum, x=dfcum.index, y=['pnl_u','pnl_c','pnl_cl'],
                   color_discrete_sequence = colors['palet']
                    ).update_layout(
                            plot_bgcolor=colors['bg_figs'],
                            paper_bgcolor = colors['surround_figs'],
                            font_color = colors['text'],
                            font_family = colors['font'],
                            margin = {'l':20, 'r':40, 't':70, 'b':0, 'pad':0},
                            title = {'x':0.5, 'y':0.95, 'font':{'size':20}},
                            xaxis = {'title':''},
                            yaxis = {'title':'', 'tickformat': '.2%'},
                            legend = {'title': '', 'orientation':'h', 'y':1.14, 'xanchor':'right', 'x':0.9}
                            )
    figln.for_each_trace(lambda t: t.update(name=legend_labels[t.name]))

    return figln
    
# donwloading data
fname = 'DC_2024trades.csv'
df = pd.read_csv(f'../{fname}', parse_dates = ['date'], index_col = 'date')

print(df)

fname2 = 'XGB_trades18-23.csv'
dfhist = pd.read_csv(f'../{fname2}', parse_dates = ['date'], index_col = 'date')
dfhist = dfhist.rename(columns = {'weighted_pnl':'pnl_u', 'betsize':'betsize_u'})

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
        dict(id='Top15', name='Top 15'),
        dict(id='Top_PnL', name='P/L', type='numeric', format=percentage),
        dict(id='Worst15', name='Worst 15'),
        dict(id='Worst_PnL', name='P/L', type='numeric', format=percentage),
      ]

# dataprep for unresponsive table for historical performance
dates = ['01-01-2018','01-01-2019','01-01-2020', '01-01-2021', '01-01-2022', '01-01-2023', '01-01-2024']

dft2 = pd.DataFrame(columns=['Year', 'All-in', 'Conditional', 'Leveraged'])
starting = 2018
for i in range(len(dates)-1):
    start_year = dates[i]
    end_year = dates[i+1]
    dfc = dfhist[(dfhist.index > start_year) & (dfhist.index < end_year)]
    pnl_u = dfc.pnl_u.sum()
    pnl_c = dfc.pnl_c.sum()
    pnl_cl = dfc.pnl_cl.sum()
    dft2.loc[i]= pd.Series({
                    'Year': starting,
                    'All-in': pnl_u,
                    'Conditional': pnl_c,
                    'Leveraged': pnl_cl
                    })
    starting += 1

dft2 = round(dft2, 4)

percentage = FormatTemplate.percentage(2)
table2_columns = [
        dict(id='Year', name='Year'),
        dict(id='All-in', name='All-in', type='numeric', format=percentage),
        dict(id='Conditional', name='Conditional', type='numeric', format=percentage),
        dict(id='Leveraged', name='Leveraged', type='numeric', format=percentage),
    ]

# pick color scheme

# Green background  /light plots
background_img1 = 'linear-gradient(to left, rgba(39,83,81,0.5), rgba(39,83,81,1))'
colors1 = {
    'bg_figs': '#FFFFFF',
    "surround_figs": '#5B706F',
    'text': '#7FDBFF',
    'font': 'Verdana',
    'palet': ['#294867', '#98691E', '#672967', '#1C778A', '#C0C0C0']
    }
button_style1 = {'background-color': colors1['palet'][2],
                'color': 'white'}

# Light background / light plots

background_img2 = 'linear-gradient(to left, rgba(248,248,248,0.5), rgba(248,248,248,1))' 
colors2 = {
    'bg_figs': '#F6F6F1',
    "surround_figs": '#DFE0D9',
    'text': '#01125C',
    'font': 'Verdana',
    'palet': ['#E8B298', '#EDCC8B', '#BDD1C5', '#9DAAA2', '#A26360']
    }
button_style2 = {'background-color': colors2['palet'][0],
                'color': 'black'}


# Dark Background / light plots
background_img3 = 'linear-gradient(to left, rgba(8,9,16,0.75), rgba(8,9,16,1))'
colors3 = {
    'bg_figs': '#FFFFFF',
    "surround_figs": '#0C333B',
    'text': '#7FDBFF',
    'font': 'Verdana',
    'palet': ['#E25605', '#05E268', '#05BDE2', '#1C778A', '#C0C0C0']
    }
button_style3 = {'background-color': colors3['palet'][0],
                'color': 'white'}

# Black BG / same color in layers
background_img4 = 'linear-gradient(to left, rgba(0,0,0,0.9), rgba(0,0,0,1))'
colors4 = {
    'bg_figs': '#FFFFFF',
    "surround_figs": '#3F4344',
    'text': '#7FDBFF',
    'font': 'Verdana',
    'palet': ['#43A4AF', '#56CFDE', '#287F89', '#3F4344', '#C9E7EB']
    }
button_style4 = {'background-color': colors4['palet'][0],
                'color': 'white'}

# PICK COLOR THEME
colors = colors2
background_img = background_img2
button_style = button_style2



legend_labels = {'pnl_u': 'All-in', 'pnl_c': 'Conditional', 'pnl_cl': 'Leveraged'}

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.YETI],
               meta_tags=[{'name': 'viewport',
                          'concent': 'width=device-width, initial-scale=1.0'}]
               )
server = app.server

main_page_layout = html.Div(style={'background-image': background_img,
                             'height': '100vh'},
                      children=[
    dbc.Container([
        # Your layout components...
        dbc.Row([
            dbc.Col([
                html.Div(style={'height': '10px'}),
                dbc.Button('Main', outline=True, color='primary', className='me-2', id='mainpage', n_clicks=0, style=button_style),
                dbc.Button('Stocks', outline=True, color='primary', className='me-2', id='stockspage', n_clicks=0, style=button_style),
                dbc.Button('Historical', outline=True, color='primary', className='me-2', id='historicalpage', n_clicks=0, style=button_style),
                dbc.Button('Description', outline=False, color='primary', className='me-1', id='descriptionpage', n_clicks=0, style={'position': 'fixed', 'bottom': '10px', 'right': '10px'})
                ], width=3),
            
            dbc.Col([html.H1('Tracking a proxy Russell 1000 Portfolio',
                            className='text-center text-primary, mb-2', style={'font-size':'32px', 'color':colors['palet'][4]}), html.H2('An AI-powered Daily Machine Learning algorithm', className='text-center text-primary, mb-4', style={'font-size':'22px', 'color':colors['palet'][4]})
                     ], width=6),
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Col([
                    dcc.DatePickerRange(
                        id='date-range-picker',
                        min_date_allowed=df.index.min(),
                        max_date_allowed=df.index.max(),
                        start_date=df.index.min(),
                        end_date=df.index.max()
                    ),
                    dbc.Button('MTD', id='mtd', n_clicks=0, style={'margin-left': '40px'}),
                    dbc.Button('QTD', id='qtd', n_clicks=0, style={'margin-left': '12px'}),
                    dbc.Button('YTD', id='ytd', n_clicks=0, style={'margin-left': '12px'}),
                    ]),
                html.Div(style={'height': '10px'}),
                dcc.Graph(
                    id='line-fig',
                    figure={},
                    style={'height':'40vh', 'border-radius': '15px', 'border':'4px solid #C0C0C0'}
                )
            ], width=8),  # Set width for the first graph column
            
            dbc.Col([
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H4('Portfolio', className='card-title', style={'color':colors['text']}),
                            html.P(id='pnl', className='card-text', style={'color':colors['text']})
                        ]
                        ),
                        style={'width': '18rem', 'height': '5.5rem', 'border-radius': '15px', 'border':'1px solid #C0C0C0', 'background-color':colors['surround_figs']}
                    ),
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H4('Sharpe Ratio', className='card-title', style={'color':colors['text']}),
                            html.P(id='sharp', className='card-text', style={'color':colors['text']})
                        ]
                        ),
                        style={'width': '18rem', 'height': '5.5rem', 'border-radius': '15px', 'border':'1px solid #C0C0C0', 'background-color':colors['surround_figs']}
                    ),                
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H4('Max DrawDown', className='card-title', style={'color':colors['text']}),
                            html.P(id='maxdd', className='card-text', style={'color':colors['text']})
                        ]
                        ),
                        style={'width': '18rem', 'height': '5.5rem', 'border-radius': '15px', 'border':'1px solid #C0C0C0', 'background-color':colors['surround_figs']}
                    ),
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H4('WinRate', className='card-title', style={'color':colors['text']}),
                            html.P(id='winrate', className='card-text', style={'color':colors['text']})
                        ]
                        ),
                        style={'width': '18rem', 'height': '5.5rem', 'border-radius': '15px', 'border':'1px solid #C0C0C0', 'background-color':colors['surround_figs']}
                    ),                
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H4('ProfitRatio', className='card-title', style={'color':colors['text']}),
                            html.P(id='profitrate', className='card-text', style={'color':colors['text']})
                        ]
                        ),
                        style={'width': '18rem', 'height': '5.5rem', 'border-radius': '15px', 'border':'1px solid #C0C0C0', 'background-color':colors['surround_figs']}
                    )                
                ])
        ]),
        dbc.Row([
                html.Div(style={'height': '30px'}),
                    dbc.Col([
                    dcc.Graph(
                        id='sharpe',
                        figure={},
                        responsive=True,
                        style={'height':'19vh', 'border-radius': '10px', 'border':'4px solid #ddd'}),
                    ], width = 2),
    
                    dbc.Col([
                    dcc.Graph(
                        id='drawdown',
                        figure={},
                        style={'height':'19vh', 'border-radius': '10px', 'border':'4px solid #ddd'}),
                    ], width = 2),
                    
                    dbc.Col([
                    dcc.Graph(
                        id='win',
                        figure={},
                        style={'height':'19vh', 'border-radius': '10px', 'border':'4px solid #ddd'}),
                    ], width = 2),
                    
                    
                        html.Div(style={'height': '10px'}),
                        
                        dbc.Col([
                        dcc.Graph(
                            id='to',
                            figure={},
                            style={'height':'19vh', 'border-radius': '10px', 'border':'4px solid #ddd'})
                        ], width = 2),
                        
                        dbc.Col([
                        dcc.Graph(
                            id='max_to',
                            figure={},
                            style={'height':'19vh', 'border-radius': '10px', 'border':'4px solid #ddd'})
                        ], width = 2),

                        dbc.Col([
                        dcc.Graph(
                            id='pr',
                            figure={},
                            style={'height':'19vh', 'border-radius': '10px', 'border':'4px solid #ddd'})
                        ], width = 2),
                dbc.Card(
                    dbc.CardBody([
                        html.H4('Legend Explanation', className='card-title', style={'font-size':'14px', 'font-weight':'bold'}),
                        html.P("'All-in': selected stocks trade each day as per algorithmic prediction", className='card-text', style={'color':colors['palet'][0]}),
                        html.P("'Conditional': selected stock portfolio is divided into smaller clusters. A cluster will become active only when certain daily criteria are checked.", className='card-text', style={'color':colors['palet'][1]}),
                        html.P("'Leveraged': like 'conditional', however when a cluster is inactive more weight is given to active clusters", className='card-text', style={'color':colors['palet'][2]})
                        ]),
                    style={'width': '30rem', 'margin-left':'60px', 'border-radius': '10px', 'border':'4px solid #ddd', 'font-size':'11px'}
                    )
            ]),
        ])
    ])

second_page_layout =  html.Div(style={'background-image': background_img,
                             'height': '100vh'},
                      children=[
    dbc.Container([
        # Your layout components...
        dbc.Row([
            dbc.Col([
                html.Div(style={'height': '10px'}),
                dbc.Button('Main', outline=False, color='primary', className='me-2', id='mainpage', n_clicks=0, style=button_style),
                dbc.Button('Stocks', outline=True, color='primary', className='me-2', id='stockspage', n_clicks=0, style=button_style),
                dbc.Button('Historical', outline=True, color='primary', className='me-2', id='historicalpage', n_clicks=0, style=button_style),
                dbc.Button('Description', outline=False, color='primary', className='me-1', id='descriptionpage', n_clicks=0, style={'position': 'fixed', 'bottom': '10px', 'right': '10px'})
                ], width=3),
            
            dbc.Col([html.H1('Tracking a proxy Russell 1000 Portfolio',
                            className='text-center text-primary, mb-2', style={'font-size':'32px', 'color':colors['palet'][4]}), html.H2('An AI-powered Daily Machine Learning algorithm', className='text-center text-primary, mb-4', style={'font-size':'22px', 'color':colors['palet'][4]})
                     ], width=6),
        ]),
        html.Div(style={'height': '25px'}),
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
                          style={'border-radius': '15px', 'border':'4px solid #ddd'}
                          )
                ], width = 6, style = {'margin-right':'80px'}),
            dbc.Col([
                html.Div([dash_table.DataTable(
                        id='table',
                        data=dft.to_dict('records'),
                        columns = table1_columns,
                       # columns=[{'name': col, 'id': col, 'format':FormatTemplate.percentage(2)} for col in dft.columns],
                        style_header={'backgroundColor': colors['surround_figs'],'color': colors['text']},
                        style_table = {'borderRadius': '10px', 'border':'4px solid #ddd', 'overflow': 'hidden'}
                        )
                    ])
                ], width = 3)
        ]),
    ])
])

third_page_layout = html.Div(style={'background-image': background_img, 'height': '100vh'},
                      children=[
    dbc.Container([
        # Your layout components...
        dbc.Row([
            dbc.Col([
                html.Div(style={'height': '10px'}),
                dbc.Button('Main', outline=True, color='primary', className='me-2', id='mainpage', n_clicks=0, style=button_style),
                dbc.Button('Stocks', outline=True, color='primary', className='me-2', id='stockspage', n_clicks=0, style=button_style),
                dbc.Button('Historical', outline=True, color='primary', className='me-2', id='historicalpage', n_clicks=0, style=button_style),
                dbc.Button('Description', outline=False, color='primary', className='me-1', id='descriptionpage', n_clicks=0, style={'position': 'fixed', 'bottom': '10px', 'right': '10px'})
                ], width=3),
            
            dbc.Col([html.H1('Tracking a proxy Russell 1000 Portfolio',
                            className='text-center text-primary, mb-2', style={'font-size':'32px', 'color':colors['palet'][4]}), html.H2('An AI-powered Daily Machine Learning algorithm', className='text-center text-primary, mb-4', style={'font-size':'22px', 'color':colors['palet'][4]})
                     ], width=6),
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Col([
                    dcc.DatePickerRange(
                        id='date-range-picker',
                        min_date_allowed=dfhist.index.min(),
                        max_date_allowed=dfhist.index.max(),
                        start_date=dfhist.index.min(),
                        end_date=dfhist.index.max()
                    ),
                    dbc.Button('2018', id='y2018', n_clicks=0, style={'margin-left': '40px'}),
                    dbc.Button('2019', id='y2019', n_clicks=0, style={'margin-left': '12px'}),
                    dbc.Button('2020', id='y2020', n_clicks=0, style={'margin-left': '12px'}),
                    dbc.Button('2021', id='y2021', n_clicks=0, style={'margin-left': '12px'}),
                    dbc.Button('2022', id='y2022', n_clicks=0, style={'margin-left': '12px'}),
                    dbc.Button('2023', id='y2023', n_clicks=0, style={'margin-left': '12px'}),
                    dbc.Button('Total', id='yall', n_clicks=0, style={'margin-left': '12px'}),

                    ]),
                html.Div(style={'height': '10px'}),
                dcc.Graph(
                    id='hist_fig',
                    figure={},
                    style={'height':'40vh', 'border-radius': '15px', 'border':'4px solid #ddd'}
                   # style={'border-radius': '15px', 'border':'4px solid #ddd', 'display': 'inline-block'}
                )
            ], width=8),  # Set width for the first graph column
            
            dbc.Col([
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H4('Portfolio', className='card-title', style={'color':colors['text']}),
                            html.P(id='pnlh', className='card-text', style={'color':colors['text']})
                        ]
                        ),
                        style={'width': '18rem', 'height': '5.5rem', 'border-radius': '15px', 'border':'1px solid #C0C0C0', 'background-color':colors['surround_figs']}
                    ),
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H4('Sharpe Ratio', className='card-title', style={'color':colors['text']}),
                            html.P(id='sharph', className='card-text', style={'color':colors['text']})
                        ]
                        ),
                        style={'width': '18rem', 'height': '5.5rem', 'border-radius': '15px', 'border':'1px solid #C0C0C0', 'background-color':colors['surround_figs']}
                    ),                
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H4('Max DrawDown', className='card-title', style={'color':colors['text']}),
                            html.P(id='maxddh', className='card-text', style={'color':colors['text']})
                        ]
                        ),
                        style={'width': '18rem', 'height': '5.5rem', 'border-radius': '15px', 'border':'1px solid #C0C0C0', 'background-color':colors['surround_figs']}
                    ),
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H4('WinRate', className='card-title', style={'color':colors['text']}),
                            html.P(id='winrateh', className='card-text', style={'color':colors['text']})
                        ]
                        ),
                        style={'width': '18rem', 'height': '5.5rem', 'border-radius': '15px', 'border':'1px solid #C0C0C0', 'background-color':colors['surround_figs']}
                    ),                
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H4('ProfitRatio', className='card-title', style={'color':colors['text']}),
                            html.P(id='profitrateh', className='card-text', style={'color':colors['text']})
                        ]
                        ),
                        style={'width': '18rem', 'height': '5.5rem', 'border-radius': '15px', 'border':'1px solid #C0C0C0', 'background-color':colors['surround_figs']}
                    )                
                ])


        ]),
        
        html.Div(style={'height': '10px'}),
        
        dbc.Row([
            dbc.Col([
                dcc.Graph(
                    id='sharpeh',
                    figure={},
                    responsive=True,
                    style={'height':'19vh', 'border-radius': '10px', 'border':'4px solid #ddd'}),
                ], width = 2),
            dbc.Col([
                dcc.Graph(
                    id='drawdownh',
                    figure={},
                    style={'height':'19vh', 'border-radius': '10px', 'border':'4px solid #ddd'}),
                ], width = 2),
            dbc.Col([
                dcc.Graph(
                    id='winh',
                    figure={},
                    style={'height':'19vh', 'border-radius': '10px', 'border':'4px solid #ddd'}),                 
                ], width = 2),
            dbc.Col([
                html.Div([dash_table.DataTable(
                        data=dft2.to_dict('records'),
                        columns = table2_columns,
            #             columns=[{'name': col, 'id': col, 'format': {'locale':{'symbol': ['','%']}}} for col in dft2.columns],
                        style_header={'backgroundColor': colors['surround_figs'],'color':colors['text']},
                        style_table = {'borderRadius': '10px', 'border':'4px solid #ddd', 'overflow': 'hidden', 'margin-right': '50px'}
                        )                                
                    ])                  
                ], width = 5),
            dbc.Col([
                dcc.Graph(
                    id='toh',
                    figure={},
                    style={'height':'19vh', 'border-radius': '10px', 'border':'4px solid #ddd'}),
                ], width = 2),
            dbc.Col([
                dcc.Graph(
                    id='max_toh',
                    figure={},
                    style={'height':'19vh', 'border-radius': '10px', 'border':'4px solid #ddd'}),
                ], width = 2),
            dbc.Col([
                dcc.Graph(
                    id='prh',
                    figure={},
                    style={'height':'19vh', 'border-radius': '10px', 'border':'4px solid #ddd'})
                ], width = 2),
        ])
        # dbc.Row([
        #         html.Div(style={'height': '30px'}),
                
        #         dbc.Col([
        #         dcc.Graph(
        #             id='sharpeh',
        #             figure={},
        #             responsive=True,
        #             style={'height':'19vh', 'border-radius': '10px', 'border':'4px solid #ddd'}),
        #         ], width = 2),

        #         dbc.Col([
        #         dcc.Graph(
        #             id='drawdownh',
        #             figure={},
        #             style={'height':'19vh', 'border-radius': '10px', 'border':'4px solid #ddd'})
        #             #style={'border-raius': '15px', 'border':'4px solid #ddd', 'display': 'inline-block'}
        #         ], width = 2),
                
        #         dbc.Col([
        #         dcc.Graph(
        #             id='winh',
        #             figure={},
        #             style={'height':'19vh', 'border-radius': '10px', 'border':'4px solid #ddd'})
        #             #style={'border-radius': '15px', 'border':'4px solid #ddd', 'display': 'inline-block'}
        #         ], width = 2),
        #     ]),

        # dbc.Row([
        #         html.Div(style={'height': '10px'}),
                
        #         dbc.Col([
        #         dcc.Graph(
        #             id='toh',
        #             figure={},
        #             style={'height':'19vh', 'border-radius': '10px', 'border':'4px solid #ddd'})
        #             #style={'border-radius': '15px', 'border':'4px solid #ddd', 'display': 'inline-block'}
        #         ], width = 2),
                
        #         dbc.Col([
        #         dcc.Graph(
        #             id='max_toh',
        #             figure={},
        #             style={'height':'19vh', 'border-radius': '10px', 'border':'4px solid #ddd'})
        #             #style={'border-radius': '15px', 'border':'4px solid #ddd', 'display': 'inline-block'}
        #         ], width = 2),

        #         dbc.Col([
        #         dcc.Graph(
        #             id='prh',
        #             figure={},
        #             style={'height':'19vh', 'border-radius': '10px', 'border':'4px solid #ddd'})
        #             #style={'border-raius': '15px', 'border':'4px solid #ddd', 'display': 'inline-block'}
        #         ], width = 2),
        #     ]),
        #         html.Div([dash_table.DataTable(
        #                 data=dft2.to_dict('records'),
        #                 columns=[{'name': col, 'id': col, 'format': {'locale':{'symbol': ['','%']}}} for col in dft2.columns],
        #                 style_header={'backgroundColor': colors['surround_figs'],'color': 'white'},
        #                 style_table = {'borderRadius': '10px', 'border':'4px solid #ddd', 'overflow': 'hidden'}
        #                 )
        #             ])
         ])
    ])

last_page_layout =  html.Div(style={'background-image': background_img, 'height': '100vh'},
                      children=[
    dbc.Container([
        # Your layout components...
        dbc.Row([
            dbc.Col([
                html.Div(style={'height': '10px'}),
                dbc.Button('Main', outline=True, color='primary', className='me-2', id='mainpage', n_clicks=0, style=button_style),
                dbc.Button('Stocks', outline=True, color='primary', className='me-2', id='stockspage', n_clicks=0, style=button_style),
                dbc.Button('Historical', outline=True, color='primary', className='me-2', id='historicalpage', n_clicks=0, style=button_style),
                dbc.Button('Description', outline=False, color='primary', className='me-1', id='descriptionpage', n_clicks=0, style={'position': 'fixed', 'bottom': '10px', 'right': '10px'})
                ], width=3),
            
            dbc.Col([html.H1('Tracking a proxy Russell 1000 Portfolio',
                            className='text-center text-primary, mb-2', style={'font-size':'32px', 'color':colors['palet'][4]}), html.H2('An AI-powered Daily Machine Learning algorithm', className='text-center text-primary, mb-4', style={'font-size':'22px', 'color':colors['palet'][4]})
                     ], width=6),
        ]),
        html.Div(style={'height': '25px'}),
        dbc.Row([
            dbc.Col([
                dbc.Card(
                    dbc.CardBody([
                        html.H4('Description', className='card-title', style={'font-size':'17px', 'font-weight':'bold'}),
                        html.P("The algorithm presented herein represents a sophisticated, data-driven investment fund leveraging advanced AI and Machine Learning algorithms. This fund operates within the realm of single stocks equity portfolios, employing daily selections and reselections to optimise positions. The algorithm strategically engages both long and short positions to capitalise on market opportunities. ", className='card-text', style={'color':colors['palet'][3]}),
                        ]),
                    style={'width': '30rem', 'margin-left':'60px', 'border-radius': '14px', 'border':'4px solid #ddd', 'font-size':'14px'}
                    )

                ], width = 6),

            dbc.Col([
                dbc.Card(
                    dbc.CardBody([
                        html.H4('Objective', className='card-title', style={'font-size':'17px', 'font-weight':'bold'}),
                        html.P("The primary objective of the fund is to achieve compelling and sustainable returns by employing our pattern-detecting algorithm across an extensive array of stocks. Striving for both attractiveness and sustainability involves striking a balance between maximising returns and minimising risk, achieved through meticulous diversification and daily portfolio calibrations.", className='card-text', style={'color':colors['palet'][3]}),
                        ]),
                    style={'width': '30rem', 'margin-left':'60px', 'border-radius': '14px', 'border':'4px solid #ddd', 'font-size':'14px'}
                    )

                ], width = 6),
            ]),
        
        html.Div(style={'height': '25px'}),
        
        dbc.Row([
            dbc.Col([
                dbc.Card(
                    dbc.CardBody([
                        html.H4('Suitability', className='card-title', style={'font-size':'17px', 'font-weight':'bold'}),
                        html.P("This fund is specifically tailored for investors who (i) seek capital growth through exposure to global equity markets, (ii) are comfortable with a certain level of risk, (iii) acknowledge that their capital is at risk, understanding that the value of their investment may fluctuate both upward and downard.", className='card-text', style={'color':colors['palet'][3]}),
                        ]),
                    style={'width': '30rem', 'margin-left':'60px', 'border-radius': '10px', 'border':'4px solid #ddd', 'font-size':'14px'}
                    ),
                ], width = 6),
            
            dbc.Col([
                dbc.Card(
                    dbc.CardBody([
                        html.H4('Legend Explanation', className='card-title', style={'font-size':'17px', 'font-weight':'bold'}),
                        html.P("'All-in': selected stocks trade each day as per algorithmic prediction", className='card-text', style={'color':colors['palet'][0]}),
                        html.P("'Conditional': selected stock portfolio is divided into smaller clusters. A cluster will become active only when certain daily criteria are checked.", className='card-text', style={'color':colors['palet'][1]}),
                        html.P("'Leveraged': like 'conditional', however when a cluster is inactive more weight is given to active clusters", className='card-text', style={'color':colors['palet'][2]})
                        ]),
                    style={'width': '30rem', 'margin-left':'60px', 'border-radius': '10px', 'border':'4px solid #ddd', 'font-size':'14px'}
                    )

                ], width = 6)
        ]),
    ])
])


# App layout containing the navigation and page content
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content'),
    html.Button('Main Page', id='mainpage', n_clicks=0),
    html.Button('Second Page', id='stockspage', n_clicks=0),
    html.Button('Third Page', id='historicalpage', n_clicks=0),
    html.Button('Last Page', id='descriptionpage', n_clicks=0)
])

# Define callbacks to switch between main and secondary pages
@app.callback(
    Output('page-content', 'children'),
    [
     Input('url', 'pathname'),
     Input('mainpage', 'n_clicks'),
     Input('stockspage', 'n_clicks'),
     Input('historicalpage', 'n_clicks'),
     Input('descriptionpage', 'n_clicks')
     ],
    )

def display_page(pathname, main_clicks, secondary_clicks, third_clicks, other_clicks):
    # Check which button was clicked
    ctx = dash.callback_context
    button_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    # Set default values for button clicks  (without no page will load!!!)
    main_clicks = main_clicks or 0
    secondary_clicks = secondary_clicks or 0
    third_clicks = third_clicks or 0
    other_clicks = other_clicks or 0


    # Return the appropriate layout based on the button clicked
    if pathname == '/secondary' or button_id == 'stockspage':
        return second_page_layout
    elif button_id == 'mainpage':
        return main_page_layout
    elif button_id == 'historicalpage':
        return third_page_layout
    elif button_id == 'descriptionpage':
        return last_page_layout
    else:
        return main_page_layout  # Default to main page layout

#callbacks Page1
@app.callback(
    [
     Output('line-fig', 'figure'),
     Output('pnl', 'children'),
     Output('sharp', 'children'),
     Output('maxdd', 'children'),
     Output('winrate', 'children'),
     Output('profitrate', 'children'),
     Output('sharpe', 'figure'),
     Output('to', 'figure'),
     Output('max_to', 'figure'),
     Output('drawdown', 'figure'),
     Output('win', 'figure'),
     Output('pr', 'figure')
     ], # Output component ID: 'graph', property: 'figure'
    [
     Input('date-range-picker', 'start_date'),  # Input component ID: 'date-range-picker', property: 'start_date'
     Input('date-range-picker', 'end_date'), # Input component ID: 'date-range-picker', property: 'end_date'    
     Input('mtd', 'n_clicks'),
     Input('qtd', 'n_clicks'),
     Input('ytd', 'n_clicks')
     ],
    )

def update_graph(start_date, end_date, mtd, qtd, ytd):
    ctx = dash.callback_context
    button_id = None
    start_date = df.index[0]
    if ctx.triggered:
   #     return dash.no_update
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    # Determine date range based on button click
    if button_id == 'mtd':
        start_date = pd.Timestamp.now().to_period('M').start_time
    elif button_id == 'qtd':
        start_date =  pd.Timestamp.now().to_period('Q').start_time
    elif button_id == 'ytd':
        start_date = pd.Timestamp.now().to_period('Y').start_time
    
    # Function to update graphs based on the selected date range  
    dfc = df[(df.index >= start_date) & (df.index <= end_date)]
    dfD = dfc.resample('D').agg({'pnl_u':'sum', 'pnl_c':'sum', 'pnl_cl':'sum', 'betsize_u':'sum', 'betsize_c':'sum', 'betsize_cl':'sum'})
    dfD = dfD[dfD.pnl_u != 0]  
 
    pnlcols = ['pnl_u', 'pnl_c', 'pnl_cl']
    sizecols = ['betsize_u', 'betsize_c', 'betsize_cl']

    ir = 0.048  #current IBKR USD rate
    dfD[pnlcols] = dfD[pnlcols] + ir / 252     
    
    # dataprep line graph  
    cumdf = dfD[pnlcols].cumsum()
    
    # dataprep cards
    pnl = round(cumdf.pnl_cl.iloc[-1], 4)
    pnl = "{:.2%}".format(pnl)
    print('1')
    # # dataprep for top15 table
    dftop15 = dfc.groupby('ticker').pnl_c.sum().sort_values(ascending=False).head(15)
    dftop15 = dftop15.reset_index()
    dftop15 = dftop15.rename(columns={'ticker':'Top15', 'pnl_c':'Top_PnL'})
    
    dfworst15 = dfc.groupby('ticker').pnl_c.sum().sort_values(ascending=True).head(15)
    dfworst15 = dfworst15.reset_index()
    dfworst15 = dfworst15.rename(columns={'ticker':'Worst15', 'pnl_c':'Worst_PnL'})
    print('1')

    dft = pd.concat([dftop15, dfworst15], axis=1)
    table = dft.to_dict('records')
    print(table)
 #   table = []

    # dataprep Sharpe ratio barchart
    sharplist = []
    for col in pnlcols:
        sharplist.append(Sharpe(dfD[col]))
    print(sharplist)
    dfsharp = pd.DataFrame([sharplist], columns=['s_u', 's_c', 's_cl'])
    sharp = round(sharplist[2], 2)
    # dataprep F1 score
    
    # dataprep average daily Turnover
    tolist = []
    for col in sizecols:
        tolist.append(TurnOver(dfD[col]))
    dfto = pd.DataFrame([tolist], columns=['to_u', 'to_c', 'to_cl'])

    # dataprep average daily Turnover
    mtolist = []
    for col in sizecols:
        mtolist.append((dfD[col].max()))
    dfmto = pd.DataFrame([mtolist], columns=['mt_u', 'mt_c', 'mt_cl'])
    
    # dataprep drawdown
    ddlist = []
    for col in pnlcols:
        ddlist.append(DrawDown(dfD[col]))
    print(ddlist)
    dfdd = pd.DataFrame([ddlist], columns=['dd_u', 'dd_c', 'dd_cl'])
    maxdd = round(ddlist[2], 4)
    maxdd = "{:.2%}".format(maxdd)
    
    # dataprep winrate
    winlist = []
    for col in pnlcols:
        winlist.append(WinRate(dfc[col]))
    dfwin = pd.DataFrame([winlist], columns=['wr_u', 'wr_c', 'wr_cl'])
    winrate = winlist[2]
    winrate = "{:.2%}".format(winrate)
    
    #dataprep profitratio
    profitlist = []
    for col in pnlcols:
        profitlist.append(ProfitRatio(dfc[col]))
    dfprofit = pd.DataFrame([profitlist], columns=['pr_u', 'pr_c', 'pr_cl'])
    profitrate = round(profitlist[2], 2)

    
    #create LineGraph for Returns 
    
    figln = px.line(cumdf, x=dfD.index, y=['pnl_u','pnl_c','pnl_cl'], title='<b>Performance 2024</b>',
                    color_discrete_sequence = colors['palet'])
    figln.update_layout(
                        plot_bgcolor=colors['bg_figs'],
                        paper_bgcolor = colors['surround_figs'],
                        font_color = colors['text'],
                        font_family = colors['font'],
                        margin = {'l':20, 'r':40, 't':70, 'b':0, 'pad':0},
                        title = {'x':0.5, 'y':0.95, 'font':{'size':20}},
                        xaxis = {'title':''},
                        yaxis = {'title':'', 'tickformat': '.2%'},
                        legend = {'title': '', 'orientation':'h', 'y':1.14, 'xanchor':'right', 'x':0.9}
                        ) 
    figln.for_each_trace(lambda t: t.update(name=legend_labels[t.name]))
    print(cumdf)
    print(start_date, end_date)

    
    # create BarChart for Sharpe Ratio    
    figsharp = px.bar(dfsharp, x=dfsharp.index, y=['s_u', 's_c', 's_cl'], title='<b>Sharpe Ratio</b>',
                   color_discrete_sequence = colors['palet']
                     )
    figsharp.update_layout(
                        plot_bgcolor=colors['bg_figs'],
                        paper_bgcolor = colors['surround_figs'],
                        font_color = colors['text'],
                        font_family = colors['font'],
                        barmode = 'group',
                        bargap = 0.3,
                        margin = {'l':10, 'r':30, 't':40, 'b':0, 'pad':0},
                        title = {'font':{'size':14} },
                        xaxis = {'title':''},
                        yaxis = {'title':''},
                        showlegend = False
                        )


    figto = px.bar(dfto, x=dfto.index, y=['to_u', 'to_c', 'to_cl'], title='<b>Avg Daily\nTurnover</b>',
                   color_discrete_sequence = colors['palet']
                     )
    figto.update_layout(
                        plot_bgcolor=colors['bg_figs'],
                        paper_bgcolor = colors['surround_figs'],
                        font_color = colors['text'],
                        font_family = colors['font'],
                        barmode = 'group',
                        margin = {'l':10, 'r':30, 't':40, 'b':0, 'pad':0},
                        title = {'font':{'size':14} },
                        xaxis = {'title':''},
                        yaxis = {'title':''},
                        showlegend = False
                        )

    figmto = px.bar(dfmto, x=dfmto.index, y=['mt_u', 'mt_c', 'mt_cl'], title='<b>Max Daily\nTurnover</b>',
                   color_discrete_sequence = colors['palet']
                     )
    figmto.update_layout(
                        plot_bgcolor=colors['bg_figs'],
                        paper_bgcolor = colors['surround_figs'],
                        font_color = colors['text'],
                        font_family = colors['font'],
                        barmode = 'group',
                        margin = {'l':10, 'r':30, 't':40, 'b':0, 'pad':0},
                        title = {'font':{'size':14} },
                        xaxis = {'title':''},
                        yaxis = {'title':''},
                        showlegend = False
                        )

    figdd = px.bar(dfdd, x=dfdd.index, y=['dd_u', 'dd_c', 'dd_cl'], title='<b>DrawDown %</b>',
                   color_discrete_sequence = colors['palet']
                     )
    figdd.update_layout(
                        plot_bgcolor=colors['bg_figs'],
                        paper_bgcolor = colors['surround_figs'],
                        font_color = colors['text'],
                        font_family = colors['font'],
                        barmode = 'group',
                        margin = {'l':10, 'r':30, 't':40, 'b':0, 'pad':0},
                        title = {'font':{'size':14} },
                        xaxis = {'title':''},
                        yaxis = {'title':'', 'tickformat': ',.1%'},
                        showlegend = False
                        )
    
    figwin = px.bar(dfwin, x=dfwin.index, y=['wr_u', 'wr_c', 'wr_cl'], title='<b>Winrate</b>',
                   color_discrete_sequence = colors['palet']
                     )
    figwin.update_layout(
                        plot_bgcolor=colors['bg_figs'],
                        paper_bgcolor = colors['surround_figs'],
                        font_color = colors['text'],
                        font_family = colors['font'],
                        barmode = 'group',
                        margin = {'l':10, 'r':30, 't':40, 'b':0, 'pad':0},
                        title = {'font':{'size':14} },
                        xaxis = {'title':''},
                        yaxis = {'title':''},
                        showlegend = False
                        )
    
    figprofit = px.bar(dfprofit, x=dfprofit.index, y=['pr_u', 'pr_c', 'pr_cl'], title='<b>ProfitRatio</b>',
                   color_discrete_sequence = colors['palet']
                     )
    figprofit.update_layout(
                        plot_bgcolor=colors['bg_figs'],
                        paper_bgcolor = colors['surround_figs'],
                        font_color = colors['text'],
                        font_family = colors['font'],
                        barmode = 'group',
                        bargap = 0.2,
                        margin = {'l':10, 'r':30, 't':40, 'b':0, 'pad':0},
                        title = {'font':{'size':14} },
                        xaxis = {'title':''},
                        yaxis = {'title':''},
                        showlegend = False
                        )
    
    return figln, pnl, sharp, maxdd, winrate, profitrate, figsharp, figto, figmto, figdd, figwin, figprofit

#CallBacks page 2

# @app.callback(
#     Output('bar-fig', 'figure'),
#     Input('my_dpdn', 'value')
# )

# def update_bar(stock_slct):
#     figln = generate_individual_stock_graph(stock_slct, df.index[0])
#     return figln

@app.callback(
    [
     Output('bar-fig', 'figure'),
     Output('table', 'data')
     ],
    [
     Input('mtd', 'n_clicks'),
     Input('qtd', 'n_clicks'),
     Input('ytd', 'n_clicks'),
     Input('my_dpdn', 'value'),
     ],
    )

def update_stockspage(mtd, qtd, ytd, selected_stock):
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
    
    figln = generate_individual_stock_graph(selected_stock, start_date)
    
    # # dataprep for top15 table
    dfc = df[df.index >= start_date]
    dftop15 = dfc.groupby('ticker').pnl_cl.sum().sort_values(ascending=False).head(15)
    dftop15 = dftop15.reset_index()
    dftop15 = dftop15.rename(columns={'ticker':'Top15', 'pnl_cl':'Top_PnL'})
    
    dfworst15 = dfc.groupby('ticker').pnl_cl.sum().sort_values(ascending=True).head(15)
    dfworst15 = dfworst15.reset_index()
    dfworst15 = dfworst15.rename(columns={'ticker':'Worst15', 'pnl_cl':'Worst_PnL'})
    
    dft = round(pd.concat([dftop15, dfworst15], axis=1), 4)

    return figln, dft.to_dict('records')

#callbacks Page3
@app.callback(
    [
     Output('hist_fig', 'figure'),
     Output('pnlh', 'children'),
     Output('sharph', 'children'),
     Output('maxddh', 'children'),
     Output('winrateh', 'children'),
     Output('profitrateh', 'children'),
     Output('sharpeh', 'figure'),
     Output('toh', 'figure'),
     Output('max_toh', 'figure'),
     Output('drawdownh', 'figure'),
     Output('winh', 'figure'),
     Output('prh', 'figure')
     ], 
    [
     Input('date-range-picker', 'start_date'), 
     Input('date-range-picker', 'end_date'),  
     Input('y2018', 'n_clicks'),
     Input('y2019', 'n_clicks'),
     Input('y2020', 'n_clicks'),
     Input('y2021', 'n_clicks'),
     Input('y2022', 'n_clicks'),
     Input('y2023', 'n_clicks'),
     Input('yall', 'n_clicks')
     ],
    )

def update_page3(start_date, end_date, y2018, y2019, y2020, y2021, y2022, y2023, yall):
    ctx = dash.callback_context
    button_id = None
    if ctx.triggered:
   #     return dash.no_update
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    # Determine date range based on button click
    if button_id == 'y2018':
        start_date = '01-01-2018'
        end_date = '31-12-2018'
    elif button_id == 'y2019':
        start_date = '01-01-2019'
        end_date = '31-12-2019'
    elif button_id == 'y2020':
        start_date = '01-01-2020'
        end_date = '31-12-2020'
    elif button_id == 'y2021':
        start_date = '01-01-2021'
        end_date = '31-12-2021'
    elif button_id == 'y2022':
        start_date = '01-01-2022'
        end_date = '31-12-2022'
    elif button_id == 'y2023':
        start_date = '01-01-2023'
        end_date = '31-12-2023'
    elif button_id == 'yall':
        start_date = dfhist.index[0]
        end_date = dfhist.index[-1]
    
    # Function to update graphs based on the selected date range  
    dfc = dfhist[(dfhist.index >= start_date) & (dfhist.index <= end_date)]
    dfD = dfc.resample('D').agg({'pnl_u':'sum', 'pnl_c':'sum', 'pnl_cl':'sum', 'betsize_u':'sum', 'betsize_c':'sum', 'betsize_cl':'sum'})
    dfD = dfD[dfD.pnl_u != 0] 
    
    pnlcols = ['pnl_u', 'pnl_c', 'pnl_cl']
    sizecols = ['betsize_u', 'betsize_c', 'betsize_cl']
    # dataprep line graph  
    ir = 0.03  #current IBKR USD rate
    dfD[pnlcols] = dfD[pnlcols] + ir / 252     

    cumdf = dfD[pnlcols].cumsum()
    
    # dataprep cards
    pnl = round(cumdf.pnl_cl.iloc[-1], 4)
    pnl = "{:.2%}".format(pnl)
    # # dataprep for top15 table
    dftop15 = dfc.groupby('ticker').pnl_c.sum().sort_values(ascending=False).head(15)
    dftop15 = dftop15.reset_index()
    dftop15 = dftop15.rename(columns={'ticker':'Top15', 'pnl_c':'Top_PnL'})
    
    dfworst15 = dfc.groupby('ticker').pnl_c.sum().sort_values(ascending=True).head(15)
    dfworst15 = dfworst15.reset_index()
    dfworst15 = dfworst15.rename(columns={'ticker':'Worst15', 'pnl_c':'Worst_PnL'})

    dft = pd.concat([dftop15, dfworst15], axis=1)
    table = dft.to_dict('records')


    # dataprep Sharpe ratio barchart
    sharplist = []
    for col in pnlcols:
        sharplist.append(Sharpe(dfD[col]))
    print(sharplist)
    dfsharp = pd.DataFrame([sharplist], columns=['s_u', 's_c', 's_cl'])
    sharp = round(sharplist[2], 2)
    # dataprep F1 score
    
    # dataprep average daily Turnover
    tolist = []
    for col in sizecols:
        tolist.append(TurnOver(dfD[col]))
    dfto = pd.DataFrame([tolist], columns=['to_u', 'to_c', 'to_cl'])

    # dataprep average daily Turnover
    mtolist = []
    for col in sizecols:
        mtolist.append((dfD[col].max()))
    dfmto = pd.DataFrame([mtolist], columns=['mt_u', 'mt_c', 'mt_cl'])
    
    # dataprep drawdown
    ddlist = []
    for col in pnlcols:
        ddlist.append(DrawDown(dfD[col]))
    print(ddlist)
    dfdd = pd.DataFrame([ddlist], columns=['dd_u', 'dd_c', 'dd_cl'])
    maxdd = round(ddlist[2], 4)
    maxdd = "{:.2%}".format(maxdd)
    
    # dataprep winrate
    winlist = []
    for col in pnlcols:
        winlist.append(WinRate(dfc[col]))
    dfwin = pd.DataFrame([winlist], columns=['wr_u', 'wr_c', 'wr_cl'])
    winrate = winlist[2]
    winrate = "{:.2%}".format(winrate)
    
    #dataprep profitratio
    profitlist = []
    for col in pnlcols:
        profitlist.append(ProfitRatio(dfc[col]))
    dfprofit = pd.DataFrame([profitlist], columns=['pr_u', 'pr_c', 'pr_cl'])
    profitrate = round(profitlist[2], 2)

    
    #create LineGraph for Returns    
    figln = px.line(cumdf, x=dfD.index, y=['pnl_u','pnl_c','pnl_cl'], title='<b>Performance</b>',
                    color_discrete_sequence = colors['palet'])
    figln.update_layout(
                        plot_bgcolor=colors['bg_figs'],
                        paper_bgcolor = colors['surround_figs'],
                        font_color = colors['text'],
                        font_family = colors['font'],
                        margin = {'l':20, 'r':40, 't':70, 'b':0, 'pad':0},
                        title = {'x':0.5, 'y':0.95, 'font':{'size':20}},
                        xaxis = {'title':''},
                        yaxis = {'title':'', 'tickformat': '.2%'},
                        legend = {'title': '', 'orientation':'h', 'y':1.14, 'xanchor':'right', 'x':0.9}
                        ) 
    figln.for_each_trace(lambda t: t.update(name=legend_labels[t.name]))

    print(cumdf)
    print(start_date, end_date)

    
    # create BarChart for Sharpe Ratio    
    figsharp = px.bar(dfsharp, x=dfsharp.index, y=['s_u', 's_c', 's_cl'], title='<b>Sharpe Ratio</b>',
                   color_discrete_sequence = colors['palet']
                     )
    figsharp.update_layout(
                        plot_bgcolor=colors['bg_figs'],
                        paper_bgcolor = colors['surround_figs'],
                        font_color = colors['text'],
                        font_family = colors['font'],
                        barmode = 'group',
                        bargap = 0.3,
                        margin = {'l':10, 'r':30, 't':40, 'b':0, 'pad':0},
                        title = {'font':{'size':14} },
                        xaxis = {'title':''},
                        yaxis = {'title':''},
                        showlegend = False
                        )


    figto = px.bar(dfto, x=dfto.index, y=['to_u', 'to_c', 'to_cl'], title='<b>Avg Daily\nTurnover</b>',
                   color_discrete_sequence = colors['palet']
                     )
    figto.update_layout(
                        plot_bgcolor=colors['bg_figs'],
                        paper_bgcolor = colors['surround_figs'],
                        font_color = colors['text'],
                        font_family = colors['font'],
                        barmode = 'group',
                        margin = {'l':10, 'r':30, 't':40, 'b':0, 'pad':0},
                        title = {'font':{'size':14} },
                        xaxis = {'title':''},
                        yaxis = {'title':''},
                        showlegend = False
                        )

    figmto = px.bar(dfmto, x=dfmto.index, y=['mt_u', 'mt_c', 'mt_cl'], title='<b>Max Daily\nTurnover</b>',
                   color_discrete_sequence = colors['palet']
                     )
    figmto.update_layout(
                        plot_bgcolor=colors['bg_figs'],
                        paper_bgcolor = colors['surround_figs'],
                        font_color = colors['text'],
                        font_family = colors['font'],
                        barmode = 'group',
                        margin = {'l':10, 'r':30, 't':40, 'b':0, 'pad':0},
                        title = {'font':{'size':14} },
                        xaxis = {'title':''},
                        yaxis = {'title':''},
                        showlegend = False
                        )

    figdd = px.bar(dfdd, x=dfdd.index, y=['dd_u', 'dd_c', 'dd_cl'], title='<b>DrawDown %</b>',
                   color_discrete_sequence = colors['palet']
                     )
    figdd.update_layout(
                        plot_bgcolor=colors['bg_figs'],
                        paper_bgcolor = colors['surround_figs'],
                        font_color = colors['text'],
                        font_family = colors['font'],
                        barmode = 'group',
                        margin = {'l':10, 'r':30, 't':40, 'b':0, 'pad':0},
                        title = {'font':{'size':14} },
                        xaxis = {'title':''},
                        yaxis = {'title':'', 'tickformat': ',.1%'},
                        showlegend = False
                        )
    
    figwin = px.bar(dfwin, x=dfwin.index, y=['wr_u', 'wr_c', 'wr_cl'], title='<b>Winrate</b>',
                   color_discrete_sequence = colors['palet']
                     )
    figwin.update_layout(
                        plot_bgcolor=colors['bg_figs'],
                        paper_bgcolor = colors['surround_figs'],
                        font_color = colors['text'],
                        font_family = colors['font'],
                        barmode = 'group',
                        margin = {'l':10, 'r':30, 't':40, 'b':0, 'pad':0},
                        title = {'font':{'size':14} },
                        xaxis = {'title':''},
                        yaxis = {'title':''},
                        showlegend = False
                        )
    
    figprofit = px.bar(dfprofit, x=dfprofit.index, y=['pr_u', 'pr_c', 'pr_cl'], title='<b>ProfitRatio</b>',
                   color_discrete_sequence = colors['palet']
                     )
    figprofit.update_layout(
                        plot_bgcolor=colors['bg_figs'],
                        paper_bgcolor = colors['surround_figs'],
                        font_color = colors['text'],
                        font_family = colors['font'],
                        barmode = 'group',
                        bargap = 0.2,
                        margin = {'l':10, 'r':30, 't':40, 'b':0, 'pad':0},
                        title = {'font':{'size':14} },
                        xaxis = {'title':''},
                        yaxis = {'title':''},
                        showlegend = False
                        )
    
    return figln, pnl, sharp, maxdd, winrate, profitrate, figsharp, figto, figmto, figdd, figwin, figprofit


if __name__=='__main__':
    app.run_server(port=8030)

