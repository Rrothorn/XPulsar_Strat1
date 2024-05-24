# -*- coding: utf-8 -*-
"""
Created on Mon May  6 23:47:20 2024

@author: Gebruiker
"""

import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
from config import colors_config, card_config

dash.register_page(__name__)

background_img = 'linear-gradient(to left, rgba(39,83,81,0.5), rgba(0,0,0,1))'


layout =  html.Div(style={'background-image': background_img, 'height': '100vh'},
                      children=[
    dbc.Container([
        html.Div(style={'height': '25px'}),
        # Your layout components...
        dbc.Row([
            dbc.Col([
                dbc.Card(
                    dbc.CardBody([
                        html.H4('Description',
                                className='card-title',
                                style={'font-size':'20px',
                                       'font-weight':'bold',
                                       'color':'#FFFFFF',
                                       }
                                ),
                        html.P("The algorithm presented herein represents a sophisticated, data-driven investment fund leveraging advanced AI and Machine Learning algorithms. This fund operates within the realm of single stocks equity portfolios, employing daily selections and reselections to optimise positions. The algorithm strategically engages both long and short positions to capitalise on market opportunities. ", 
                               className='card-text', 
                               style={'color':'#95D7E0'},
                               ),
                        ]),
                    style={
                           'margin-left':'60px',
                           'border-radius': '14px',
                           'border':'4px solid #ddd',
                           'font-size':'15px',
                           }
                    )

                ], width = 5),

            dbc.Col([
                dbc.Card(
                    dbc.CardBody([
                        html.H4('Objective', 
                                className='card-title',
                                style={'font-size':'20px',
                                       'font-weight':'bold',
                                       'color':'#FFFFFF',
                                       }
                                ),
                        html.P("The primary objective of the fund is to achieve compelling and sustainable returns by employing our pattern-detecting algorithm across an extensive array of stocks. Striving for both attractiveness and sustainability involves striking a balance between maximising returns and minimising risk, achieved through meticulous diversification and daily portfolio calibrations.",
                               className='card-text',
                               style={'color':'#95D7E0'},
                               ),
                        ]),
                    style={'margin-left':'60px', 'border-radius': '14px', 'border':'4px solid #ddd', 'font-size':'15px'}
                    )

                ], width = 5),
            ]),
        
        html.Div(style={'height': '25px'}),
        
        dbc.Row([
            dbc.Col([
                dbc.Card(
                    dbc.CardBody([
                        html.H4('Suitability',
                                className='card-title',
                                style={'font-size':'20px',
                                       'font-weight':'bold',
                                       'color':'#FFFFFF',
                                       }
                                ),
                        html.P("This fund is specifically tailored for investors who (i) seek capital growth through exposure to global equity markets, (ii) are comfortable with a certain level of risk, (iii) acknowledge that their capital is at risk, understanding that the value of their investment may fluctuate both upward and downard.",
                               className='card-text', 
                               style={'color': '#95D7E0'}
                               ),
                        ]),
                    style={'margin-left':'60px', 'border-radius': '10px', 'border':'4px solid #ddd', 'font-size':'15px'}
                    ),
                ], width = 5),
            
            dbc.Col([
                dbc.Card(
                    dbc.CardBody([
                        html.H4('Legend Explanation',
                                className='card-title',
                                style={'font-size':'20px',
                                       'font-weight':'bold',
                                       'color':'#000000',
                                       }
                                ),
                        html.P("'All-in': selected stocks trade each day as per algorithmic prediction",
                               className='card-text',
                               style={'color':colors_config['colors']['palet'][0]}
                               ),
                        html.P("'Conditional': selected stock portfolio is divided into smaller clusters. A cluster will become active only when certain daily criteria are checked.",
                               className='card-text',
                               style={'color':colors_config['colors']['palet'][1]}),
                        html.P("'Leveraged': like 'conditional', however when a cluster is inactive more weight is given to active clusters",
                               className='card-text',
                               style={'color':colors_config['colors']['palet'][2]})
                        ]),
                    style={
                           'margin-left':'60px',
                           'border-radius': '10px',
                           'border':'4px solid #ddd',
                           'font-size':'15px',
                           'font-weight': 'bold',
                           'background-color': '#FFFFFF'
                           }
                    )

                ], width = 5)
        ]),
    ])
])
