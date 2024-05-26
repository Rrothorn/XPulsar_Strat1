# -*- coding: utf-8 -*-
"""
Created on Sun May  5 11:59:12 2024

@author: Gebruiker
"""

# =============================================================================
# This APP is a DASHBOARD to present the performance and performance metrics of a Machine Learning strategy on a portfolio of US stocks.
# The dashboard is multipage and its content has 5 pages
#  1: the 2024 performance
#  2: the 2018-2023 performance
#  3: the individual stock contributions to the 2024 performance
#  4: a description page summarising the trading strategy
#  5: a page where you can download daily reports
# Technically the dashboard is created as:
#     |-- SRC /  app.py
#                config.py
#                plots_generator.py
#                metrics_generator.py
#                report_creator.py
#                -- PAGES      /   home.py
#                                  historical.py
#                                  stock.py
#                                  description.py
#                                  report.py
#     |-- DATA / DC_2024trades.csv
#                XGB_trades18-23.csv 
#                DC_reports24.csv
#     |-- ASSETS / logo.png
# =============================================================================


# =============================================================================
# This is the app.py central page from which the app is run. 
# It contains the app.layout which is just a navigation bar and footer surrounding the content of a multipage container.  
# 
# =============================================================================


import dash
from dash import Dash, html, dcc
from config import colors_config
import dash_bootstrap_components as dbc
import os

# =============================================================================
#  Initialising the Dash app 
#  setting use_pages = True to ensure a multipage dashboard
#  import external sheets 1) a dash bootstrap theme for fluency of the different components 2) font-awesome for copyright figurine
#  include assets folder to direct to images and CSS files
# 
# =============================================================================
assets_path = os.getcwd() + '/assets'
app = Dash(__name__, use_pages=True, external_stylesheets=
           [dbc.themes.BOOTSTRAP,
            'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css'
            ], suppress_callback_exceptions=True, assets_folder=assets_path)
server = app.server   # required for publishing

######################  NAVIGATION BAR #################################

# Define the navigation bar
# navbar = html.Div([
#     html.Div([
#         html.Img(src = assets_path + '/xpulsar-logo.png', height='120px', style={'float': 'right', 'margin-top': '3px', 'margin-right': '20px'}),
#     ], style={'width': '100%'}),  # Ensures the logo stays on the same line as the navigation links

#     # Navigation links
#     dbc.Nav([
#         dbc.NavItem(dbc.NavLink('Main', href='/', active=True)),
#         dbc.NavItem(dbc.NavLink('Historical', href='/historical', active=True)),
#         dbc.NavItem(dbc.NavLink('Stocks', href='/stocks', active=True)),
#         dbc.NavItem(dbc.NavLink('Description', href='/description', active=True)),
#         dbc.NavItem(dbc.NavLink('Reports', href='/report', active=True)),
#     ], pills=False),


##################### FOOTER  ################################
#define footer
footer = dbc.Container(
    dbc.Row(
        [
            dbc.Col(html.A('XPulsar Capital', href='/'), align='right')
            ]        
        ),
        className='footer',
        fluid=True,
    )

footer =     html.Footer([
        html.Div([
            html.I(className="fa fa-copyright"),  # Font Awesome icon for copyright
            html.Span("2024, XPulsar Capital, All rights reserved")
        ])
    ],)

######################### LAYOUT of APP  and RUN ###############################

# Define the layout
app.layout = dbc.Container([
    dbc.Navbar(
        dbc.Container([
            # Nav items on the left
            dbc.Row([

                dbc.Col([
                    dbc.NavItem(dbc.NavLink("Main", href="/", active=True), style = {'margin-left':'20px'}),
                    dbc.NavItem(dbc.NavLink("Historical", href="/historical", active=True)),
                    dbc.NavItem(dbc.NavLink('Stocks', href='/stocks', active=True)),
                    dbc.NavItem(dbc.NavLink('Description', href='/description', active=True)),
                    dbc.NavItem(dbc.NavLink('Report', href='/report', active=True)),
                ], className= 'hstack gap-3'),
            ], className="g-10"),  # Use className="g-0" to remove gutter spacing between columns
            dbc.Col([
                html.Div([
                    # Heading
                    html.H1('Tracking a proxy Russell 1000 Portfolio',
                            className='text-center text-primary, mb-2',
                            style={'font-size': '28px',
                                    'color': colors_config['colors']['palet'][4],
                                    'margin_bottom': '0px',
                                    'width': '100%'  # Span the entire width
                                    }),
                    html.H2('An AI-powered Daily Machine Learning algorithm',
                            className='text-center text-primary, mb-4',
                            style={'font-size': '18px', 'color': colors_config['colors']['palet'][4], 'width': '100%'}),  # Span the entire width
                    ],),
                ], 
                width = 'auto',
                className="d-flex justify-content-center align-items-center",  # Center horizontally and vertically
                ),            
            
            # Logo on the right
            dbc.Col([
                html.Div([
                    html.Img(src='assets/xpulsar-logo.png', height='100vh', style={'float': 'right', 'margin-top': '3px', 'margin-right': '20px'}),
                ], style={'width': '100%'})
            ], width=2, className="ml-auto"),  # Use className="ml-auto" to push the logo to the right
        ], fluid=True),
        dark=True,          # Use dark theme for the navbar
    ),
    dash.page_container,  # Placeholder for page content
    footer, # Placeholder for footer component
], fluid=True)  # Make the container full-width

if __name__ == '__main__':
    app.run(debug=True, port=8031)