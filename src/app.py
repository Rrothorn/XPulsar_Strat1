# -*- coding: utf-8 -*-
"""
Created on Sun May  5 11:59:12 2024

@author: Gebruiker
"""

import dash
from dash import Dash, html, dcc
from config import colors_config
import dash_bootstrap_components as dbc

app = Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.SLATE], suppress_callback_exceptions=True)

navbar_style = {
    'position': 'fixed',
    'top': 0,
    'left': 0,
    'bottom': 0,
#    'width': '16rem',
    }
# define navigation bar
navbar = html.Div([
    dbc.Nav([
        dbc.NavItem(dbc.NavLink('Main', href='/', active=True)),
        dbc.NavItem(dbc.NavLink('Historical', href='/historical', active=True)),
        dbc.NavItem(dbc.NavLink('Stocks', href='/stocks', active=True)),
        dbc.NavItem(dbc.NavLink('Description', href='/description', active=True)),
        ],
        pills=False,
        ),
    html.H1('Tracking a proxy Russell 1000 Portfolio',
            className='text-center text-primary, mb-2',
            style={'font-size':'28px',
                   'color':colors_config['colors']['palet'][4],
                   'margin_bottom': '0px'
                   }),
    html.H2('An AI-powered Daily Machine Learning algorithm',
            className='text-center text-primary, mb-4',
            style={'font-size':'18px', 'color':colors_config['colors']['palet'][4]}),
    ],
#    style=navbar_style,
    )


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

app.layout = html.Div([
    navbar,
    dash.page_container,
    footer
])

if __name__ == '__main__':
    app.run(debug=True, port=8031)