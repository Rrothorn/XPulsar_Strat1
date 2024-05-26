# -*- coding: utf-8 -*-
"""
Created on Sun May  5 13:08:31 2024

@author: Gebruiker
"""
import pandas as pd
import numpy as np
import datetime

from config import colors_config

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

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

# donwloading data
fname = 'DC_2024trades.csv'
df = pd.read_csv(f'../{fname}', parse_dates = ['date'], index_col = 'date')

fname2 = 'XGB_trades18-23.csv'
dfhist = pd.read_csv(f'../{fname2}', parse_dates = ['date'], index_col = 'date')
dfhist = dfhist.rename(columns = {'weighted_pnl':'pnl_u', 'betsize':'betsize_u'})


legend_labels = {'pnl_u': 'All-in', 'pnl_c': 'Conditional', 'pnl_cl': 'Leveraged'}


def generate_perf_plot1(start_date, end_date, figln_title, target):
    # Function to update graphs based on the selected date range  
    dfc = df[(df.index >= start_date) & (df.index <= end_date)]
    dfD = dfc.resample('D').agg({'pnl_u':'sum', 'pnl_c':'sum', 'pnl_cl':'sum', 'betsize_u':'sum', 'betsize_c':'sum', 'betsize_cl':'sum'})
    dfD = dfD[dfD.pnl_u != 0]  
 
    pnlcols = ['pnl_u', 'pnl_c', 'pnl_cl']

    ir = 0.048  #current IBKR USD rate
    dfD[pnlcols] = dfD[pnlcols] + ir / 252     
    
    # dataprep line graph  
    cumdf = dfD[pnlcols].cumsum()

    # add a dummy initial row with 0s
    dummy_date = cumdf.index[0] - pd.Timedelta(days=1)
    cumdf.loc[dummy_date] = [0] * len(cumdf.columns)
    cumdf = cumdf.sort_index()
    
    # current value for actual vs target
    current_pnl = dfD.pnl_cl.sum()

    #create LineGraph for Returns 
    
    figln = px.line(cumdf, x=cumdf.index, y=['pnl_u','pnl_c','pnl_cl'], title=f'<b>{figln_title} Performance 2024</b>',
                    color_discrete_sequence = colors_config['colors']['palet'])
    figln.update_layout(
                        plot_bgcolor='#FFFFFF',
                        paper_bgcolor = colors_config['colors']['surround_figs'],
                        font_color = colors_config['colors']['text'],
                        font_family = colors_config['colors']['font'],
                        margin = {'l':20, 'r':40, 't':70, 'b':0, 'pad':0},
                        title = {'x':0.5, 'y':0.95, 'font':{'size':20}},
                        xaxis = {'title':'', 'gridcolor':'#808080'},
                        yaxis = {'title':'', 'tickformat': '.2%', 'gridcolor':'#808080'},
                        legend = {'title': '', 'orientation':'h', 'y':0.99, 'xanchor':'left', 'x':0.03}
                        )

    
    figln.for_each_trace(lambda t: t.update(name=legend_labels[t.name]))
    
    fig_target = go.Figure(go.Indicator(
        domain = {'x': [0, 1], 'y': [0, 1]},
        value = current_pnl,
        number={'valueformat': '.2%'},
        mode = "gauge+number+delta",
        title = {'text': f'{figln_title} Actual vs Target'},
        delta = {'reference':  target, 'valueformat': '.2%'},
        gauge = {'axis': {'range': [None, 1.3 * target], 'tickformat':',.2%'},
                 'bar': {'color': '#178474'},                 
                 'steps' : [{'range': [0, 1.3 * target], 'color': '#FFFFFF'},],
                 'threshold' : {'line': {'color': colors_config['colors']['palet'][2], 'width': 4}, 'thickness': 0.75, 'value': target}},
        ))

    fig_target.update_layout(
                        plot_bgcolor=colors_config['colors']['bg_figs'],
                        paper_bgcolor = colors_config['colors']['surround_figs'],
                        font_color = colors_config['colors']['text'],
                        font_family = colors_config['colors']['font'],
                        margin = {'l':20, 'r':40, 't':50, 'b':10, 'pad':10},
                        title = {'x':0.5, 'y':0.95, 'font':{'size':16}},
                        )

    
    return figln, fig_target

def generate_target_plot(target, target_title):
    
    fig_target = go.Figure(go.Indicator(
        domain = {'x': [0, 1], 'y': [0, 1]},
        value = 0.05,
        mode = "gauge+number+delta",
        title = {'text': "Actual vs Target"},
        delta = {'reference': 380},
        gauge = {'axis': {'range': [None, 0.05]},
                 'bar': {'color': '#178474'},                 
                 'steps' : [
                     {'range': [0, 250], 'color': "lightgray"},
                     {'range': [250, 400], 'color': "gray"}],
                 'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': target}}))
    return fig_target


def generate_month_bars():
    # Function to update graphs based on the selected date range  
    dfM = df.resample('M').agg({'pnl_u':'sum', 'pnl_c':'sum', 'pnl_cl':'sum', 'betsize_u':'sum', 'betsize_c':'sum', 'betsize_cl':'sum'})

    print(dfM)
    
    bar_months = px.bar(dfM, x=dfM.index.strftime('%b'), y=['pnl_u', 'pnl_c', 'pnl_cl'], title='<b>Monthly Performance 2024</b>',
                    color_discrete_sequence = colors_config['colors']['palet']
                     )
    bar_months.update_layout(
                        plot_bgcolor=colors_config['colors']['bg_figs'],
                        paper_bgcolor = colors_config['colors']['surround_figs'],
                        font_color = colors_config['colors']['text'],
                        font_family = colors_config['colors']['font'],
                        barmode = 'group',
                        bargap = 0.3,
                        margin = {'l':10, 'r':30, 't':40, 'b':0, 'pad':0},
                        title = {'x':0.5, 'y':0.95, 'font':{'size':16}},
                        xaxis = {'title':'', 'gridcolor':'#808080'},
                        yaxis = {'title':'', 'tickformat': '.2%', 'gridcolor':'#808080'},
                        showlegend = False
                        )    
    
    return bar_months

def generate_annual_bars():
    # Function to update graphs based on the selected date range  
    dfY = dfhist.resample('Y').agg({'pnl_u':'sum', 'pnl_c':'sum', 'pnl_cl':'sum', 'betsize_u':'sum', 'betsize_c':'sum', 'betsize_cl':'sum'})
    print(dfY)
    bar_annual = px.bar(dfY, x=dfY.index.strftime('%y'), y=['pnl_u', 'pnl_c', 'pnl_cl'], title='<b>Annual Performance 2018-2023</b>',
                    color_discrete_sequence = colors_config['colors']['palet']
                     )
    bar_annual.update_layout(
                        plot_bgcolor=colors_config['colors']['bg_figs'],
                        paper_bgcolor = colors_config['colors']['surround_figs'],
                        font_color = colors_config['colors']['text'],
                        font_family = colors_config['colors']['font'],
                        barmode = 'group',
                        bargap = 0.3,
                        margin = {'l':10, 'r':30, 't':40, 'b':0, 'pad':0},
                        title = {'x':0.5, 'y':0.95, 'font':{'size':16}},
                        xaxis = {'title':'', 'gridcolor':'#808080'},
                        yaxis = {'title':'', 'tickformat': '.2%', 'gridcolor':'#808080'},
                        showlegend = False
                        )    
    
    return bar_annual

def generate_metrics_bars(start_date, end_date, bar_title):
    # Function to update graphs based on the selected date range  
    dfc = df[(df.index >= start_date) & (df.index <= end_date)]
    
    # Set dataframe on a daily timeframe
    dfD = dfc.resample('D').agg({'pnl_u':'sum', 'pnl_c':'sum', 'pnl_cl':'sum', 'betsize_u':'sum', 'betsize_c':'sum', 'betsize_cl':'sum'})
    dfD = dfD[dfD.pnl_u != 0]  
    
    dfM = dfc.resample('M').agg({'pnl_u':'sum', 'pnl_c':'sum', 'pnl_cl':'sum', 'betsize_u':'sum', 'betsize_c':'sum', 'betsize_cl':'sum'})
 
    pnlcols = ['pnl_u', 'pnl_c', 'pnl_cl']
    sizecols = ['betsize_u', 'betsize_c', 'betsize_cl']

    ir = 0.048  #current IBKR USD rate
    dfD[pnlcols] = dfD[pnlcols] + ir / 252     
    dfM[pnlcols] = dfM[pnlcols] + ir / 12     
    
    # calculate the Sharpe Ratio
    sharplist = []
    for col in pnlcols:
        sharplist.append(Sharpe(dfD[col]))

    dfsharp = pd.DataFrame([sharplist], columns=['s_u', 's_c', 's_cl'])
    
    # dataprep drawdown
    ddlist = []
    for col in pnlcols:
        ddlist.append(DrawDown(dfD[col]))
    print(ddlist)
    dfdd = pd.DataFrame([ddlist], columns=['dd_u', 'dd_c', 'dd_cl'])

    # dataprep winrate
    winlist = []
    for col in pnlcols:
        winlist.append(WinRate(dfc[col]))
    dfwin = pd.DataFrame([winlist], columns=['wr_u', 'wr_c', 'wr_cl'])
    
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
    
    #dataprep profitratio
    profitlist = []
    for col in pnlcols:
        profitlist.append(ProfitRatio(dfc[col]))
    dfprofit = pd.DataFrame([profitlist], columns=['pr_u', 'pr_c', 'pr_cl'])
    
    # dataprep the winning days vs the losing days
    windlist = []
    for col in pnlcols:
        windlist.append(WinRate(dfD[col]))
    dfwind = pd.DataFrame([windlist], columns=['wdr_u', 'wdr_c', 'wdr_cl'])

    # dataprep the winning months vs the losing months
    winmlist = []
    for col in pnlcols:
        winmlist.append(WinRate(dfM[col]))
    dfwinm = pd.DataFrame([winmlist], columns=['wmr_u', 'wmr_c', 'wmr_cl'])
    
    # create BarChart for Sharpe Ratio    
    bar_sharp = px.bar(dfsharp, x=dfsharp.index, y=['s_u', 's_c', 's_cl'], title=f'<b>{bar_title} Sharpe Ratio</b>',
                    color_discrete_sequence = colors_config['colors']['palet']
                     )
    bar_sharp.update_layout(
                        plot_bgcolor=colors_config['colors']['bg_figs'],
                        paper_bgcolor = colors_config['colors']['surround_figs'],
                        font_color = colors_config['colors']['text'],
                        font_family = colors_config['colors']['font'],
                        barmode = 'group',
                        bargap = 0.3,
                        margin = {'l':10, 'r':30, 't':40, 'b':0, 'pad':0},
                        title = {'font':{'size':14} },
                        xaxis = {'title':''},
                        yaxis = {'title':''},
                        showlegend = False
                        )
    
    bar_dd = px.bar(dfdd, x=dfdd.index, y=['dd_u', 'dd_c', 'dd_cl'], title=f'<b>{bar_title} DrawDown %</b>',
                    color_discrete_sequence = colors_config['colors']['palet']
                     )
    bar_dd.update_layout(
                        plot_bgcolor=colors_config['colors']['bg_figs'],
                        paper_bgcolor = colors_config['colors']['surround_figs'],
                        font_color = colors_config['colors']['text'],
                        font_family = colors_config['colors']['font'],
                        barmode = 'group',
                        margin = {'l':10, 'r':30, 't':40, 'b':0, 'pad':0},
                        title = {'font':{'size':14} },
                        xaxis = {'title':''},
                        yaxis = {'title':'', 'tickformat': ',.0%'},
                        showlegend = False
                        )

    bar_win = px.bar(dfwin, x=dfwin.index, y=['wr_u', 'wr_c', 'wr_cl'], title=f'<b>{bar_title} Winrate</b>',
                    color_discrete_sequence = colors_config['colors']['palet']
                     )
    bar_win.update_layout(
                        plot_bgcolor=colors_config['colors']['bg_figs'],
                        paper_bgcolor = colors_config['colors']['surround_figs'],
                        font_color = colors_config['colors']['text'],
                        font_family = colors_config['colors']['font'],
                        barmode = 'group',
                        margin = {'l':10, 'r':30, 't':40, 'b':0, 'pad':0},
                        title = {'font':{'size':14} },
                        xaxis = {'title':''},
                        yaxis = {'title':'', 'tickformat':',.0%'},
                        showlegend = False
                        )    
    
    
    bar_to = px.bar(dfto, x=dfto.index, y=['to_u', 'to_c', 'to_cl'], title=f'<b>{bar_title} Avg Daily\nTurnover</b>',
                    color_discrete_sequence = colors_config['colors']['palet']
                     )
    bar_to.update_layout(
                        plot_bgcolor=colors_config['colors']['bg_figs'],
                        paper_bgcolor = colors_config['colors']['surround_figs'],
                        font_color = colors_config['colors']['text'],
                        font_family = colors_config['colors']['font'],
                        barmode = 'group',
                        margin = {'l':10, 'r':30, 't':40, 'b':0, 'pad':0},
                        title = {'font':{'size':14} },
                        xaxis = {'title':''},
                        yaxis = {'title':''},
                        showlegend = False
                        )
    
    bar_mto = px.bar(dfmto, x=dfmto.index, y=['mt_u', 'mt_c', 'mt_cl'], title=f'<b>{bar_title} Max Daily\nTurnover</b>',
                    color_discrete_sequence = colors_config['colors']['palet']
                     )
    bar_mto.update_layout(
                        plot_bgcolor=colors_config['colors']['bg_figs'],
                        paper_bgcolor = colors_config['colors']['surround_figs'],
                        font_color = colors_config['colors']['text'],
                        font_family = colors_config['colors']['font'],
                        barmode = 'group',
                        margin = {'l':10, 'r':30, 't':40, 'b':0, 'pad':0},
                        title = {'font':{'size':14} },
                        xaxis = {'title':''},
                        yaxis = {'title':''},
                        showlegend = False
                        )
    
    bar_pr = px.bar(dfprofit, x=dfprofit.index, y=['pr_u', 'pr_c', 'pr_cl'], title=f'<b>{bar_title} ProfitRatio</b>',
                    color_discrete_sequence = colors_config['colors']['palet']
                     )
    bar_pr.update_layout(
                        plot_bgcolor=colors_config['colors']['bg_figs'],
                        paper_bgcolor = colors_config['colors']['surround_figs'],
                        font_color = colors_config['colors']['text'],
                        font_family = colors_config['colors']['font'],
                        barmode = 'group',
                        bargap = 0.2,
                        margin = {'l':10, 'r':30, 't':40, 'b':0, 'pad':0},
                        title = {'font':{'size':14} },
                        xaxis = {'title':''},
                        yaxis = {'title':''},
                        showlegend = False
                        ) 
    
    bar_wind = px.bar(dfwind, x=dfwind.index, y=['wdr_u', 'wdr_c', 'wdr_cl'], title=f'<b>{bar_title} Win Days</b>',
                    color_discrete_sequence = colors_config['colors']['palet']
                     )
    bar_wind.update_layout(
                        plot_bgcolor=colors_config['colors']['bg_figs'],
                        paper_bgcolor = colors_config['colors']['surround_figs'],
                        font_color = colors_config['colors']['text'],
                        font_family = colors_config['colors']['font'],
                        barmode = 'group',
                        margin = {'l':10, 'r':30, 't':40, 'b':0, 'pad':0},
                        title = {'font':{'size':14} },
                        xaxis = {'title':''},
                        yaxis = {'title':'', 'tickformat':',.0%'},
                        showlegend = False
                        ) 
    bar_winm = px.bar(dfwinm, x=dfwinm.index, y=['wmr_u', 'wmr_c', 'wmr_cl'], title=f'<b>{bar_title} Win Months</b>',
                    color_discrete_sequence = colors_config['colors']['palet']
                     )
    bar_winm.update_layout(
                        plot_bgcolor=colors_config['colors']['bg_figs'],
                        paper_bgcolor = colors_config['colors']['surround_figs'],
                        font_color = colors_config['colors']['text'],
                        font_family = colors_config['colors']['font'],
                        barmode = 'group',
                        margin = {'l':10, 'r':30, 't':40, 'b':0, 'pad':0},
                        title = {'font':{'size':14} },
                        xaxis = {'title':''},
                        yaxis = {'title':'', 'tickformat':',.0%'},
                        showlegend = False
                        ) 

    return bar_sharp, bar_dd, bar_win, bar_to, bar_mto, bar_pr, bar_wind, bar_winm

def generate_multi_barplot(start_date, end_date, bar_title):
    # Function to update graphs based on the selected date range  
    dfc = df[(df.index >= start_date) & (df.index <= end_date)]
    
    # Set dataframe on a daily timeframe
    dfD = dfc.resample('D').agg({'pnl_u':'sum', 'pnl_c':'sum', 'pnl_cl':'sum', 'betsize_u':'sum', 'betsize_c':'sum', 'betsize_cl':'sum'})
    dfD = dfD[dfD.pnl_u != 0]  
    
    dfM = dfc.resample('M').agg({'pnl_u':'sum', 'pnl_c':'sum', 'pnl_cl':'sum', 'betsize_u':'sum', 'betsize_c':'sum', 'betsize_cl':'sum'})
 
    pnlcols = ['pnl_u', 'pnl_c', 'pnl_cl']
    sizecols = ['betsize_u', 'betsize_c', 'betsize_cl']

    ir = 0.048  #current IBKR USD rate
    dfD[pnlcols] = dfD[pnlcols] + ir / 252     
    dfM[pnlcols] = dfM[pnlcols] + ir / 12     
    
    # calculate the Sharpe Ratio
    sharplist = []
    for col in pnlcols:
        sharplist.append(Sharpe(dfD[col]))

    dfsharp = pd.DataFrame([sharplist], columns=['s_u', 's_c', 's_cl'])
    
    # dataprep drawdown
    ddlist = []
    for col in pnlcols:
        ddlist.append(DrawDown(dfD[col]))
    print(ddlist)
    dfdd = pd.DataFrame([ddlist], columns=['dd_u', 'dd_c', 'dd_cl'])

    # dataprep winrate
    winlist = []
    for col in pnlcols:
        winlist.append(WinRate(dfc[col]))
    dfwin = pd.DataFrame([winlist], columns=['wr_u', 'wr_c', 'wr_cl'])
    
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
    
    #dataprep profitratio
    profitlist = []
    for col in pnlcols:
        profitlist.append(ProfitRatio(dfc[col]))
    dfprofit = pd.DataFrame([profitlist], columns=['pr_u', 'pr_c', 'pr_cl'])
    
    # dataprep the winning days vs the losing days
    windlist = []
    for col in pnlcols:
        windlist.append(WinRate(dfD[col]))
    dfwind = pd.DataFrame([windlist], columns=['wdr_u', 'wdr_c', 'wdr_cl'])

    # dataprep the winning months vs the losing months
    winmlist = []
    for col in pnlcols:
        winmlist.append(WinRate(dfM[col]))
    dfwinm = pd.DataFrame([winmlist], columns=['wmr_u', 'wmr_c', 'wmr_cl'])
    
    # create BarChart for Sharpe Ratio    
    bar_sharp = px.bar(dfsharp, x=dfsharp.index, y=['s_u', 's_c', 's_cl'],
                    color_discrete_sequence = colors_config['colors']['palet']
                     )
    
    bar_dd = px.bar(dfdd, x=dfdd.index, y=['dd_u', 'dd_c', 'dd_cl'],
                    color_discrete_sequence = colors_config['colors']['palet']
                     )

    bar_win = px.bar(dfwin, x=dfwin.index, y=['wr_u', 'wr_c', 'wr_cl'],
                    color_discrete_sequence = colors_config['colors']['palet']
                     )    
    
    bar_to = px.bar(dfto, x=dfto.index, y=['to_u', 'to_c', 'to_cl'],
                    color_discrete_sequence = colors_config['colors']['palet']
                     )
    
    bar_mto = px.bar(dfmto, x=dfmto.index, y=['mt_u', 'mt_c', 'mt_cl'],
                    color_discrete_sequence = colors_config['colors']['palet']
                     )
    
    bar_pr = px.bar(dfprofit, x=dfprofit.index, y=['pr_u', 'pr_c', 'pr_cl'],
                    color_discrete_sequence = colors_config['colors']['palet']
                     )
    
    bar_wind = px.bar(dfwind, x=dfwind.index, y=['wdr_u', 'wdr_c', 'wdr_cl'], 
                    color_discrete_sequence = colors_config['colors']['palet']
                     )

    bar_winm = px.bar(dfwinm, x=dfwinm.index, y=['wmr_u', 'wmr_c', 'wmr_cl'], 
                    color_discrete_sequence = colors_config['colors']['palet']
                     )

    # Create a subplot figure
    fig = make_subplots(rows=2, cols=4, subplot_titles=(
        "<b>Sharpe Ratio</b>",
        "<b>DrawDown %</b>",
        "<b>WinRate</b>",
        "<b>Winning Days</b>",
        "<b>Avg Daily Turnover</b>",
        "<b>Max Daily Turnover</b>",
        "<b>ProfitRatio</b>",
        "<b>Winning Months</b>"
    ))
    
    # Add traces to the subplot
    fig.add_trace(bar_sharp.data[0], row=1, col=1)
    fig.add_trace(bar_sharp.data[1], row=1, col=1)
    fig.add_trace(bar_sharp.data[2], row=1, col=1)
    
    fig.add_trace(bar_dd.data[0], row=1, col=2)
    fig.add_trace(bar_dd.data[1], row=1, col=2)
    fig.add_trace(bar_dd.data[2], row=1, col=2)

    fig.add_trace(bar_win.data[0], row=1, col=3)
    fig.add_trace(bar_win.data[1], row=1, col=3)
    fig.add_trace(bar_win.data[2], row=1, col=3)

    fig.add_trace(bar_wind.data[0], row=1, col=4)
    fig.add_trace(bar_wind.data[1], row=1, col=4)
    fig.add_trace(bar_wind.data[2], row=1, col=4)
    
    fig.add_trace(bar_to.data[0], row=2, col=1)
    fig.add_trace(bar_to.data[1], row=2, col=1)
    fig.add_trace(bar_to.data[2], row=2, col=1)
    
    fig.add_trace(bar_mto.data[0], row=2, col=2)
    fig.add_trace(bar_mto.data[1], row=2, col=2)
    fig.add_trace(bar_mto.data[2], row=2, col=2)

    fig.add_trace(bar_pr.data[0], row=2, col=3)
    fig.add_trace(bar_pr.data[1], row=2, col=3)
    fig.add_trace(bar_pr.data[2], row=2, col=3)

    fig.add_trace(bar_winm.data[0], row=2, col=4)
    fig.add_trace(bar_winm.data[1], row=2, col=4)
    fig.add_trace(bar_winm.data[2], row=2, col=4)
    
    # Update layout
    fig.update_layout(showlegend=False)
    fig.update_layout(plot_bgcolor=colors_config['colors']['bg_figs'],
                      paper_bgcolor=colors_config['colors']['surround_figs'],
                      font_color=colors_config['colors']['text'],
                      font_family=colors_config['colors']['font'],
                      margin=dict(l=25, r=20, t=20, b=10)  # Minimize margins
                      )
    # Set tick format for the y-axis of the bottom-right subplot to percentage
    fig.update_yaxes(tickformat='.0%', row=1, col=2)
    fig.update_yaxes(tickformat='.0%', row=1, col=3)
    fig.update_yaxes(tickformat='.0%', row=1, col=4)
    fig.update_yaxes(tickformat='.0%', row=2, col=4)
    
    # Emptying x-axis values
    for i in range(8):
        row = 1 if i < 4 else 2
        col = i % 4 + 1    
        fig.update_xaxes(showticklabels=False, row=row, col=col)  

    return fig


def generate_hist_metrics_bars(start_date, end_date, bar_title):
    # Function to update graphs based on the selected date range  
    dfc = dfhist[(dfhist.index >= start_date) & (dfhist.index <= end_date)]
    
    dfD = dfc.resample('D').agg({'pnl_u':'sum', 'pnl_c':'sum', 'pnl_cl':'sum', 'betsize_u':'sum', 'betsize_c':'sum', 'betsize_cl':'sum'})
    dfD = dfD[dfD.pnl_u != 0]  
 
    pnlcols = ['pnl_u', 'pnl_c', 'pnl_cl']
    sizecols = ['betsize_u', 'betsize_c', 'betsize_cl']

    ir = 0.02  #current IBKR USD rate
    dfD[pnlcols] = dfD[pnlcols] + ir / 252  
    
    dfM = dfc.resample('M').agg({'pnl_u':'sum', 'pnl_c':'sum', 'pnl_cl':'sum', 'betsize_u':'sum', 'betsize_c':'sum', 'betsize_cl':'sum'})
    dfM[pnlcols] = dfM[pnlcols] + ir / 12  
    
    # calculate the Sharpe Ratio
    sharplist = []
    for col in pnlcols:
        sharplist.append(Sharpe(dfD[col]))

    dfsharp = pd.DataFrame([sharplist], columns=['s_u', 's_c', 's_cl'])
    
    # dataprep drawdown
    ddlist = []
    for col in pnlcols:
        ddlist.append(DrawDown(dfD[col]))
    print(ddlist)
    dfdd = pd.DataFrame([ddlist], columns=['dd_u', 'dd_c', 'dd_cl'])

    # dataprep winrate
    winlist = []
    for col in pnlcols:
        winlist.append(WinRate(dfc[col]))
    dfwin = pd.DataFrame([winlist], columns=['wr_u', 'wr_c', 'wr_cl'])
    
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
    
    #dataprep profitratio
    profitlist = []
    for col in pnlcols:
        profitlist.append(ProfitRatio(dfc[col]))
    dfprofit = pd.DataFrame([profitlist], columns=['pr_u', 'pr_c', 'pr_cl'])

    # dataprep the winning days vs the losing days
    windlist = []
    for col in pnlcols:
        windlist.append(WinRate(dfD[col]))
    dfwind = pd.DataFrame([windlist], columns=['wdr_u', 'wdr_c', 'wdr_cl'])

    # dataprep the winning months vs the losing months
    winmlist = []
    for col in pnlcols:
        winmlist.append(WinRate(dfM[col]))
    dfwinm = pd.DataFrame([winmlist], columns=['wmr_u', 'wmr_c', 'wmr_cl'])
        
    # create BarChart for Sharpe Ratio    
    bar_sharp = px.bar(dfsharp, x=dfsharp.index, y=['s_u', 's_c', 's_cl'], title=f'<b>{bar_title} Sharpe Ratio</b>',
                    color_discrete_sequence = colors_config['colors']['palet']
                     )
    bar_sharp.update_layout(
                        plot_bgcolor=colors_config['colors']['bg_figs'],
                        paper_bgcolor = colors_config['colors']['surround_figs'],
                        font_color = colors_config['colors']['text'],
                        font_family = colors_config['colors']['font'],
                        barmode = 'group',
                        bargap = 0.3,
                        margin = {'l':10, 'r':30, 't':40, 'b':0, 'pad':0},
                        title = {'font':{'size':14} },
                        xaxis = {'title':''},
                        yaxis = {'title':''},
                        showlegend = False
                        )
    
    bar_dd = px.bar(dfdd, x=dfdd.index, y=['dd_u', 'dd_c', 'dd_cl'], title=f'<b>{bar_title} DrawDown %</b>',
                    color_discrete_sequence = colors_config['colors']['palet']
                     )
    bar_dd.update_layout(
                        plot_bgcolor=colors_config['colors']['bg_figs'],
                        paper_bgcolor = colors_config['colors']['surround_figs'],
                        font_color = colors_config['colors']['text'],
                        font_family = colors_config['colors']['font'],
                        barmode = 'group',
                        margin = {'l':10, 'r':30, 't':40, 'b':0, 'pad':0},
                        title = {'font':{'size':14} },
                        xaxis = {'title':''},
                        yaxis = {'title':'', 'tickformat': ',.0%'},
                        showlegend = False
                        )

    bar_win = px.bar(dfwin, x=dfwin.index, y=['wr_u', 'wr_c', 'wr_cl'], title=f'<b>{bar_title} Winrate</b>',
                    color_discrete_sequence = colors_config['colors']['palet']
                     )
    bar_win.update_layout(
                        plot_bgcolor=colors_config['colors']['bg_figs'],
                        paper_bgcolor = colors_config['colors']['surround_figs'],
                        font_color = colors_config['colors']['text'],
                        font_family = colors_config['colors']['font'],
                        barmode = 'group',
                        margin = {'l':10, 'r':30, 't':40, 'b':0, 'pad':0},
                        title = {'font':{'size':14} },
                        xaxis = {'title':''},
                        yaxis = {'title':'', 'tickformat':',.0%'},
                        showlegend = False
                        )    
    
    
    bar_to = px.bar(dfto, x=dfto.index, y=['to_u', 'to_c', 'to_cl'], title=f'<b>{bar_title} Avg Daily\nTurnover</b>',
                    color_discrete_sequence = colors_config['colors']['palet']
                     )
    bar_to.update_layout(
                        plot_bgcolor=colors_config['colors']['bg_figs'],
                        paper_bgcolor = colors_config['colors']['surround_figs'],
                        font_color = colors_config['colors']['text'],
                        font_family = colors_config['colors']['font'],
                        barmode = 'group',
                        margin = {'l':10, 'r':30, 't':40, 'b':0, 'pad':0},
                        title = {'font':{'size':14} },
                        xaxis = {'title':''},
                        yaxis = {'title':''},
                        showlegend = False
                        )
    
    # multiplot = go.Figure(data=[bar_mto, bar_pr, bar_wind, bar_winm])
    # multiplot.update_layout(height=600, width=800, title='Multiple Bar Charts')
    bar_mto = px.bar(dfmto, x=dfmto.index, y=['mt_u', 'mt_c', 'mt_cl'], title=f'<b>{bar_title} Max Daily\nTurnover</b>',
                    color_discrete_sequence = colors_config['colors']['palet']
                     )

    
    
    bar_mto.update_layout(
                        plot_bgcolor=colors_config['colors']['bg_figs'],
                        paper_bgcolor = colors_config['colors']['surround_figs'],
                        font_color = colors_config['colors']['text'],
                        font_family = colors_config['colors']['font'],
                        barmode = 'group',
                        margin = {'l':10, 'r':30, 't':40, 'b':0, 'pad':0},
                        title = {'font':{'size':14} },
                        xaxis = {'title':''},
                        yaxis = {'title':''},
                        showlegend = False
                        )
    
    bar_pr = px.bar(dfprofit, x=dfprofit.index, y=['pr_u', 'pr_c', 'pr_cl'], title=f'<b>{bar_title} ProfitRatio</b>',
                    color_discrete_sequence = colors_config['colors']['palet']
                      )
    bar_pr.update_layout(
                        plot_bgcolor=colors_config['colors']['bg_figs'],
                        paper_bgcolor = colors_config['colors']['surround_figs'],
                        font_color = colors_config['colors']['text'],
                        font_family = colors_config['colors']['font'],
                        barmode = 'group',
                        bargap = 0.2,
                        margin = {'l':10, 'r':30, 't':40, 'b':0, 'pad':0},
                        title = {'font':{'size':14} },
                        xaxis = {'title':''},
                        yaxis = {'title':''},
                        showlegend = False
                        )  
    
    bar_wind = px.bar(dfwind, x=dfwind.index, y=['wdr_u', 'wdr_c', 'wdr_cl'], title=f'<b>{bar_title} Win Days</b>',
                    color_discrete_sequence = colors_config['colors']['palet']
                      )
    bar_wind.update_layout(
                        plot_bgcolor=colors_config['colors']['bg_figs'],
                        paper_bgcolor = colors_config['colors']['surround_figs'],
                        font_color = colors_config['colors']['text'],
                        font_family = colors_config['colors']['font'],
                        barmode = 'group',
                        margin = {'l':10, 'r':30, 't':40, 'b':0, 'pad':0},
                        title = {'font':{'size':14} },
                        xaxis = {'title':''},
                        yaxis = {'title':'', 'tickformat': ',.0%'},
                        showlegend = False
                        ) 
    bar_winm = px.bar(dfwinm, x=dfwinm.index, y=['wmr_u', 'wmr_c', 'wmr_cl'], title=f'<b>{bar_title} Win Months</b>',
                    color_discrete_sequence = colors_config['colors']['palet']
                      )
    bar_winm.update_layout(
                        plot_bgcolor=colors_config['colors']['bg_figs'],
                        paper_bgcolor = colors_config['colors']['surround_figs'],
                        font_color = colors_config['colors']['text'],
                        font_family = colors_config['colors']['font'],
                        barmode = 'group',
                        margin = {'l':10, 'r':30, 't':40, 'b':0, 'pad':0},
                        title = {'font':{'size':14} },
                        xaxis = {'title':''},
                        yaxis = {'title':'', 'tickformat':',.0%'},
                        showlegend = False
                        ) 
    
    # multiplot = go.Figure(data=[bar_pr, bar_winm, bar_wind, bar_sharp])
    # multiplot.update_layout(height=600, width=800, title='Multiple Bar Charts')

    return bar_sharp, bar_dd, bar_win, bar_to, bar_mto, bar_pr, bar_wind, bar_winm

def generate_hist_multi_barplot(start_date, end_date, bar_title):
    # Function to update graphs based on the selected date range  
    dfc = dfhist[(dfhist.index >= start_date) & (dfhist.index <= end_date)]
    
    
    dfD = dfc.resample('D').agg({'pnl_u':'sum', 'pnl_c':'sum', 'pnl_cl':'sum', 'betsize_u':'sum', 'betsize_c':'sum', 'betsize_cl':'sum'})
    dfD = dfD[dfD.pnl_u != 0]  
 
    pnlcols = ['pnl_u', 'pnl_c', 'pnl_cl']
    sizecols = ['betsize_u', 'betsize_c', 'betsize_cl']

    ir = 0.02  #current IBKR USD rate
    dfD[pnlcols] = dfD[pnlcols] + ir / 252  
    
    dfM = dfc.resample('M').agg({'pnl_u':'sum', 'pnl_c':'sum', 'pnl_cl':'sum', 'betsize_u':'sum', 'betsize_c':'sum', 'betsize_cl':'sum'})
    dfM[pnlcols] = dfM[pnlcols] + ir / 12  
    
    dflongs = dfc[pnlcols][dfc.posi == 1]
    dfshorts = dfc[pnlcols][dfc.posi == -1]
    
    # # calculate the Sharpe Ratio
    # sharplist = []
    # for col in pnlcols:
    #     sharplist.append(Sharpe(dfD[col]))

    # dfsharp = pd.DataFrame([sharplist], columns=['s_u', 's_c', 's_cl'])
    
    # # dataprep drawdown
    # ddlist = []
    # for col in pnlcols:
    #     ddlist.append(DrawDown(dfD[col]))
    # dfdd = pd.DataFrame([ddlist], columns=['dd_u', 'dd_c', 'dd_cl'])

    # # dataprep winrate
    # winlist = []
    # for col in pnlcols:
    #     winlist.append(WinRate(dfc[col]))
    # dfwin = pd.DataFrame([winlist], columns=['wr_u', 'wr_c', 'wr_cl'])
    
    # # dataprep average daily Turnover
    # tolist = []
    # for col in sizecols:
    #     tolist.append(TurnOver(dfD[col]))
    # dfto = pd.DataFrame([tolist], columns=['to_u', 'to_c', 'to_cl'])
    
    # # dataprep average daily Turnover
    # mtolist = []
    # for col in sizecols:
    #     mtolist.append((dfD[col].max()))
    # dfmto = pd.DataFrame([mtolist], columns=['mt_u', 'mt_c', 'mt_cl'])
    
    # #dataprep profitratio
    # profitlist = []
    # for col in pnlcols:
    #     profitlist.append(ProfitRatio(dfc[col]))
    # dfprofit = pd.DataFrame([profitlist], columns=['pr_u', 'pr_c', 'pr_cl'])

    # # dataprep the winning days vs the losing days
    # windlist = []
    # for col in pnlcols:
    #     windlist.append(WinRate(dfD[col]))
    # dfwind = pd.DataFrame([windlist], columns=['wdr_u', 'wdr_c', 'wdr_cl'])

    # # dataprep the winning months vs the losing months
    # winmlist = []
    # for col in pnlcols:
    #     winmlist.append(WinRate(dfM[col]))
    # dfwinm = pd.DataFrame([winmlist], columns=['wmr_u', 'wmr_c', 'wmr_cl'])
    # print(dfwinm)
    
    # # dataprep the performance of the long trades
    # longslist = []
    # for col in pnlcols:
    #     longslist.append(dflongs[col].sum())
    # dflong = pd.DataFrame([longslist], columns=['l_u', 'l_c', 'l_cu'])
    # print(dflong)
    
    # # dataprep the performance of the short trades
    # shortslist = []
    # for col in pnlcols:
    #     shortslist.append(dfshorts[col].sum())
    # dfshort = pd.DataFrame([shortslist], columns=['sh_u', 'sh_c', 'sh_cu']) 
       
    # # Create individual bar charts
    # bar_sharp = px.bar(dfsharp, y=['s_u', 's_c', 's_cl'], color_discrete_sequence=colors_config['colors']['palet'])
    # bar_dd = px.bar(dfdd, y=['dd_u', 'dd_c', 'dd_cl'], color_discrete_sequence=colors_config['colors']['palet'])
    # bar_win = px.bar(dfwin, y=['wr_u', 'wr_c', 'wr_cl'], color_discrete_sequence=colors_config['colors']['palet'])
    # bar_to = px.bar(dfto, y=['to_u', 'to_c', 'to_cl'], color_discrete_sequence=colors_config['colors']['palet'])
    # bar_mto = px.bar(dfmto, y=['mt_u', 'mt_c', 'mt_cl'], color_discrete_sequence=colors_config['colors']['palet'])
    # bar_pr = px.bar(dfprofit, y=['pr_u', 'pr_c', 'pr_cl'], color_discrete_sequence=colors_config['colors']['palet'])
    # bar_wind = px.bar(dfwind, y=['wdr_u', 'wdr_c', 'wdr_cl'], color_discrete_sequence=colors_config['colors']['palet'])
    # bar_winm = px.bar(dfwinm, y=['wmr_u', 'wmr_c', 'wmr_cl'], color_discrete_sequence=colors_config['colors']['palet'])
    # bar_longs = px.bar(dflong, y=['l_u', 'l_c', 'l_cl'], color_discrete_sequence=colors_config['colors']['palet'])
    # bar_shorts = px.bar(dfshort, y=['sh_u', 'sh_c', 'sh_cl'], color_discrete_sequence=colors_config['colors']['palet'])

    # Data preparation
    sharplist = [Sharpe(dfD[col]) for col in pnlcols]
    dfsharp = pd.DataFrame([sharplist], columns=['s_u', 's_c', 's_cl'])
    
    ddlist = [DrawDown(dfD[col]) for col in pnlcols]
    dfdd = pd.DataFrame([ddlist], columns=['dd_u', 'dd_c', 'dd_cl'])
    
    winlist = [WinRate(dfc[col]) for col in pnlcols]
    dfwin = pd.DataFrame([winlist], columns=['wr_u', 'wr_c', 'wr_cl'])
    
    tolist = [TurnOver(dfD[col]) for col in sizecols]
    dfto = pd.DataFrame([tolist], columns=['to_u', 'to_c', 'to_cl'])
    
    mtolist = [dfD[col].max() for col in sizecols]
    dfmto = pd.DataFrame([mtolist], columns=['mt_u', 'mt_c', 'mt_cl'])
    
    profitlist = [ProfitRatio(dfc[col]) for col in pnlcols]
    dfprofit = pd.DataFrame([profitlist], columns=['pr_u', 'pr_c', 'pr_cl'])
    
    windlist = [WinRate(dfD[col]) for col in pnlcols]
    dfwind = pd.DataFrame([windlist], columns=['wdr_u', 'wdr_c', 'wdr_cl'])
    
    winmlist = [WinRate(dfM[col]) for col in pnlcols]
    dfwinm = pd.DataFrame([winmlist], columns=['wmr_u', 'wmr_c', 'wmr_cl'])
    
    longslist = [dflongs[col].sum() for col in pnlcols]
    dflong = pd.DataFrame([longslist], columns=['l_u', 'l_c', 'l_cl'])
    
    shortslist = [dfshorts[col].sum() for col in pnlcols]
    dfshort = pd.DataFrame([shortslist], columns=['s_u', 's_c', 's_cl'])

    # Colors
    colors = colors_config['colors']['palet']

    # Create individual bar charts
    bar_charts = []
    dataframes = [dfsharp, dfdd, dfwin, dfwind, dflong, dfto, dfmto, dfprofit, dfwinm, dfshort]
    
    for i, df in enumerate(dataframes):
        bar_chart = px.bar(df.T.reset_index(), x='index', y=0, 
                           color_discrete_sequence=colors)
        bar_charts.append(bar_chart)   
        
    # Create a subplot figure
    fig = make_subplots(rows=2, cols=5, subplot_titles=[
        "Sharpe Ratio", "DrawDown %", "WinRate", "Winning Days", "Performance Longs", "Avg Daily Turnover",
        "Max Daily Turnover", "ProfitRatio", "Winning Months", "Performance Shorts"
    ])
    
        # Helper function to add traces from a bar chart to the subplot
    def add_bar_traces(bar_chart, row, col):
        for trace in bar_chart.data:
            fig.add_trace(trace, row=row, col=col)
    
    # Add traces to the subplot
    for i, bar_chart in enumerate(bar_charts):
        row = 1 if i < 5 else 2
        col = i % 5 + 1
        add_bar_traces(bar_chart, row, col)

    # Update layout
    fig.update_layout(showlegend=False)
    fig.update_layout(plot_bgcolor=colors_config['colors']['bg_figs'],
                      paper_bgcolor=colors_config['colors']['surround_figs'],
                      font_color=colors_config['colors']['text'],
                      font_family=colors_config['colors']['font'],
                      barmode = 'group',
                      margin=dict(l=25, r=20, t=20, b=10),
                      )  # Minimize margins

    # Set tick format for the y-axis of the bottom-right subplot to percentage
    fig.update_yaxes(tickformat='.0%', row=1, col=2)
    fig.update_yaxes(tickformat='.0%', row=1, col=3)
    fig.update_yaxes(tickformat='.0%', row=1, col=4)
    fig.update_yaxes(tickformat='.1%', row=1, col=5)
    fig.update_yaxes(tickformat='.0%', row=2, col=4)
    fig.update_yaxes(tickformat='.1%', row=2, col=5)
    
    for i in range(10):
        row = 1 if i < 5 else 2
        col = i % 5 + 1    
        fig.update_xaxes(showticklabels=False, row=row, col=col)    
    
    return fig



def generate_histperf(start_date, end_date, plot_title):
    
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

    #create LineGraph for Returns    
    figln = px.line(cumdf, x=dfD.index, y=['pnl_u','pnl_c','pnl_cl'], title=f'<b>Performance {plot_title}</b>',
                    color_discrete_sequence = colors_config['colors']['palet'])
    figln.update_layout(
                        plot_bgcolor= "#FFFFFF",
                        paper_bgcolor = colors_config['colors']['surround_figs'],
                        font_color = colors_config['colors']['text'],
                        font_family = colors_config['colors']['font'],
                        margin = {'l':20, 'r':40, 't':70, 'b':0, 'pad':0},
                        title = {'x':0.5, 'y':0.95, 'font':{'size':20}},
                        xaxis = {'title':'', 'gridcolor':'#808080'},
                        yaxis = {'title':'', 'tickformat': '.2%', 'gridcolor':'#808080'},
                        legend = {'title': '', 'orientation':'h', 'y':0.99, 'xanchor':'left', 'x':0.03}
                        ) 
    figln.for_each_trace(lambda t: t.update(name=legend_labels[t.name]))
    
    return figln

def generate_donut(selected_tf):
    #data prep for Quarterly and Monthly comparisons / donutchart
    quarterlies = dfhist['pnl_cl'].resample('Q').sum()
    total_quarterlies = pd.DataFrame(quarterlies.groupby(quarterlies.index.quarter).mean())
    q_names = ['Q1', 'Q2', 'Q3', 'Q4']
    total_quarterlies['names'] = q_names
    
    monthlies = dfhist['pnl_cl'].resample('M').sum()
    total_monthlies = pd.DataFrame(monthlies.groupby(monthlies.index.month).mean())
    
    m_names = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL','AUG','SEP','OCT','NOV','DEC' ]
    total_monthlies['names'] = m_names
    
    pie_dict = {'Monthly':total_monthlies, 'Quarterly':total_quarterlies}
    df_pie = round(pie_dict[selected_tf], 4)    
    
    donut = px.pie(
                df_pie,
                values='pnl_cl',
                names='names',
                title = f'<b>Expected {selected_tf} Performance</b>',
                color_discrete_sequence = colors_config['colors']['palet'],
                hole=0.4,
                labels = {'pnl_cl': 'Actual Value', 'tickformat':',.2%'}
                )
    
    donut.update_layout(
                        plot_bgcolor=colors_config['colors']['bg_figs'],
                        paper_bgcolor = colors_config['colors']['surround_figs'],
                        font_color = colors_config['colors']['text'],
                        font_family = colors_config['colors']['font'],
                        barmode = 'group',
                        bargap = 0.2,
                        margin = {'l':20, 'r':30, 't':60, 'b':20, 'pad':0},
                        title = {'font':{'size':18} },
                        showlegend = True
                        )
    donut.update_traces(textinfo='label+value',
                        texttemplate='%{label}: %{value:.2%}',  # Format text as label: percentage
                        hovertemplate='%{label}: %{value:.2%}',  # Format hover text as label: percentage
                        )
    return donut

def generate_individual_stock_graph(ticker, start_date):
    """ Creates a lineplot with performance of individual stocks """
    
    dfu = df[df.index >= start_date]
    dff = dfu[dfu['ticker'] == ticker]
    pnlcols = ['pnl_u', 'pnl_c', 'pnl_cl']
    dfcum = dff[pnlcols].cumsum()
    
    dummy_date = dfcum.index[0] - pd.Timedelta(days=1)
    dfcum.loc[dummy_date] = [0] * len(dfcum.columns)
    dfcum = dfcum.sort_index()
    
    figln = px.line(dfcum, x=dfcum.index, y=['pnl_u','pnl_c','pnl_cl'], title = f'{ticker}',
                   color_discrete_sequence = colors_config['colors']['palet']
                    ).update_layout(
                            plot_bgcolor= '#FFFFFF',
                            paper_bgcolor = colors_config['colors']['surround_figs'],
                            font_color = colors_config['colors']['text'],
                            font_family = colors_config['colors']['font'],
                            margin = {'l':20, 'r':40, 't':70, 'b':0, 'pad':0},
                            title = {'x':0.5, 'y':0.95, 'font':{'size':20}},
                            xaxis = {'title':''},
                            yaxis = {'title':'', 'tickformat': '.2%'},
                            legend = {'title': '', 'orientation':'h', 'y':0.99, 'xanchor':'left', 'x':0.03}
                            )
    figln.for_each_trace(lambda t: t.update(name=legend_labels[t.name]))

    return figln

def generate_bestinhistory_bar(selected_strat):
    # # dataprep for top15 table
    strat = selected_strat
    dfc = dfhist.copy()
    dftop10 = dfc.groupby('ticker')[strat].sum().sort_values(ascending=False).head(10)
    dftop10 = dftop10.reset_index()
    dftop10 = dftop10.rename(columns={'ticker':'Top10', strat:'Top_PnL'})
    dftop10 = dftop10.sort_values(by='Top_PnL', ascending=True)
    
    dfworst10 = dfc.groupby('ticker')[strat].sum().sort_values(ascending=True).head(10)
    dfworst10 = dfworst10.reset_index()
    dfworst10 = dfworst10.rename(columns={'ticker':'Worst10', strat:'Worst_PnL'})
  #  dfworst10 = dfworst10[strat].sort_values(ascending=False)

    colors1 = px.colors.diverging.RdYlGn
    colors2 = px.colors.sequential.YlOrRd

 # Create the bar chart
    bar_best = px.bar(dftop10, x='Top_PnL', y='Top10', title=f'<b>Top 10 Stocks of All Time</b>', 
                      orientation='h', color='Top10', color_discrete_sequence=colors1)

    bar_best.update_layout(
                        plot_bgcolor= '#000000',
                        paper_bgcolor = colors_config['colors']['surround_figs'],
                        font_color = colors_config['colors']['text'],
                        font_family = colors_config['colors']['font'],
                        margin = {'l':10, 'r':30, 't':40, 'b':0, 'pad':0},
                        title = {'x':0.5, 'font':{'size':16} },
                        xaxis = {'title':'', 'tickformat':',.2%'},
                        yaxis = {'title':'', 'categoryorder': 'total ascending'},
                        showlegend = False
                        ) 
    bar_worst = px.bar(dfworst10, x='Worst_PnL', y='Worst10', title=f'<b>Worst 10 Stocks of All Time</b>', 
                      orientation='h', color='Worst10', color_discrete_sequence=colors1)

    bar_worst.update_layout(
                        plot_bgcolor= '#000000',
                        paper_bgcolor = colors_config['colors']['surround_figs'],
                        font_color = colors_config['colors']['text'],
                        font_family = colors_config['colors']['font'],
                        margin = {'l':10, 'r':30, 't':40, 'b':0, 'pad':0},
                        title = {'x':0.5, 'font':{'size':16} },
                        xaxis = {'title':'', 'tickformat':',.2%'},
                        yaxis = {'title':''},
                        showlegend = False
                        ) 
    
    return bar_best, bar_worst
    
    