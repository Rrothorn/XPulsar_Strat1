# -*- coding: utf-8 -*-
"""
Created on Sun May  5 13:08:31 2024

@author: Gebruiker
"""
import pandas as pd
import numpy as np
import datetime

from config import colors_config


import plotly.express as px

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
df = pd.read_csv(f'C:/Users/Gebruiker/Documents/Trading/DC_reports/{fname}', parse_dates = ['date'], index_col = 'date')

fname2 = 'XGB_trades18-23.csv'
dfhist = pd.read_csv(f'C:/Users/Gebruiker/Documents/Trading/BackTest/{fname2}', parse_dates = ['date'], index_col = 'date')
dfhist = dfhist.rename(columns = {'weighted_pnl':'pnl_u', 'betsize':'betsize_u'})


legend_labels = {'pnl_u': 'All-in', 'pnl_c': 'Conditional', 'pnl_cl': 'Leveraged'}


def generate_perf_plot1(start_date, end_date, figln_title):
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

    #create LineGraph for Returns 
    
    figln = px.line(cumdf, x=cumdf.index, y=['pnl_u','pnl_c','pnl_cl'], title=f'<b>{figln_title} Performance 2024</b>',
                    color_discrete_sequence = colors_config['colors']['palet'])
    figln.update_layout(
                        plot_bgcolor=colors_config['colors']['bg_figs'],
                        paper_bgcolor = colors_config['colors']['surround_figs'],
                        font_color = colors_config['colors']['text'],
                        font_family = colors_config['colors']['font'],
                        margin = {'l':20, 'r':40, 't':70, 'b':0, 'pad':0},
                        title = {'x':0.5, 'y':0.95, 'font':{'size':20}},
                        xaxis = {'title':'', 'gridcolor':'#808080'},
                        yaxis = {'title':'', 'tickformat': '.2%', 'gridcolor':'#808080'},
                        legend = {'title': '', 'orientation':'h', 'y':1.14, 'xanchor':'right', 'x':0.9}
                        )

    
    figln.for_each_trace(lambda t: t.update(name=legend_labels[t.name]))
    return figln  

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
    dfD = dfc.resample('D').agg({'pnl_u':'sum', 'pnl_c':'sum', 'pnl_cl':'sum', 'betsize_u':'sum', 'betsize_c':'sum', 'betsize_cl':'sum'})
    dfD = dfD[dfD.pnl_u != 0]  
 
    pnlcols = ['pnl_u', 'pnl_c', 'pnl_cl']
    sizecols = ['betsize_u', 'betsize_c', 'betsize_cl']

    ir = 0.048  #current IBKR USD rate
    dfD[pnlcols] = dfD[pnlcols] + ir / 252     
      
    
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
                        yaxis = {'title':'', 'tickformat': ',.1%'},
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
                        yaxis = {'title':''},
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

    return bar_sharp, bar_dd, bar_win, bar_to, bar_mto, bar_pr  

def generate_hist_metrics_bars(start_date, end_date, bar_title):
    # Function to update graphs based on the selected date range  
    dfc = dfhist[(dfhist.index >= start_date) & (dfhist.index <= end_date)]
    dfD = dfc.resample('D').agg({'pnl_u':'sum', 'pnl_c':'sum', 'pnl_cl':'sum', 'betsize_u':'sum', 'betsize_c':'sum', 'betsize_cl':'sum'})
    dfD = dfD[dfD.pnl_u != 0]  
 
    pnlcols = ['pnl_u', 'pnl_c', 'pnl_cl']
    sizecols = ['betsize_u', 'betsize_c', 'betsize_cl']

    ir = 0.048  #current IBKR USD rate
    dfD[pnlcols] = dfD[pnlcols] + ir / 252     
      
    
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
                        yaxis = {'title':'', 'tickformat': ',.1%'},
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
                        yaxis = {'title':''},
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

    return bar_sharp, bar_dd, bar_win, bar_to, bar_mto, bar_pr  



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
                        plot_bgcolor=colors_config['colors']['bg_figs'],
                        paper_bgcolor = colors_config['colors']['surround_figs'],
                        font_color = colors_config['colors']['text'],
                        font_family = colors_config['colors']['font'],
                        margin = {'l':20, 'r':40, 't':70, 'b':0, 'pad':0},
                        title = {'x':0.5, 'y':0.95, 'font':{'size':20}},
                        xaxis = {'title':'', 'gridcolor':'#808080'},
                        yaxis = {'title':'', 'tickformat': '.2%', 'gridcolor':'#808080'},
                        legend = {'title': '', 'orientation':'h', 'y':1.14, 'xanchor':'right', 'x':0.9}
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
                labels = {'pnl_cl': 'Actual Value'}
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
    donut.update_traces(textinfo='label+value')
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
    print(dfcum)
    figln = px.line(dfcum, x=dfcum.index, y=['pnl_u','pnl_c','pnl_cl'], title = f'{ticker}',
                   color_discrete_sequence = colors_config['colors']['palet']
                    ).update_layout(
                            plot_bgcolor= colors_config['colors']['bg_figs'],
                            paper_bgcolor = colors_config['colors']['surround_figs'],
                            font_color = colors_config['colors']['text'],
                            font_family = colors_config['colors']['font'],
                            margin = {'l':20, 'r':40, 't':70, 'b':0, 'pad':0},
                            title = {'x':0.5, 'y':0.95, 'font':{'size':20}},
                            xaxis = {'title':''},
                            yaxis = {'title':'', 'tickformat': '.2%'},
                            legend = {'title': '', 'orientation':'h', 'y':1.14, 'xanchor':'right', 'x':0.9}
                            )
    figln.for_each_trace(lambda t: t.update(name=legend_labels[t.name]))

    return figln