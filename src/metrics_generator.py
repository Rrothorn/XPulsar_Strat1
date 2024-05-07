# -*- coding: utf-8 -*-
"""
Created on Sun May  5 13:30:29 2024

@author: Gebruiker
"""

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


# donwloading data
fname = 'DC_2024trades.csv'
df = pd.read_csv(f'C:/Users/Gebruiker/Documents/Trading/DC_reports/{fname}', parse_dates = ['date'], index_col = 'date')

fname2 = 'XGB_trades18-23.csv'
dfhist = pd.read_csv(f'C:/Users/Gebruiker/Documents/Trading/BackTest/{fname2}', parse_dates = ['date'], index_col = 'date')
dfhist = dfhist.rename(columns = {'weighted_pnl':'pnl_u', 'betsize':'betsize_u'})


def generate_metrics(start_date, end_date):
    # Function to update graphs based on the selected date range  
    dfc = df[(df.index >= start_date) & (df.index <= end_date)]
    dfD = dfc.resample('D').agg({'pnl_u':'sum', 'pnl_c':'sum', 'pnl_cl':'sum', 'betsize_u':'sum', 'betsize_c':'sum', 'betsize_cl':'sum'})
    dfD = dfD[dfD.pnl_u != 0]  
 
    pnlcols = ['pnl_u', 'pnl_c', 'pnl_cl']

    ir = 0.048  #current IBKR USD rate
    dfD[pnlcols] = dfD[pnlcols] + ir / 252     
    
    # dataprep line graph  
    cumdf = dfD[pnlcols].cumsum()
    
    # dataprep cards
    # calculate the performance
    pnl = round(cumdf.pnl_cl.iloc[-1], 4)
    pnl = "{:.2%}".format(pnl)
    
    # calculate the Sharpe Ratio
    sharplist = []
    for col in pnlcols:
        sharplist.append(Sharpe(dfD[col]))

    dfsharp = pd.DataFrame([sharplist], columns=['s_u', 's_c', 's_cl'])
    sharp = round(sharplist[2], 2)
    
    # calculate the maximum drawdown over the period
    ddlist = []
    for col in pnlcols:
        ddlist.append(DrawDown(dfD[col]))
    dfdd = pd.DataFrame([ddlist], columns=['dd_u', 'dd_c', 'dd_cl'])
    maxdd = round(ddlist[2], 4)
    maxdd = "{:.2%}".format(maxdd)
    
    #calculate the winrate of all trades
    winlist = []
    for col in pnlcols:
        winlist.append(WinRate(dfc[col]))
    dfwin = pd.DataFrame([winlist], columns=['wr_u', 'wr_c', 'wr_cl'])
    winrate = winlist[2]
    winrate = "{:.2%}".format(winrate)

    #calculate the ratio of the average winning pnl vs the average losing pnl
    profitlist = []
    for col in pnlcols:
        profitlist.append(ProfitRatio(dfc[col]))
    dfprofit = pd.DataFrame([profitlist], columns=['pr_u', 'pr_c', 'pr_cl'])
    profitrate = round(profitlist[2], 2)
    
    return pnl, sharp, maxdd, winrate, profitrate

def generate_hist_metrics(start_date, end_date):
    # Function to update graphs based on the selected date range  
    dfc = dfhist[(dfhist.index >= start_date) & (dfhist.index <= end_date)]
    dfD = dfc.resample('D').agg({'pnl_u':'sum', 'pnl_c':'sum', 'pnl_cl':'sum', 'betsize_u':'sum', 'betsize_c':'sum', 'betsize_cl':'sum'})
    dfD = dfD[dfD.pnl_u != 0]  
 
    pnlcols = ['pnl_u', 'pnl_c', 'pnl_cl']

    ir = 0.048  #current IBKR USD rate
    dfD[pnlcols] = dfD[pnlcols] + ir / 252     
    
    # dataprep line graph  
    cumdf = dfD[pnlcols].cumsum()
    
    # dataprep cards
    # calculate the performance
    pnl = round(cumdf.pnl_cl.iloc[-1], 4)
    pnl = "{:.2%}".format(pnl)
    
    # calculate the Sharpe Ratio
    sharplist = []
    for col in pnlcols:
        sharplist.append(Sharpe(dfD[col]))

    dfsharp = pd.DataFrame([sharplist], columns=['s_u', 's_c', 's_cl'])
    sharp = round(sharplist[2], 2)
    
    # calculate the maximum drawdown over the period
    ddlist = []
    for col in pnlcols:
        ddlist.append(DrawDown(dfD[col]))
    dfdd = pd.DataFrame([ddlist], columns=['dd_u', 'dd_c', 'dd_cl'])
    maxdd = round(ddlist[2], 4)
    maxdd = "{:.2%}".format(maxdd)
    
    #calculate the winrate of all trades
    winlist = []
    for col in pnlcols:
        winlist.append(WinRate(dfc[col]))
    dfwin = pd.DataFrame([winlist], columns=['wr_u', 'wr_c', 'wr_cl'])
    winrate = winlist[2]
    winrate = "{:.2%}".format(winrate)

    #calculate the ratio of the average winning pnl vs the average losing pnl
    profitlist = []
    for col in pnlcols:
        profitlist.append(ProfitRatio(dfc[col]))
    dfprofit = pd.DataFrame([profitlist], columns=['pr_u', 'pr_c', 'pr_cl'])
    profitrate = round(profitlist[2], 2)
    
    return pnl, sharp, maxdd, winrate, profitrate
    