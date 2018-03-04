"""MC2-P1: Market simulator.

Copyright 2017, Georgia Tech Research Corporation
Atlanta, Georgia 30332-0415
All Rights Reserved
"""

import pandas as pd
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data

def compute_portvals(orders_file = "./orders/orders.csv", start_val = 1000000, commission=9.95, impact=0.005):
    # this is the function the autograder will call to test your code
    # NOTE: orders_file may be a string, or it may be a file object. Your
    # code should work correctly with either input
    # TODO: Your code here

    orders = pd.read_csv(orders_file, index_col='Date', parse_dates=True, na_values=['nan'], usecols=['Date', 'Symbol', 'Order', 'Shares'])
    orders.sort_index(inplace = True)
    symbols = np.array(orders.Symbol.unique()).tolist()
    prices = get_data(symbols, pd.date_range(orders.index[0].to_datetime(), orders.index[-1].to_datetime()))
    pr = prices.index
    rows = orders.iterrows()
    series = pd.Series(start_val, pr)
    prices['Portfolio'], prices['Cash'] = series, series
    for symbol in symbols:  prices[symbol + ' Shares'] =  pd.Series(0, pr)
    for left, row in rows:
        ord = row['Order']
        symbol = row['Symbol']
        share = row['Shares']
        symShare = symbol + ' Shares'
        shares = prices.ix[left:, symShare]
        cash = prices.ix[left, symbol] * share
        if ord == 'SELL':
            shares -= share
            cash *= (1 - impact)
            cash -= commission
            prices.ix[left:, 'Cash'] += cash
        else:
            shares += share
            cash *= (1 + impact)
            cash += commission
            prices.ix[left:, 'Cash'] -= cash
        prices.ix[left:, symShare] = shares
    rows = prices.iterrows()
    for left, row in rows:
        i = 0
        for symbol in symbols:
            symShare, sym = symbol + ' Shares', row[symbol]
            i += prices.ix[left, symShare] * sym
            prices.ix[left, 'Portfolio'] = prices.ix[left, 'Cash'] + i
    return prices.ix[:, 'Portfolio']

def test_code():
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters

    of = "./orders/orders-02.csv"
    sv = 1000000

    # Process orders
    portvals = compute_portvals(orders_file = of, start_val = sv)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]] # just get the first column
    else:
        "warning, code did not return a DataFrame"
    
    # Get portfolio stats
    # Here we just fake the data. you should use your code from previous assignments.
    start_date = dt.datetime(2008,1,1)
    end_date = dt.datetime(2008,6,1)
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [0.2,0.01,0.02,1.5]
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = [0.2,0.01,0.02,1.5]

    # Compare portfolio against $SPX
    print ("Date Range: {} to {}".format(start_date, end_date))
    print
    print ("Sharpe Ratio of Fund: {}".format(sharpe_ratio))
    print ("Sharpe Ratio of SPY : {}".format(sharpe_ratio_SPY))
    print
    print ("Cumulative Return of Fund: {}".format(cum_ret))
    print ("Cumulative Return of SPY : {}".format(cum_ret_SPY))
    print
    print ("Standard Deviation of Fund: {}".format(std_daily_ret))
    print ("Standard Deviation of SPY : {}".format(std_daily_ret_SPY))
    print
    print ("Average Daily Return of Fund: {}".format(avg_daily_ret))
    print ("Average Daily Return of SPY : {}".format(avg_daily_ret_SPY))
    print
    print ("Final Portfolio Value: {}".format(portvals[-1]))

if __name__ == "__main__":
    test_code()
