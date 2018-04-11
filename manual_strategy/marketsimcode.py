import pandas as pd
import numpy as np
from util import get_data

def compute_portvals(orders, start_val = 100000, commission=0.00, impact=0.00):
    # orders = pd.read_csv(orders_file, index_col='Date', parse_dates=True, na_values=['nan'], usecols=['Date', 'Symbol', 'Order', 'Shares'])
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
