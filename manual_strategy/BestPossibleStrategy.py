import matplotlib
matplotlib.use('Agg')
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import marketsimcode as ms
from util import get_data
import indicators as ind

def testPolicy(symbol = "AAPL", sd=dt.datetime(2010,1,1), ed=dt.datetime(2011,12,31), sv = 100000):
    syms = [symbol]
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    prices_normalized = ind.normalize_stocks(prices)
    columns = ['Symbol', 'Order', 'Shares']
    orders = pd.DataFrame(0, index=prices_normalized.index, columns=[columns[2]])
    buy_sell = pd.DataFrame('BUY', index=prices_normalized.index, columns=[columns[1]])
    symbol_df = pd.DataFrame(symbol, index=prices_normalized.index, columns=[columns[0]])
    total_holdings = 0
    for index, row in prices_normalized.iterrows():
        cur_iloc = prices_normalized.index.get_loc(index)
        current_price = row[symbol]
        if (cur_iloc < prices_normalized.shape[0] - 1):
            next_index = prices_normalized.index[cur_iloc + 1]
            future_price = prices_normalized.loc[next_index][symbol]
            if (future_price > current_price) and (total_holdings < 1000):
                buy_sell.loc[index]['Order'] = 'BUY'
                if total_holdings == 0:
                    orders.loc[index]['Shares'] = 1000
                    total_holdings += 1000
                elif total_holdings == -1000:
                    orders.loc[index]['Shares'] = 2000
                    total_holdings += 2000
            elif (future_price < current_price) and (total_holdings > -1000):
                buy_sell.loc[index]['Order'] = 'SELL'
                if total_holdings == 0:
                    orders.loc[index]['Shares'] = 1000
                    total_holdings = total_holdings - 1000
                elif total_holdings == 1000:
                    orders.loc[index]['Shares'] = 2000
                    total_holdings = total_holdings - 2000
    df_trades = pd.concat([symbol_df, buy_sell, orders], axis=1)
    df_trades.columns = ['Symbol', 'Order', 'Shares']
    df_trades = df_trades[df_trades.Shares != 0]
    return df_trades

def test_code():
    sd=dt.datetime(2003,1,1)
    ed=dt.datetime(2004,12,31)
    sv = 100000
    df_trades = testPolicy(symbol = "NFLX", sd=sd, ed=ed, sv = sv)
    portvals = ms.compute_portvals(df_trades, start_val = sv)
    syms = ['NFLX']
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)
    prices_NFLX = prices_all['NFLX']
    prices_NFLX_normalized = ind.normalize_stocks(prices_NFLX)
    prices_portval_normalized = ind.normalize_stocks(portvals)
    chart_df = pd.concat([prices_portval_normalized, prices_NFLX_normalized], axis=1)
    chart_df.columns = ['Portfolio', 'Benchmark']
    chart_df.plot(grid=True, title='Best Possible Strategy', use_index=True, color=['Black', 'Blue'])

    plt.savefig('newBest.png')


if __name__ == "__main__":
    test_code()