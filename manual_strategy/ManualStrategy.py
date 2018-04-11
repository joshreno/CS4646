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
    prices_all = get_data(syms, pd.date_range(sd, ed))  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    prices_normalized = ind.normalize_stocks(prices)
    volume = get_data(syms, pd.date_range(sd, ed), colname='Volume')[symbol]
    volume_normalized = ind.normalize_stocks(volume)
    days = 20
    sma = ind.sma(prices_normalized, days)
    prices_sma_ratio = pd.DataFrame(0, index = sma.index, columns = ['Price/SMA'])
    prices_sma_ratio['Price/SMA'] = prices_normalized[symbol]/sma['SMA']
    bb = ind.bollinger_bands(prices_normalized, days, sma)
    bb_percent = pd.DataFrame(0, index = prices_normalized.index, columns = ['BBP'])
    bb_percent['BBP'] = (prices_normalized[symbol] - bb['LOWER']) / (bb['UPPER'] - bb['LOWER'])
    volume = ind.vrc(volume_normalized, days / 2)
    orders = pd.DataFrame(0, index = prices_normalized.index, columns = ['Shares'])
    buy_sell = pd.DataFrame('BUY', index = prices_normalized.index, columns = ['Order'])
    symbol_df = pd.DataFrame(symbol, index = prices_normalized.index, columns = ['Symbol'])
    total_holdings = 0
    for index, row in prices_normalized.iterrows():
        price_sma = prices_sma_ratio.loc[index]['Price/SMA']
        bbp_value = bb_percent.loc[index]['BBP']
        vrc = volume.loc[index]['Volume Rate of Change']
        if (bbp_value > 0.3 and bbp_value < 0.8) and (vrc < 0) and (price_sma < 1) and (total_holdings < 1000):
            buy_sell.loc[index]['Order'] = 'BUY'
            if total_holdings == 0:
                orders.loc[index]['Shares'] = 1000
                total_holdings += 1000
            elif total_holdings == -1000:
                orders.loc[index]['Shares'] = 2000
                total_holdings += 2000
        elif (bbp_value > 1)  and (vrc > 0.5) and (price_sma > 1.1) and (total_holdings > -1000):
            buy_sell.loc[index]['Order'] = 'SELL'
            if total_holdings == 0:
                orders.loc[index]['Shares'] = 1000
                total_holdings = total_holdings - 1000
            elif total_holdings == 1000:
                orders.loc[index]['Shares'] = 2000
                total_holdings = total_holdings - 2000

    if total_holdings == 2000:
        buy_sell.loc[prices_normalized.index[-1]]['Order'] = 'SELL'
        orders.loc[prices_normalized.index[-1]]['Shares'] = 2000
    else:
        buy_sell.loc[prices_normalized.index[-1]]['Order'] = 'BUY'
        orders.loc[prices_normalized.index[-1]]['Shares'] = 2000

    df_trades = pd.concat([symbol_df, buy_sell, orders], axis=1)
    df_trades.columns = ['Symbol', 'Order', 'Shares']
    df_trades = df_trades[df_trades.Shares != 0]

    return df_trades

def code():
    sd=dt.datetime(2003,1,1)
    ed=dt.datetime(2004,12,31)
    sv = 100000
    df_trades = testPolicy(symbol = "NFLX", sd=sd, ed=ed, sv = sv)
    portvals = ms.compute_portvals(df_trades, start_val = sv, commission=9.95, impact=0.005)
    prices_portval_normalized = ind.normalize_stocks(portvals)
    prices_SPY_normalized = pd.read_csv('first.csv', names=['First', 'Second'])
    prices_SPY_normalized.set_index(prices_SPY_normalized['First'], inplace=True)
    del prices_SPY_normalized['First']
    chart_df = pd.concat([prices_portval_normalized, prices_SPY_normalized], axis=1)
    chart_df.columns = ['Portfolio', 'Benchmark']
    chart_df.plot(grid=True, title='Manual Strategy vs. Benchmark Index (In-Sample)', use_index=True, color=['Black', 'Blue'])
    num = 0
    for index, row in df_trades.iterrows():
        if df_trades.loc[index]['Order'] == 'BUY':
            num += df_trades.loc[index]['Shares']
            if num == 0:
                plt.axvline(x=index, color='k', linestyle='-')
            else:
                plt.axvline(x=index, color='g', linestyle='-')
        elif df_trades.loc[index]['Order'] == 'SELL':
            num -= df_trades.loc[index]['Shares']
            if num == 0:
                plt.axvline(x=index, color='k', linestyle='-')
            else:
                plt.axvline(x=index, color='r', linestyle='-')
    plt.savefig('insample.png')
    plt.savefig('insample.png')
    print('Out of Sample Stats:')
    sd=dt.datetime(2005,1,1)
    ed=dt.datetime(2006,12,31)
    sv = 100000
    df_trades = testPolicy(symbol = "NFLX", sd=sd, ed=ed, sv = sv)
    portvals = ms.compute_portvals(df_trades, start_val = sv)
    syms = ['NFLX']
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)
    prices_portval_normalized = ind.normalize_stocks(portvals)
    prices_SPY_normalized = pd.read_csv('second.csv', names=['First', 'Second'])
    prices_SPY_normalized.set_index(prices_SPY_normalized['First'], inplace=True)
    del prices_SPY_normalized['First']
    chart_df = pd.concat([prices_portval_normalized, prices_SPY_normalized], axis=1)
    chart_df.columns = ['Portfolio', 'Benchmark']
    chart_df.plot(grid=True, title='Manual Strategy vs. Benchmark Index (Out-Of-Sample)', use_index=True, color=['Black', 'Blue'])

if __name__ == "__main__":
    code()