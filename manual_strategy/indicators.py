import matplotlib
matplotlib.use('Agg')
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from util import get_data

def indicators(symbols = ['NFLX'], sd = dt.datetime(2003, 1, 1), ed=dt.datetime(2004, 12, 31)):
    symbol = symbols[0]
    prices = get_data(symbols, pd.date_range(sd, ed))[symbol]
    volume = get_data(symbols, pd.date_range(sd, ed), colname='Volume')[symbol]
    prices_normalized = normalize_stocks(prices)
    volume_normalized = normalize_stocks(volume)
    days = 20

    simple = sma(prices_normalized, days)
    bb = bollinger_bands(prices_normalized, days, simple)
    volume = vrc(volume_normalized, days / 2)

    prices_sma_ratio = pd.DataFrame(0, index=prices_normalized.index, columns=['Price/SMA'])
    prices_sma_ratio['Price/SMA'] = prices_normalized / simple['SMA']
    sma_plot = pd.concat([prices_normalized, simple, prices_sma_ratio], axis=1)
    sma_plot.columns = [symbol, 'SMA', 'Price/SMA']
    sma_plot.plot(grid=True, title='Simple Moving Average', use_index=True)

    plt.savefig('sma.png')

    bb_percent = pd.DataFrame(0, index=prices_normalized.index, columns=['BBP'])
    bb_percent['BBP'] = (prices_normalized - bb['LOWER']) / (bb['UPPER'] - bb['LOWER'])
    bb_plot = pd.concat([prices_normalized, bb['LOWER'], bb['UPPER'], bb_percent['BBP']], axis=1)
    bb_plot.columns = [symbol, 'Lower band', 'Upper band', "BBP"]
    bb_plot.plot(grid=True, title='Bollinger Bands', use_index=True)
    plt.savefig('bb.png')

    vol_plot = pd.concat([volume_normalized, volume, prices_normalized], axis=1)
    vol_plot.columns = [symbol + ' Volume', 'Volume Rate of Change', 'Price']
    vol_plot.plot(grid=True, title='Volume Rate of Change', use_index=True)

    plt.savefig('volume.png')

def sma(prices, days):
    col = ['SMA']
    sma = pd.DataFrame(0, index=prices.index, columns = col)
    sma['SMA'] = prices.rolling(window=days, min_periods=days).mean()
    return sma

def bollinger_bands(prices, days, sma, sd = 2):
    col = ['LOWER', 'UPPER']
    bb = pd.DataFrame(0, index= prices.index, columns=col)
    bands = pd.DataFrame(0, index= prices.index, columns=['BAND'])
    bands['BAND'] = prices.rolling(window=days, min_periods=days).std()
    bb['UPPER'] = sma['SMA'] + (bands['BAND'] * sd)
    bb['LOWER'] = sma['SMA'] - (bands['BAND'] * sd)
    return bb

def vrc(volume, days):
    col = ['Volume Rate of Change']
    vol = pd.DataFrame(0, index = volume.index, columns = col)
    vol[col] = (volume - volume.rolling(window=days, min_periods=days).mean())/volume.rolling(window=days, min_periods=days).mean()
    return vol

def normalize_stocks(prices):
    fill_missing_values(prices)
    return prices / prices.ix[0,]

def fill_missing_values(prices):
    prices.fillna(method='ffill')
    prices.fillna(method='bfill')

def test_code():
    indicators()

if __name__ == "__main__":
    test_code()