import matplotlib
matplotlib.use('Agg')
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from util import get_data

def compute_indicators(sd=dt.datetime(2003, 1, 1), ed=dt.datetime(2004, 12, 31), \
    syms=['NFLX'], rolling_days=20):
    prices_all = get_data(syms, pd.date_range(sd, ed))
    prices = prices_all[syms]
    prices_normalized = normalize_stocks(prices)
    volume = get_data(syms, pd.date_range(sd, ed), colname='Volume')[syms[0]]
    volume = normalize_stocks(volume)

    sma = SMA(prices_normalized, rolling_days)
    columns = ['Price/SMA']
    prices_sma_ratio = pd.DataFrame(0, index = prices_normalized.index, columns = columns)
    prices_sma_ratio['Price/SMA'] = prices_normalized[syms[0]]/sma['SMA']
    bb = bollinger_bands(prices_normalized, rolling_days, sma)
    momentum = MOMENTUM(prices_normalized, rolling_days)
    vol = compute_vrc(volume, rolling_days)

    bb_percent = pd.DataFrame(0, index = prices_normalized.index, columns = columns)
    bb_percent['BBP'] = (prices_normalized[syms[0]] - bb['lower']) / (bb['upper'] - bb['lower'])

    return prices_normalized, prices_sma_ratio, momentum, bb_percent, vol


def SMA(prices_normalized, rolling_days):
    columns = ['SMA']
    sma = pd.DataFrame(0, index = prices_normalized.index, columns = columns)
    sma['SMA'] = prices_normalized.rolling(window=rolling_days).mean()
    return sma

def bollinger_bands(prices_normalized, rolling_days, sma, sd=2):
    columns = ['lower', 'upper']
    bb = pd.DataFrame(0, index = prices_normalized.index, columns = columns)
    bands = pd.DataFrame(0, index = prices_normalized.index, columns = ['band'])
    bands['band'] = prices_normalized.rolling(window = rolling_days, min_periods = rolling_days).std()
    bb['upper'] = sma['SMA'] + (bands['band'] * sd)
    bb['lower'] = sma['SMA'] - (bands['band'] * sd)
    return bb


def MOMENTUM(prices_normalized, rolling_days):
    columns =['Momentum']
    momentum = pd.DataFrame(0, index = prices_normalized.index, columns = columns)
    momentum['Momentum'] = prices_normalized.diff(rolling_days)/prices_normalized.shift(rolling_days)
    return momentum

def compute_vrc(volume, rolling_days):
    col = ['Volume']
    vol = pd.DataFrame(0, index=volume.index, columns=col)
    #vol[col] = (volume - volume.rolling(window=rolling_days).mean()) / volume.rolling(window=rolling_days).mean()
    vol[col] = (volume - volume.rolling(window=rolling_days).mean()) / volume.rolling(window=rolling_days).mean()
    return vol

def normalize_stocks(prices):
    fill_missing_values(prices)
    return prices / prices.ix[0, :]

def fill_missing_values(prices):
    """Fill missing values in data frame, in place."""
    prices.fillna(method='ffill', inplace=True)
    prices.fillna(method='bfill', inplace=True)

def test_code():
    compute_indicators()

if __name__ == "__main__":
    test_code()