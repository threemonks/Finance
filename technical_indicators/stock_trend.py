import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import sys
import argparse


def fetch_stock_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date)
    return df


def calculate_rsi(data, periods=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_trend_index(data):
    data['MA'] = data['Close'].rolling(window=4).mean()
    data['Low_4'] = data['Close'].rolling(window=4).min()
    data['High_4'] = data['Close'].rolling(window=4).max()
    data['MA_Index'] = 100 * (data['MA'] - data['Low_4']) / (data['High_4'] - data['Low_4'])
    data['RSI'] = calculate_rsi(data)
    data['Trend_Index'] = data['MA_Index'] - data['RSI']
    return data


def analyze_trend(data):
    data['Trend'] = 'Neutral'
    data.loc[data['Trend_Index'] > 30, 'Trend'] = 'Strong'
    data.loc[data['Trend_Index'] > 15, 'Trend'] = 'Very Strong'
    data['Curve'] = data['Trend_Index'].diff()
    return data


def plot_results(data, ticker):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    ax1.plot(data.index, data['Close'], label='Close Price')
    ax1.set_ylabel('Price')
    ax1.set_title(f'{ticker} Stock Price and Trend Analysis')
    ax1.legend()

    ax2.plot(data.index, data['Trend_Index'], label='Trend Index')
    ax2.axhline(y=15, color='r', linestyle='--', label='Very Strong Trend')
    ax2.axhline(y=30, color='g', linestyle='--', label='Strong Trend')
    ax2.set_ylabel('Trend Index')
    ax2.legend()

    plt.xlabel('Date')
    plt.tight_layout()
    plt.show()


def main(ticker):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # Get one year of data

    try:
        data = fetch_stock_data(ticker, start_date, end_date)
        if data.empty:
            print(f"No data available for ticker {ticker}")
            return

        data = calculate_trend_index(data)
        data = analyze_trend(data)

        print(data[['Close', 'MA_Index', 'RSI', 'Trend_Index', 'Trend', 'Curve']].tail())

        plot_results(data, ticker)
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze stock trend for a given ticker.")
    parser.add_argument("ticker", type=str, help="Stock ticker symbol")
    args = parser.parse_args()

    main(args.ticker.upper())