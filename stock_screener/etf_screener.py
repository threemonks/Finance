import argparse
import logging
import os
import datetime
import sys
import time

from pandas_datareader import data as pdr
from pandas import ExcelWriter
import yfinance as yf
import pandas as pd
from requests import Session
from requests_cache import CacheMixin, SQLiteCache
from requests_ratelimiter import LimiterMixin, MemoryQueueBucket
from pyrate_limiter import Duration, RequestRate, Limiter

class CachedLimiterSession(CacheMixin, LimiterMixin, Session):
    pass

DEFAULT_FINANCE_FILEPATH_ROOT = r'/data/finance'

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

def bulk_download(tickers, start_date=None, end_date=None, datafile_root=None):

    all_stocks_outputfilepath = os.path.join(datafile_root, 'AllETFs.xlsx')
    yf.pdr_override()

    session = CachedLimiterSession(
        limiter=Limiter(RequestRate(2, Duration.SECOND * 5)),  # max 2 requests per 5 seconds
        bucket_class=MemoryQueueBucket,
        backend=SQLiteCache("yfinance.cache"),
    )

    all_stocks_df = yf.download(tickers, start=start_date, end=end_date, session=session)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print('\n', all_stocks_df)
    writer = ExcelWriter(all_stocks_outputfilepath)
    all_stocks_df.to_excel(writer, "Sheet1", index=True)
    writer.close()
    logger.info(f"saved all stocks to {all_stocks_outputfilepath}")

def batch_download(tickers, start_date=None, end_date=None, datafile_root=None):

    all_stocks_outputfilepath = os.path.join(datafile_root, 'AllETFsWithPercent52Week.xlsx')
    yf.pdr_override()

    session = CachedLimiterSession(
        limiter=Limiter(RequestRate(2, Duration.SECOND * 5)),  # max 2 requests per 5 seconds
        bucket_class=MemoryQueueBucket,
        backend=SQLiteCache("yfinance.cache"),
    )

    all_stocks = []

    for idx, ticker in enumerate(tickers):
        logger.info(f"[{idx}] processing ticker {ticker}")

        stock = yf.Ticker(ticker, session=session)
        name = stock.info['shortName']
        category = stock.info.get('category')
        fundFamily = stock.info.get('fundFamily')
        # The scraped response will be stored in the cache
        logger.info(f"{name=} {category=} {fundFamily=}")

        ticker_file = os.path.join(datafile_root, f'{ticker}.csv')
        df = None
        # don't download again if data file exists and it is < 24 hours old
        # if os.path.exists(ticker_file):
        #     file_time = os.path.getmtime(ticker_file)
        #     if (time.time() - file_time) < 86400:
        #         df = pd.read_csv(ticker_file, index_col=0)

        if df is None or df.empty:
            # Download historical data as CSV for each stock (makes the process faster)
            df = pdr.get_data_yahoo(ticker, start_date, end_date)
            if df.empty:
                logging.warning(f"no data for ticker {ticker}")
                continue
            df.to_csv(ticker_file)
            # pause for 1 second to avoid getting blocked by Yahoo Finance
            time.sleep(1)

        sma = [50, 150, 200]
        for x in sma:
            df["SMA_" + str(x)] = round(df['Adj Close'].rolling(window=x).mean(), 2)

        # Storing required values
        currentClose = df["Adj Close"].iloc[-1]
        moving_average_50 = df["SMA_50"].iloc[-1]
        moving_average_150 = df["SMA_150"].iloc[-1]
        moving_average_200 = df["SMA_200"].iloc[-1]
        low_of_52week = round(min(df["Low"].iloc[-260:]), 2)
        high_of_52week = round(max(df["High"].iloc[-260:]), 2)
        dividend_per_share = df["DividendPerShare"].iloc[-1]
        dividends_payable = df["DividendsPayable"].iloc[-1]


        percent_52week = round((currentClose - low_of_52week) / (high_of_52week - low_of_52week) * 100, 2)

        # all stocks
        all_stocks.append({'Stock': ticker, "category": category, "fundFamily": fundFamily,
                           "Name": name, "Adj Close": currentClose, "50 Day MA": moving_average_50,
                                        "150 Day Ma": moving_average_150, "200 Day MA": moving_average_200,
                                        "52 Week Low": low_of_52week, "52 Week High": high_of_52week, "Percent 52 Week": percent_52week,
                                        "DividendPerShare": dividend_per_share, "DividendsPayable": dividends_payable})

    # all stocks df
    all_stocks_df = pd.DataFrame(all_stocks,
        columns=['Stock', "category", "fundFamily", "Name", "Adj Close", "50 Day MA", "150 Day Ma", "200 Day MA", "52 Week Low", "52 Week High",
                 "Percent 52 Week", "DividendPerShare", "DividendsPayable"])
    all_stocks_df = all_stocks_df.sort_values(by='Percent 52 Week', ascending=True)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print('\n', all_stocks_df)
    writer = ExcelWriter(all_stocks_outputfilepath)
    all_stocks_df.to_excel(writer, "Sheet1", index=False)
    writer.close()
    logger.info(f"saved all stocks to {all_stocks_outputfilepath}")

def main():
    aparser = argparse.ArgumentParser(description="ETF Screener", epilog="Good luck!")
    aparser.add_argument("--loglevel", dest="loglevel", type=int, action="store",
                      help="loglevel")
    aparser.add_argument("-r", "--datafile_root", dest="datafile_root", type=str, action="store",
                      help="the output csv file root to write history data into, individual file will be asin_<datatype>.csv")
    aparser.add_argument("-i", "--inputfile", dest="inputfile", type=str, action="store",
                      help="the input file that contains all tickers to be processed")

    args, extras = aparser.parse_known_args()

    FORMAT = "%(asctime)s - {%(pathname)s:%(lineno)s} - %(levelname)s - %(message)s"
    logging.basicConfig(format=FORMAT, level=args.loglevel or logging.INFO)

    logging.getLogger(__name__).setLevel(args.loglevel or logging.INFO)

    start_date = datetime.datetime.now() - datetime.timedelta(days=365)
    end_date = datetime.date.today()

    tickers = extras
    if not tickers:
        if args.inputfile:
            with open(args.inputfile, 'r') as f:
                tickers = [line.strip() for line in f.readlines()]
        else:
            logger.warning('Must provide tickers')
            sys.exit(1)

    bulk_download(tickers[:2], start_date, end_date, args.datafile_root or DEFAULT_FINANCE_FILEPATH_ROOT)
    batch_download(tickers, start_date, end_date, args.datafile_root or DEFAULT_FINANCE_FILEPATH_ROOT)

if __name__ == '__main__':
    main()