import argparse
import json
import logging
import os
from datetime import datetime, date, timedelta
import time

from pandas_datareader import data as pdr
from pandas import ExcelWriter
import yfinance as yf
import pandas as pd
from requests import Session
from requests_cache import CacheMixin, SQLiteCache
from requests_ratelimiter import LimiterMixin, MemoryQueueBucket
from pyrate_limiter import Duration, RequestRate, Limiter

import matplotlib.pyplot as plt

class CachedLimiterSession(CacheMixin, LimiterMixin, Session):
    pass

DEFAULT_FINANCE_FILEPATH_ROOT = r'/data/finance'


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

class SimpleJsonEncoder(json.JSONEncoder):
    def default(self, z):
        if isinstance(z, datetime.datetime):
            return (str(z))
        else:
            return super().default(z)

def get_sp500_tickers():
    companies = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    table = companies[0]
    # the below code removes tickers that have no data (from previous experience)
    df = table[table["Symbol"].str.contains("BRK.B|BF.B") == False]
    symbols = df['Symbol'].to_list()

    return symbols

def stock_screen(symbols, start_date=None, end_date=None, datafile_root=None, outputfile=None):

    yyyymmdd = datetime.now().strftime("%Y%m%d")

    # Download the dividends for each symbol and concatenate the results
    summary_keys = ['symbol', 'shortName', 'longName', 'sector', 'category', 'fundFamily', 'marketCap', 'totalAssets', 'industry', 'yearLow', 'yearHigh', 'lastPrice']
    symbol_summary_data = []
    for idx, symbol in enumerate(symbols):
        logger.info(f"[{idx}] processing symbol {symbol}")
        symbol_object_info = None
        symbol_info_data_filepath = os.path.join(datafile_root, f'{symbol}_info.csv')
        if os.path.exists(symbol_info_data_filepath):
            # if the file is less than 24 hours old, read from file
            symbol_info_file_time = os.path.getmtime(symbol_info_data_filepath)
            if (time.time() - symbol_info_file_time) < 86400:
                logger.info(f"Reading {symbol} from data file")
                # Reading from file

                with open(symbol_info_data_filepath, "r") as f:
                    symbol_object_info = json.load(f)

        if not symbol_object_info:
            try:
                ticker_object = yf.Ticker(symbol)
                symbol_object_info = ticker_object.info
                fastinfo = dict(ticker_object.fast_info)
                symbol_object_info.update(**fastinfo)
            except Exception as e:
                logger.error(f"Error processing ticker {symbol}: {e}")
                continue

        # Convert and write JSON object to file
        with open(symbol_info_data_filepath, "w") as outfile:
            json.dump(symbol_object_info, outfile, cls=SimpleJsonEncoder)

        summary = {key: symbol_object_info[key] for key in summary_keys if key in symbol_object_info}

        symbol_summary_data.append(summary)

    ticker_summary_df = pd.DataFrame.from_dict(symbol_summary_data, orient="columns")
    ticker_summary_df = ticker_summary_df.set_index('symbol')
    if not outputfile:
        outputfile = os.path.join(datafile_root, f'all_symbol_summary_{yyyymmdd}.csv')
        logger.info(f"\nwrite ticker summary into file {outputfile=}")

    ticker_summary_df.to_csv(outputfile)

    # import matplotlib.pyplot as plt
    # df = ticker_summary_df
    # df['Price_52w_Range'] = (df['lastPrice'] - df['yearLow']) / (df['yearHigh'] - df['yearLow'])
    # groups = df.index[:20]
    # values1 = df.Price_52w_Range[:20]
    # values2 = [1 - x for x in df.Price_52w_Range[:20]]
    #
    # fig, ax = plt.subplots()
    #
    # # Stacked bar chart
    # ax.bar(groups, values1)
    # ax.bar(groups, values2, bottom = values1)
    # plt.xticks(rotation=90)
    #
    # plt.show()

def main():
    aparser = argparse.ArgumentParser(description="ETF Screener", epilog="Good luck!")
    aparser.add_argument("--loglevel", dest="loglevel", type=int, action="store",
                      help="loglevel")
    aparser.add_argument("-r", "--datafile_root", dest="datafile_root", type=str, action="store",
                      help="the output csv file root to write history data into, individual file will be asin_<datatype>.csv")
    aparser.add_argument("-i", "--inputfile", dest="inputfile", type=str, action="store",
                      help="the input file that contains all tickers to be processed")

    aparser.add_argument("-o", "--outputfile", dest="outputfile", type=str, action="store",
                      help="the output file that contains all tickers summary")

    args, extras = aparser.parse_known_args()

    FORMAT = "%(asctime)s - {%(pathname)s:%(lineno)s} - %(levelname)s - %(message)s"
    logging.basicConfig(format=FORMAT, level=args.loglevel or logging.INFO)

    logging.getLogger(__name__).setLevel(args.loglevel or logging.INFO)

    start_date = datetime.now() - timedelta(days=365)
    end_date = date.today()

    tickers = extras
    if not tickers:
        if args.inputfile:
            with open(args.inputfile, 'r') as f:
                tickers = [line.strip() for line in f.readlines()]
        else:
            logger.warning('No tickers specified, use sp500 tickers')
            tickers = get_sp500_tickers()

    stock_screen(tickers, start_date, end_date, args.datafile_root or DEFAULT_FINANCE_FILEPATH_ROOT, args.outputfile or None)

if __name__ == '__main__':
    main()