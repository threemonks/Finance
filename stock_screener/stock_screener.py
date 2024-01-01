import argparse
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

ETF_TICKERS = [
    'AAPB', 'AAPD', 'AAPU', 'AGNG', 'AIQ', 'AIRR', 'ALUM', 'AQWA', 'ARKG', 'ARKK', 'ARKW', 'ARKX', 'ARMR', 'ARVR', 'AWAY', 'BABX', 'BATT', 'BBC', 'BBH', 'BBP', 'BBRE', 'BCDF', 'BDCX', 'BDCZ', 'BDRY', 'BECO', 'BERZ', 'BIB', 'BIS', 'BITQ', 'BITW', 'BIZD', 'BKCH', 'BLCN', 'BLDG', 'BLKC', 'BLLD', 'BLOK', 'BMED', 'BNGE', 'BNKD', 'BNKU', 'BOAT', 'BPAY', 'BTEC', 'BTEK', 'BUG', 'BULD', 'BULZ', 'BUZZ', 'BWEB', 'BYRE', 'CGW', 'CHB', 'CHIH', 'CHII', 'CHIK', 'CHIM', 'CHIR', 'CHIS', 'CHIU', 'CHIX', 'CIBR', 'CIRC', 'CLDL', 'CLIX', 'CLNR', 'CLOU', 'CNBS', 'CNCR', 'CONL', 'COPX', 'CQQQ', 'CRIT', 'CRPT', 'CURE', 'CUT', 'CWEB', 'DAPP', 'DAT', 'DFEN', 'DFGR', 'DFNL', 'DGIN', 'DMAT', 'DPST', 'DRN', 'DRUP', 'DRV', 'DTEC', 'DULL', 'DUSL', 'DUST', 'EATV', 'EATZ', 'EBLU', 'ECLN', 'ECON', 'EDOC', 'EFRA', 'EKG', 'EMFQ', 'EMIF', 'EMQQ', 'ERET', 'EUFN', 'EVX', 'EXI', 'FAS', 'FAZ', 'FBL', 'FBT', 'FCLD', 'FCOM', 'FDHT', 'FDIG', 'FDN', 'FFND', 'FHLC', 'FIDU', 'FINX', 'FITE', 'FIVG', 'FIW', 'FMAT', 'FMET', 'FMQQ', 'FNCL', 'FNGD', 'FNGG', 'FNGO', 'FNGU', 'FPRO', 'FREL', 'FRI', 'FSTA', 'FTAG', 'FTEC', 'FTRI', 'FTXG', 'FTXH', 'FTXL', 'FTXO', 'FTXR', 'FUTY', 'FXG', 'FXH', 'FXL', 'FXO', 'FXR', 'FXU', 'FXZ', 'FYLG', 'GABF', 'GAMR', 'GAST', 'GBLD', 'GDOC', 'GDX', 'GDXD', 'GDXJ', 'GDXU', 'GERM', 'GFOF', 'GII', 'GINN', 'GLIF', 'GMET', 'GNOM', 'GNR', 'GOAU', 'GOEX', 'GQRE', 'GREI', 'GREK', 'GRID', 'GRNR', 'GUNR', 'HACK', 'HAIL', 'HAP', 'HART', 'HAUS', 'HAUZ', 'HDGE', 'HELX', 'HOMZ', 'HTEC', 'HYLG', 'IAI', 'IAK', 'IAT', 'IBB', 'IBBQ', 'IBLC', 'IBOT', 'IBRN', 'ICF', 'IDAT', 'IDNA', 'IDU', 'IETC', 'IEUS', 'IEV', 'IFGL', 'IFRA', 'IGE', 'IGF', 'IGM', 'IGN', 'IGPT', 'IGV', 'IHAK', 'IHE', 'IHF', 'IHI', 'INDF', 'INDS', 'INFL', 'INFR', 'INQQ', 'IPAY', 'IQM', 'IRBO', 'ISRA', 'ITA', 'ITB', 'ITEQ', 'IVEG', 'IVES', 'IWFH', 'IWTR', 'IXG', 'IXJ', 'IXN', 'IXP', 'IYF', 'IYG', 'IYH', 'IYJ', 'IYK', 'IYM', 'IYR', 'IYT', 'IYW', 'IYZ', 'IZRL', 'JDST', 'JETD', 'JETS', 'JETU', 'JFWD', 'JHMU', 'JNUG', 'JPRE', 'JRE', 'JXI', 'KBE', 'KBWB', 'KBWD', 'KBWP', 'KBWR', 'KBWY', 'KCE', 'KEMQ', 'KFVG', 'KIE', 'KLIP', 'KNCT', 'KNGS', 'KOIN', 'KRE', 'KROP', 'KSTR', 'KTEC', 'KURE', 'KWEB', 'KXI', 'LABD', 'LABU', 'LEGR', 'LIT', 'LOUP', 'LRNZ', 'LTL', 'MAKX', 'MDEV', 'MEDI', 'METV', 'MINV', 'MOO', 'MOON', 'MORT', 'MOTO', 'MRAD', 'MSFD', 'MSFU', 'MXI', 'NAIL', 'NANR', 'NBDS', 'NDIV', 'NETL', 'NFRA', 'NUGT', 'NURE', 'NVDL', 'NVDS', 'NXTG', 'OGIG', 'PAVE', 'PBE', 'PBJ', 'PEX', 'PFI', 'PHDG', 'PHO', 'PICK', 'PILL', 'PINK', 'PIO', 'PJP', 'PKB', 'PNQI', 'POTX', 'PPA', 'PPH', 'PPTY', 'PRN', 'PRNT', 'PSCC', 'PSCF', 'PSCH', 'PSCI', 'PSCM', 'PSCT', 'PSCU', 'PSI', 'PSIL', 'PSL', 'PSP', 'PSR', 'PTF', 'PTH', 'PUI', 'PYZ', 'QABA', 'QQH', 'QQQ', 'QTEC', 'QTUM', 'RBLD', 'RDOG', 'REET', 'REIT', 'REK', 'REM', 'REMX', 'REW', 'REZ', 'RFEU', 'RING', 'RITA', 'RNEW', 'ROBO', 'ROKT', 'ROM', 'ROOF', 'RSPC', 'RSPF', 'RSPH', 'RSPM', 'RSPN', 'RSPR', 'RSPS', 'RSPT', 'RSPU', 'RWO', 'RWR', 'RWX', 'RXD', 'RXL', 'SARK', 'SATO', 'SBIO', 'SCHH', 'SDP', 'SEA', 'SEF', 'SEMI', 'SGDJ', 'SGDM', 'SHLD', 'SHNY', 'SHPP', 'SIJ', 'SIL', 'SILJ', 'SIMS', 'SKF', 'SKYU', 'SKYY', 'SLVP', 'SLX', 'SMH', 'SMN', 'SNSR', 'SOCL', 'SOXL', 'SOXQ', 'SOXS', 'SOXX', 'SPRE', 'SPRX', 'SRET', 'SRS', 'SSG', 'STCE', 'SUPL', 'SZK', 'TARK', 'TCHI', 'TDIV', 'TDV', 'TECL', 'TECS', 'TEMP', 'THNQ', 'TIME', 'TINT', 'TINY', 'TOLZ', 'TPOR', 'TRFK', 'TWEB', 'TYLG', 'UBOT', 'UCYB', 'UGE', 'UPW', 'URA', 'URE', 'URNM', 'USD', 'USRT', 'UTES', 'UTSL', 'UXI', 'UYG', 'UYM', 'VAW', 'VDC', 'VEGI', 'VERS', 'VFH', 'VGT', 'VHT', 'VIS', 'VMOT', 'VNQ', 'VNQI', 'VOX', 'VPC', 'VPN', 'VPU', 'VR', 'VRAI', 'WBAT', 'WBIF', 'WBIL', 'WCBR', 'WCLD', 'WDNA', 'WEBL', 'WEBS', 'WFH', 'WGMI', 'WOOD', 'WPS', 'WTAI', 'WUGI', 'XAR', 'XBI', 'XDAT', 'XHB', 'XHE', 'XHS', 'XITK', 'XLB', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV', 'XME', 'XNTK', 'XPH', 'XPND', 'XSD', 'XSW', 'XT', 'XTL', 'XTN', 'XWEB', 'YUMY', 'ZIG', 'TLT'
]

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

def get_sp500_tickers():
    companies = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    table = companies[0]
    # the below code removes tickers that have no data (from previous experience)
    df = table[table["Symbol"].str.contains("BRK.B|BF.B") == False]
    symbols = df['Symbol'].to_list()

    return symbols

def stock_dividend(symbols, start_date=None, end_date=None, datafile_root=None):

    # Download the dividends for each symbol and concatenate the results
    dfs = []
    prices = []
    for idx, symbol in enumerate(symbols):
        logger.info(f"[{idx}] processing ticker {symbol}")
        stock = yf.Ticker(symbol)
        history = stock.history(start=start_date, end=end_date)
        dividends = history["Dividends"].to_frame(name=symbol)
        dfs.append(dividends)

        price = stock.fast_info["lastPrice"]
        prices.append(price)

    # drop columns with NaN and/or zero values
    df = pd.concat(dfs, axis=1).dropna(axis=1, how='all')
    # df = df.loc[:, (df != 0).any(axis=0)]
    df.index.name = "Date"

    # add 'Annual Dividend' as the last row

    df.loc['Annual Dividend'] = df.sum(axis=0)

    # create a new frame with 'Annual Dividend' row only

    annual_div = df.loc['Annual Dividend'].to_frame()
    print(f"\n{annual_div=}")

    # add 'Current_Price' column
    annual_div["Current_Price"] = prices

    annual_div['Dividend Yield %'] = round((annual_div['Annual Dividend'] / annual_div['Current_Price']) * 100, 2)

    annual_div = annual_div.sort_values('Dividend Yield %', ascending=False)

    fig, ax = plt.subplots(figsize=(20, 7))
    annual_div['Dividend Yield %'][:15].plot.barh(ax=ax, color=['g', 'r']*2)
    ax.set_title('Dividend Yields 2023')
    ax.set_ylabel('Stock Symbol')
    ax.set_xlabel('Dividend Yield')

    # Add text labels to the bars
    for i, v in enumerate(annual_div['Dividend Yield %'][:15]):
        ax.text(v + 0.2, i, "{:.2f}%".format(v))

    plt.show()

def main():
    aparser = argparse.ArgumentParser(description="ETF Screener", epilog="Good luck!")
    aparser.add_argument("--loglevel", dest="loglevel", type=int, action="store",
                      help="loglevel")
    aparser.add_argument("-r", "--datafile_root", dest="datafile_root", type=str, action="store",
                      help="the output csv file root to write history data into, individual file will be asin_<datatype>.csv")

    args, extras = aparser.parse_known_args()

    FORMAT = "%(asctime)s - {%(pathname)s:%(lineno)s} - %(levelname)s - %(message)s"
    logging.basicConfig(format=FORMAT, level=args.loglevel or logging.INFO)

    logging.getLogger(__name__).setLevel(args.loglevel or logging.INFO)

    start_date = datetime.now() - timedelta(days=365)
    end_date = date.today()

    tickers = extras
    if not tickers:
        logger.warning('No tickers specified, use sp500 tickers')
        tickers = get_sp500_tickers()

    stock_dividend(tickers[:15], start_date, end_date, args.datafile_root or DEFAULT_FINANCE_FILEPATH_ROOT)
    # bulk_download(tickers[:2], start_date, end_date, args.datafile_root or DEFAULT_FINANCE_FILEPATH_ROOT)
    # batch_download(tickers, start_date, end_date, args.datafile_root or DEFAULT_FINANCE_FILEPATH_ROOT)

if __name__ == '__main__':
    main()