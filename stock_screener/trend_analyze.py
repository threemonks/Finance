import argparse
import logging

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import nltk.stem

DEFAULT_FINANCE_FILEPATH_ROOT = r'/data/finance'


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

english_stemmer = nltk.stem.SnowballStemmer('english')

class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(TfidfVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))

def trend_study(inputfile):
    df = pd.read_csv(inputfile, header = 0, sep='\t')

    # drop rows with null values in any column
    df = df.dropna()

    # Dropping the team 1
    # drop rows that any of the columns shortName, longName, category or fundFamily contain one of partial strings "china", "short", "bear", "inverse"
    discard = ["china", "short", "bear", "inverse"]
    df = df[~df.shortName.str.contains('|'.join(discard), case=False) & ~df.longName.str.contains('|'.join(discard), case=False) & ~df.category.str.contains('|'.join(discard), case=False) & ~df.fundFamily.str.contains('|'.join(discard), case=False)]

    # Assuming 'df' is your DataFrame with ETF data

    # Select ETFs with the lowest prices relative to the past 52-week range
    threshold_value = 0.20  # Set your threshold value
    lowest_price_etfs = df[df['price52WRangePct'] <= threshold_value]

    my_stop_words = text.ENGLISH_STOP_WORDS.union(["etf", "proshares", "global", "china", "msci", "ishares", "trading", "funds", "trust", "miscellaneous", "index", "broad", "vaneck"])

    # Text data analysis
    tfidf_vectorizer = StemmedTfidfVectorizer(stop_words=list(my_stop_words), analyzer='word')
    text_features = tfidf_vectorizer.fit_transform(lowest_price_etfs['shortName'] + lowest_price_etfs['longName'] + ' ' + lowest_price_etfs['category'] + ' ' + lowest_price_etfs['fundFamily'])

    # Get feature names (words)
    feature_names = tfidf_vectorizer.get_feature_names_out()

    # Summarize the importance of words
    word_importance = pd.DataFrame({
        'Word': feature_names,
        'Importance': text_features.sum(axis=0).A1
    })

    # Display top words by importance
    top_words = word_importance.sort_values(by='Importance', ascending=False).head(20)
    print(top_words)

    # Create a word cloud
    wordcloud = WordCloud(width=1200, height=600, background_color='white').generate_from_frequencies(dict(zip(top_words['Word'], top_words['Importance'])))

    # Plot the WordCloud image
    plt.figure(figsize=(20, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()


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

    trend_study(args.inputfile)

if __name__ == '__main__':
    main()