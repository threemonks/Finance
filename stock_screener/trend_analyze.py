import argparse
import logging

import matplotlib.pyplot as plt
import pandas as pd
import spacy
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

def trend_study_tfidf(inputfile, word_count=20, threshold_value=0.20, top=True):
    df = pd.read_csv(inputfile, header = 0, sep='\t')

    # drop rows with null values in any column
    df = df.dropna()

    # Dropping the team 1
    # drop rows that any of the columns shortName, longName, category or fundFamily contain one of partial strings "china", "short", "bear", "inverse"
    discard = ["china", "short", "bear", "inverse"]
    df = df[~df.shortName.str.contains('|'.join(discard), case=False) & ~df.longName.str.contains('|'.join(discard), case=False) & ~df.category.str.contains('|'.join(discard), case=False) & ~df.fundFamily.str.contains('|'.join(discard), case=False)]

    # Assuming 'df' is your DataFrame with ETF data

    # Select ETFs with the lowest prices relative to the past 52-week range
    if not top:
        relevant_price_etfs = df[df['price52WRangePct'] <= threshold_value]
    else:
        relevant_price_etfs = df[df['price52WRangePct'] >= threshold_value]

    my_stop_words = text.ENGLISH_STOP_WORDS.union(["etf", "proshares", "global", "china", "msci", "ishares", "trading", "funds", "trust", "miscellaneous", "index", "broad", "vaneck"])

    # Text data analysis
    tfidf_vectorizer = StemmedTfidfVectorizer(stop_words=list(my_stop_words), analyzer='word')
    text_features = tfidf_vectorizer.fit_transform(relevant_price_etfs['shortName'] + relevant_price_etfs['longName'] + ' ' + relevant_price_etfs['category'] + ' ' + relevant_price_etfs['fundFamily'])

    # Get feature names (words)
    feature_names = tfidf_vectorizer.get_feature_names_out()

    # Summarize the importance of words
    word_importance = pd.DataFrame({
        'Word': feature_names,
        'Importance': text_features.sum(axis=0).A1
    })

    # Display top words by importance
    top_words = word_importance.sort_values(by='Importance', ascending=False).head(word_count)
    print(top_words)

    # Create a word cloud
    wordcloud = WordCloud(width=1200, height=600, background_color='white').generate_from_frequencies(dict(zip(top_words['Word'], top_words['Importance'])))

    # Plot the WordCloud image
    plt.figure(figsize=(20, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()


def trend_study_embeddings(inputfile, word_count=20, threshold_value=0.20, top=True):
    df = pd.read_csv(inputfile, header=0, sep='\t')

    # drop rows with null values in any column
    df = df.dropna()

    # Dropping the team 1
    # drop rows that any of the columns shortName, longName, category or fundFamily contain one of partial strings "china", "short", "bear", "inverse"
    discard = ["china", "short", "bear", "inverse"]
    df = df[~df.shortName.str.contains('|'.join(discard), case=False) & ~df.longName.str.contains('|'.join(discard), case=False) & ~df.category.str.contains('|'.join(discard), case=False) & ~df.fundFamily.str.contains('|'.join(discard), case=False)]

    # Assuming 'df' is your DataFrame with ETF data

    # Select ETFs with the lowest prices relative to the past 52-week range
    if not top:
        relevant_price_etfs = df[df['price52WRangePct'] <= threshold_value]
    else:
        relevant_price_etfs = df[df['price52WRangePct'] >= threshold_value]

    # Load pre-trained word embeddings model
    nlp = spacy.load('en_core_web_md')

    # Define your stop word list
    custom_stop_words = ["etf", "proshares", "global", "china", "msci", "ishares", "trading", "funds", "trust", "miscellaneous", "index", "broad", "vaneck", "x"]

    # Add custom stop words to spaCy's stop words
    for stop_word in custom_stop_words:
        nlp.vocab[stop_word].is_stop = True

    # Text data analysis
    text_data = relevant_price_etfs['shortName'].str.lower() + relevant_price_etfs['longName'].str.lower() + ' ' + relevant_price_etfs['category'].str.lower() + ' ' + relevant_price_etfs['fundFamily'].str.lower()

    # Tokenize and get word embeddings while ignoring stop words
    word_embeddings = []
    feature_names = []

    # Tokenize and get word embeddings while ignoring stop words
    for text in text_data:
        doc = nlp(text)
        current_embeddings = []
        current_feature_names = []

        for token in doc:
            if token.text not in nlp.Defaults.stop_words and token.text.lower() not in custom_stop_words:
                current_embeddings.append(token.vector)
                current_feature_names.append(token.text)

        # Append only if non-empty
        if current_embeddings:
            word_embeddings.append(current_embeddings)
            feature_names.append(current_feature_names)

    # Flatten the feature names and embeddings
    feature_names = [word for sublist in feature_names for word in sublist]
    word_embeddings = [emb for sublist in word_embeddings for emb in sublist]

    # Summarize the importance of words
    word_importance = pd.DataFrame({
        'Word': feature_names,
        'Importance': word_embeddings  # Flatten the list of vectors
    })

    # Sum the importance values for each word
    top_words = word_importance.groupby('Word')['Importance'].apply(lambda x: sum(map(sum, x))).reset_index()

    # Display all words by importance
    top_words = top_words.sort_values(by='Importance', ascending=False)
    print(top_words)

    # Create a word cloud
    wordcloud = WordCloud(width=1200, height=600, background_color='white').generate_from_frequencies(dict(zip(top_words['Word'], top_words['Importance'])))

    # Plot the WordCloud image
    plt.figure(figsize=(20, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()


def main():
    aparser = argparse.ArgumentParser(description="ETF Trend Analyzer - extract text feature by analyzing what text classification / feature does most ETFs within this range of the price relative to past 52 week price range share", epilog="")
    aparser.add_argument("--loglevel", dest="loglevel", type=int, action="store",
                      help="loglevel")
    aparser.add_argument("-r", "--datafile_root", dest="datafile_root", type=str, action="store",
                      help="the output csv file root to write history data into, individual file will be asin_<datatype>.csv")
    aparser.add_argument("-i", "--inputfile", dest="inputfile", type=str, action="store",
                      help="the input file that contains all tickers to be processed")
    aparser.add_argument("-t", "--threshold_value", dest="threshold_value", type=float, action="store",
                      help="the threshold value to filter ETFs price52WRangePct")
    aparser.add_argument("-u", "--top", dest="top", type=bool, action="store",
                         help="True for top part ETFs, False for bottom part ETFs")
    aparser.add_argument("-w", "--word_count", dest="word_count", type=int, action="store", default=20,
                      help="the number of top words to display in the word cloud")

    args, extras = aparser.parse_known_args()

    if args.inputfile is None:
        aparser.print_help()
        return

    FORMAT = "%(asctime)s - {%(pathname)s:%(lineno)s} - %(levelname)s - %(message)s"
    logging.basicConfig(format=FORMAT, level=args.loglevel or logging.INFO)

    logging.getLogger(__name__).setLevel(args.loglevel or logging.INFO)

    # trend_study_tfidf(args.inputfile, word_count=args.word_count, threshold_value=args.threshold_value or 0.20, top=args.top)
    trend_study_embeddings(args.inputfile, word_count=args.word_count, threshold_value=args.threshold_value or 0.20, top=args.top)

if __name__ == '__main__':
    main()