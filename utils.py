#!/usr/bin/env python
# coding: utf-8

# In[4]:


#!/usr/bin/env python
# coding: utf-8

# Data Wrangling
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import twint
import yfinance as yf
from datetime import datetime as dt
from datetime import timedelta
from datetime import date

# EDA
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
import spacy
import re
import html
from bs4 import BeautifulSoup
import unicodedata
from wordcloud import WordCloud
get_ipython().run_line_magic('matplotlib', 'inline')
from textblob import TextBlob, Blobber
from spacy import displacy
from IPython.display import display_html
from itertools import chain,cycle
import scipy.stats as stats


from textblob.sentiments import NaiveBayesAnalyzer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import feature_selection
from sklearn.linear_model import SGDClassifier, LogisticRegression, LogisticRegressionCV
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, cross_val_score
import matplotlib.pyplot as plt
import twint
from bs4 import BeautifulSoup
import re
import yfinance as yf
from nltk.tokenize import WordPunctTokenizer
from datetime import datetime as dt
from datetime import timedelta
import demoji
from datetime import date
import contractions
import wikipedia
from sklearn.pipeline import make_pipeline
import contractions
nlp = spacy.load('en_core_web_lg')
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline




def get_text_features(df, text_col):
    
    def average_word_length(text):
        words = text.split()
        word_length = 0
        for word in words:
            word_length += len(word)
        return word_length / len(words)
    
    pattern_email = r'([a-zA-Z0-9+._-]+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9_-]+)'
    
    df['word counts'] = df[text_col].apply(lambda x: len(str(x).split())) # word counts
    df['character counts'] = df[text_col].apply(lambda x: len(x)) # character counts
    df['average word length'] = df[text_col].apply(lambda x: average_word_length(x)) #avg word length
    df['stop word counts'] = df[text_col].apply(lambda x: len([t for t in x.split() if t in STOP_WORDS])) #stopword
    df['text length'] = df['cleaned'].apply(lambda x: len(x))
    #df['mention counts'] = df['cleaned'].apply(lambda x: len([t for t in x.split() if t.startswith('@')]))
    #df['url counts'] = df['cleaned'].apply(lambda x: len([t for t in x.split() if t.startswith('http')]))
    df['number counts'] = df[text_col].apply(lambda x: len([t for t in x.split() if t.isdigit()])) # numbers
    df['upper counts'] = df[text_col].apply(lambda x: len([t for t in x.split() if t.isupper() and len(x) > 3]))
    df['email counts'] = df[text_col].apply(lambda x: len(re.findall(pattern_email, x))) #number of emails


def clean_text(text_col, preserve_syntax=False, remove_hashtags=True, stop_words=STOP_WORDS):
       
    def remove_accents(text):
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        return text

    def make_to_base(text):
        x_list = []
        doc = nlp(text)
    
        for token in doc:
            lemma = str(token.lemma_)
            x_list.append(lemma)
        return " ".join(x_list)

    def remove_junk_words(text_col):
        all_text = " ".join(text_col)
        freq_words = pd.Series(all_text.split()).value_counts()
        words = all_text.split()
        junk_words = [word for word in words if len(word) <= 2]
        text_col = " ".join([t for t in text_col.split() if t not in junk_words])
        rare = freq_words[freq_words.values == 1]
        text_col = " ".join([t for t in text_col.split() if t not in rare])
        
        return text_col
        
    pattern_email = r'([a-zA-Z0-9+._-]+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9_-]+)'    

    # See about adding a period if links, mentions or hashtags are followed by a word that is capitalized.
    text = re.sub(r"(https?|ftp)\S+", '.', text_col) # Remove links
    text = " ".join([contractions.fix(word) for word in text.split()])
    text = BeautifulSoup(text, 'lxml').get_text() # Remove HTML tags
    text = html.unescape(text) # Remove HTML encoding
    text = remove_accents(text) # Remove accented characters
    text = re.sub(pattern_email, '', text)
    text = re.sub(r"@\S+", '', text) # Remove @mentions (period if capital followed?)
    if remove_hashtags == True:
        text = re.sub(r"#\S+", '', text) # Remove #hashtags
    else:
        text = re.sub(r"#", '', text) #  #hashtags
    text = " ".join(text.split()) # Remove redundant white space
    text = re.sub(r'\.+', ".", text)
    text = re.sub(r'\s([?.!"](?:\s|$))', r'\1', text) 
     #https://stackoverflow.com/questions/18878936/how-to-strip-whitespace-from-before-but-not-after-punctuation-in-python
    text = re.sub(r'^\.?', '', text)
    text = re.sub('[^A-Z a-z 0-9 .?!,]+', '', text) # Remove special characters
    text = re.sub(r'\.{2,3}', '.', text)
    text = re.sub(r'(\. \. )', '.', text)
    text = re.sub(r'\.\s\.', '.', text)
    text = re.sub(r'!\.', '!', text)
    text = re.sub(r'[.!]{2,3}', '.', text)
    text = " ".join(text.split()) # Remove redundant white space
    if preserve_syntax == True:
        return text
    else:
        text = text.lower() # Normalize capitalization
        text = re.sub('[0-9]+', '', text) # Remove numbers
        text = re.sub('[^a-z ]+', '', text)# Remove all special characters
        text = " ".join([t for t in text.split() if t not in STOP_WORDS]) #stopwords
        text = remove_junk_words(text) # Remove short/rare words
        text = " ".join(text.split()) # Remove redundant white space
        text = make_to_base(text) # Lemmatize
        text = remove_junk_words(text)
        return text
    
    
def make_wordcloud(text):
    from wordcloud import WordCloud
    all_text = " ".join(text)
    wc = WordCloud(width = 1000, height=800).generate(all_text)
    plt.imshow(wc)
    plt.axis('off')
    plt.show()


def combine_tweets(df):
    """
    takes in a twint dataframe and returns a dataframe with each row a combination of
    tweets from that day.
    """
    username = df.username
    collected_tweets = {}
    #df['time'] = pd.to_datetime(df['time'])
    #df['number of tweets'] = 1

    
    # If tweet is earlier than 9:30, it applies to that price (opening) on the same date. 

    df['time'] = pd.to_datetime(df['time'], format='%H:%M:%S').dt.time
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d').dt.date

    for i in range(len(df)):
        if df['time'].iloc[i] < dt.strptime('09:30:00', '%H:%M:%S').time():
            df['time'].iloc[i] = dt.strptime('09:30:00', '%H:%M:%S').time()
        
    # It tweet is after 9:30, but before 16:00 (closing), it applies to the following price on the same date.
    
        if (df['time'].iloc[i] > dt.strptime('09:30:00', '%H:%M:%S').time()) and (df['time'].iloc[i] < dt.strptime('16:00:00', '%H:%M:%S').time()):
            df['time'].iloc[i] = dt.strptime('16:00:00', '%H:%M:%S').time()
            
    # If tweet is after 16:00, apply it to the next opening date.
    #for i in range(len(df)):
        if df['time'].iloc[i] > dt.strptime('16:00:00', '%H:%M:%S').time():
            df['date'].iloc[i] = df['date'].iloc[i] + timedelta(days=1)
            df['time'].iloc[i] = dt.strptime('09:30:00', '%H:%M:%S').time()
            
    # Combine dates and times
    df['date'] = df['date'].astype(str)
    df['time'] = df['time'].astype(str)
    df['date'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    
    tweet = ""
    df['number of tweets'] = 1
    to_merge = df.groupby('date').sum()
    #to_merge['date'] = pd.to_datetime(to_merge['date'], format='%Y-%M-%d').dt.date
    # date is dictionary key
    collected_tweets[df['date'].iloc[0]] = tweet
   
    for i in range(len(df.index)):
        current_date = df['date'].iloc[i]
        if current_date in collected_tweets:
            collected_tweets[current_date] += " " + str(df['tweet'].iloc[i])
        else:
            collected_tweets[current_date] = str(df['tweet'].iloc[i])
            
    df = pd.DataFrame.from_dict(collected_tweets, orient='index', columns = ['tweet'])
    df.reset_index(inplace=True)
    df = df.rename(columns={'index':'date'})
    df['username'] = username
    df_merged = pd.merge(df, to_merge.reset_index(), on='date')
    
    return df_merged


def organize_stocks(stock):

        # Instatiate Open and Close
        stock_open = stock[['date','open']]
        stock_close = stock[['date','close']]

        # Convert dates to datetime objects
        stock_open['date'] = pd.to_datetime(stock_open['date'])
        stock_close['date'] = pd.to_datetime(stock_close['date'])

        # Convert datetimes into datetime string format
        stock_open['date'] = stock_open['date'].dt.strftime('%Y-%m-%d 09:30:00')
        stock_close['date'] = stock_close['date'].dt.strftime('%Y-%m-%d 16:00:00')

        # Convert strings back into datetime objects
        stock_open['date'] = pd.to_datetime(stock_open['date'])
        stock_close['date'] = pd.to_datetime(stock_close['date'])

        # Get earliest and latest stock price dates to create a date index
        stock_open['price'] = stock_open['open']
        stock_open.drop('open', axis=1, inplace=True)

        stock_close['price'] = stock_close['close']
        stock_close.drop('close', axis=1, inplace=True)

        start_date_open = dt.strftime(stock_open.reset_index().date.min(), '%Y-%m-%d %H:%M:%S')
        end_date_open = dt.strftime(stock_open.reset_index().date.max(), '%Y-%m-%d %H:%M:%S')

        start_date_close = dt.strftime(stock_close.reset_index().date.min(), '%Y-%m-%d %H:%M:%S')
        end_date_close = dt.strftime(stock_close.reset_index().date.max(), '%Y-%m-%d %H:%M:%S')

        date_indx_open = pd.date_range(start_date_open, end_date_open).tolist()
        date_indx_close = pd.date_range(start_date_close, end_date_close).tolist()
        date_indx_open = pd.Series(date_indx_open, name='date')
        date_indx_close = pd.Series(date_indx_close, name='date')

        # Merge date index onto stock dataframes
        stock_open = pd.merge(date_indx_open, stock_open, how='left')
        stock_close = pd.merge(date_indx_close, stock_close, how='left')

        # Interpolate missing values
        stock_open['price'].interpolate(method='linear', inplace=True)
        stock_close['price'].interpolate(method='linear', inplace=True)

        # Reset index and join open and close dataframes together
        stock_open.set_index('date', inplace=True)
        stock_close.set_index('date', inplace=True)

        stock = pd.concat([stock_open, stock_close])
        
        stock.sort_index(inplace=True)
        
        return stock



