import sys
import nltk
import time
import pickle
import re
import pandas as pd
import numpy as np

from sqlalchemy import create_engine
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

def load_data(database_filepath: str):
    """"
    load messages table from sqlite database and splits the table into:
        X: message str, training feature
        Y: target variables

    parameters
    -----------
    database_filename: str
        Database name using the following format 'sqlite:///<database name>.db'


    returns
    -------
    X: pd.DataFrame
    message str, training feature
    Y: pd.DataFrame
    target variables
    category_names: list
        list containing categories to be used
    """
    engine = create_engine(database_filepath)
    df = pd.read_sql_table("messages", engine)
    X = df['message'].copy()
    Y = df.drop(
        labels=['id', 'message', 'original', 'genre'],
        axis=1).copy()

    category_names = Y.columns

    return X, Y, category_names


def tokenize(text: str):
    """"
    Clean and tokenize the text data

    parameters
    -----------
    text: str
        Long string to be cleaned and tokenize

    returns
    -------
    clean_tokens: list
        list of cleaned word tokens

    """

    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    # get list of all urls using regex
    detected_urls = re.findall(url_regex, text)

    # replace each url in text string with placeholder
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    # tokenize text
    tokens = word_tokenize(text)

    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(lemmatizer.lemmatize(tok), pos='v')
        clean_tok = clean_tok.lower()
        clean_tok = clean_tok.strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """"
    Builds a Pipeline using:
        + CountVectorizer:
            tokenize(text: str)
        + TfidfTransformer
        + MultiOutputClassifier(RandomForestClassifier)


    parameters
    -----------
    None

    returns
    -------
    pipeline: Pipeline

    """

    pipeline = Pipeline(
        [('vect', CountVectorizer(tokenizer=tokenize)),
         ('tfidf', TfidfTransformer()),
         ('clf', MultiOutputClassifier(
             RandomForestClassifier(),
             n_jobs=1))])


    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """"
    Uses classification_report function to show precision, recall, and f1-score across all the available categories

    parameters
    -----------
    model: Pipeline
    X_test: pd.Series
    Series containing the test cases
    Y_test: pd.DataFrame
    DataFrame containing the correct categories.
    category_names: list
    list containing categories to be used



    returns
    -------
    None

    """

    Y_pred = model.predict(X_test)

    Y_pred = pd.DataFrame(
            data=Y_pred,
            columns=category_names,
            index=Y_test.index)

    for col in category_names:
        print(col)
        print(classification_report(
            y_true=Y_test[col],
            y_pred=Y_pred[col]))




def save_model(model, model_filepath):
    """"
    Saves the model as pickle file

    parameters
    -----------
    model: Pipeline
    model_filepath: model's file path


    returns
    -------
    None

    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()