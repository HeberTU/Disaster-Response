import json
import plotly
import pandas as pd
import re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Pie
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

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

    # lower case text
    text = text.lower()

    # Remove stop words and puntuation
    tokens = word_tokenize(text)
    tokens = [re.sub(r"[^a-z0-9]", "", word) for word in tokens if word not in stopwords.words("english")]
    tokens = [word for word in tokens if word != '']

    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        tok = tok.strip()
        clean_tok = lemmatizer.lemmatize(lemmatizer.lemmatize(tok), pos='v')
        clean_tokens.append(clean_tok)

    return clean_tokens


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    columns = df.drop(
        labels=['id', 'message', 'original', 'genre'],
        axis=1).columns

    categories_freq = df[columns].sum() / df.shape[0]
    categories_freq.sort_values(
        ascending=False,
        inplace=True)

    values = categories_freq.values.tolist()
    complement = list(map(lambda x: 1 - x, values))

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        # Graph 1
        {
            'data': [
                Pie(
                    labels=genre_names,
                    values=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        # graph 2
        {
            'data': [
                Bar(
                    x=categories_freq.index.tolist(),
                    y=values,
                    name='% Obs. Cases',
                    marker=dict(
                            color='rgb(255, 193, 37)'
                                )
                ),
                Bar(
                    x=categories_freq.index.tolist(),
                    y=complement,
                    name='',
                    hoverinfo='none',
                    marker=dict(
                            color='rgb(255, 193, 37)',
                            opacity=0.3
                                )
                )
            ],

            'layout': {
                'title': 'Distribution of Message by Category',
                'yaxis': {
                    'title': "frequency"
                },
                'xaxis': {
                    'title': "Category"
                },
                'barmode' : 'stack'
            }
        }


    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(debug=True)


if __name__ == '__main__':
    main()