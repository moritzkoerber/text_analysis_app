import json
import plotly
import pandas as pd
import numpy as np
import string
import nltk
from nltk.corpus import stopwords

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Histogram
from sklearn.externals import joblib
from sqlalchemy import create_engine

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('maxent_ne_chunker')
nltk.download('wordnet')

rm = set(stopwords.words('english'))

app = Flask(__name__) #template_folder='templates'

def tokenize(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower().strip()
    text = word_tokenize(text)
    text = list(set(text) - rm)
    text = [WordNetLemmatizer().lemmatize(w) for w in text]
    return text

# load data
engine = create_engine('sqlite:///../data/database.db')
df = pd.read_sql_table('data', engine)

# load model
model = joblib.load("../models/model.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    df['genre_direct'] = np.where(df[['genre_social', 'genre_news']].sum(axis=1) == 0, 1, 0)
    genre_counts = df[['genre_social', 'genre_news', 'genre_direct']].sum()
    genre_names = ['genre_social', 'genre_news', 'genre_direct']
    
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
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
        {
            'data': [
                Histogram(
                    x=df['len']
                )
            ],

            'layout': {
                'title': 'Distribution of Message Lengths',
                'xaxis': {
                    'title': "Length of Message"
                }
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
    length = len(query)
    input = pd.DataFrame({'message':query,'len':length}, index=[1])

    # use model to predict classification for query
    classification_labels = model.predict(input)[0]
    classification_results = dict(zip(df.columns[3:-3], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()