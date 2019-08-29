from sklearn.pipeline import Pipeline
import sys
import argparse
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sqlalchemy import create_engine
import string
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, RepeatedKFold
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
import pickle
from sklearn.metrics import accuracy_score, f1_score
from nltk.stem.wordnet import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('maxent_ne_chunker')
nltk.download('wordnet')

parser = argparse.ArgumentParser(description='Processes the data.')

parser.add_argument(
    'database_filepath',
    action='store',
    metavar="['/path/to/database.db']",
    help='Provide the location of the database')

parser.add_argument(
    'model_filepath',
    action='store',
    metavar="['/path/to/model']",
    help='Provide the destination of the produced pickle file')

rm = set(stopwords.words("english"))


def load_data(database_filepath):
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('data', engine)
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower().strip()
    text = word_tokenize(text)
    text = list(set(text) - rm)
    #text = [SnowballStemmer("english").stem(w) for w in text]
    text = [WordNetLemmatizer().lemmatize(w) for w in text]
    return text


def build_model():
    pipeline = Pipeline([
        ('vecttext', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', RandomForestClassifier())
    ])

    parameters = [
        # {"clf": [RandomForestClassifier()],
        #  "clf__estimator__n_estimators": [1],
        #  "clf__estimator__max_depth": [8],
        #  "clf__estimator__random_state":[42]}
        {"clf": [OneVsRestClassifier(LinearSVC())],
         "clf__estimator__C": [1, 10],
         "clf__estimator__max_iter": [5000],
         #'vecttext__max_df': [0.5, 0.75, 1.0],
         #'vecttext__ngram_range': [(1, 1), (1, 2)],
         "clf__estimator__random_state": [42]
         },
        {"clf": [OneVsRestClassifier(MultinomialNB())],
         #'vecttext__max_df': [0.5, 0.75, 1.0],
         #'vecttext__ngram_range': [(1, 1), (1, 2)],
         "clf__estimator__random_state": [42]
         }
    ]

    n_iter_search = 20

    rkf = RepeatedKFold(
        n_splits=3,
        n_repeats=1,
        random_state=42
    )

    cv = GridSearchCV(
        pipeline,
        parameters,
        # n_iter=n_iter_search,
        cv=rkf,
        scoring='f1_micro',# ['f1_micro', 'f1_samples','roc_auc'],
        n_jobs=-1)

    return cv


def evaluate_model(model):
    print("Results:")
    print(model.cv_results_)
    print("Best parameters set:{}".format(model.best_estimator_.get_params()["clf"]))


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def remove_constants(Y):
    drops = []
    for col in Y.columns:
        if len(np.unique(Y[col])) < 2:
            drops.append(col)
    Y.drop(drops, axis=1, inplace=True)


def main():
    if len(sys.argv) == 3:
        args = parser.parse_args()

        database_filepath = args.database_filepath
        model_filepath = args.model_filepath

        print('Loading data...\n    DATABASE: {}'.format(database_filepath))

        X, Y, category_names = load_data(database_filepath)

        remove_constants(Y)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X, Y)

        print('Evaluating model...')
        evaluate_model(model)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
