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
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
import pickle
from sklearn.metrics import accuracy_score, f1_score

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('maxent_ne_chunker')

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
    # text = list(set(words) - set(stopwords.words("english")))
    return text


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = [
        {"clf": [RandomForestClassifier()],
         "clf__n_estimators": [10, 100, 250],
         "clf__max_depth":[8],
         "clf__min_samples_leaf":[5,10],
         "clf__random_state":[42]},
        {"clf": [LinearSVC()],
         "clf__C": [1.0, 10.0, 100.0, 1000.0],
         "clf__random_state":[42]},
        {"clf": [GaussianNB()]}
    ]

    n_iter_search = 1

    rkf = RepeatedKFold(
        n_splits=5,
        n_repeats=1,
        random_state=1337
    )

    cv = GridSearchCV(
        pipeline,
        parameters,
        # n_iter=n_iter_search,
        cv=rkf,
        scoring='accuracy',
        n_jobs=-1)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    pass


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        # database_filepath, model_filepath = sys.argv[1:]
        args = parser.parse_args()

        database_filepath = args.database_filepath
        model_filepath = args.model_filepath

        print('Loading data...\n    DATABASE: {}'.format(database_filepath))

        X, Y, category_names = load_data(database_filepath)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X, Y)

        print('Evaluating model...')
        #evaluate_model(model, X_test, Y_test, category_names)

        print("Best score: %0.3f" % model.best_score_)
        print("Best parameters set:{}".format(
            print(model.best_estimator_.get_params()["clf"])))

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
