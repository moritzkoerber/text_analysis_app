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
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
import pickle
from nltk.stem.wordnet import WordNetLemmatizer
#from skmultilearn.problem_transform import ClassifierChain
from sklearn.compose import ColumnTransformer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('maxent_ne_chunker')
nltk.download('wordnet')

parser = argparse.ArgumentParser(description='Processes the data.')

parser.add_argument(
    'database_filepath',
    action='store',
    metavar='["/path/to/database.db"]',
    help='Provide the location of the database')

parser.add_argument(
    'model_filepath',
    action='store',
    metavar='["/path/to/model"]',
    help='Provide the destination of the produced pickle file')

rm = set(stopwords.words('english'))


def load_data(database_filepath):
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('data', engine)
    X = df[['message', 'len']]
    Y = df.drop(['id', 'message', 'original', 'len',
                 'genre_news', 'genre_social'], axis=1)
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower().strip()
    text = word_tokenize(text)
    text = list(set(text) - rm)
    #text = [SnowballStemmer('english').stem(w) for w in text]
    text = [WordNetLemmatizer().lemmatize(w) for w in text]
    return text


text_transformer = Pipeline([
    ('vecttext', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer())
])

preprocessor = ColumnTransformer(
    [('text', text_transformer, 'message')], remainder='passthrough')


def build_model():
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('clf', RandomForestClassifier())
    ])

    parameters = [
        # {'clf': [RandomForestClassifier()],
        #  'clf_n_estimators': [5, 50, 100, 250],
        #  'clf__max_depth': [5, 8, 10],
        #  'preprocessor__text__vecttext__max_df': [0.5, 0.75, 1.0],
        #  'preprocessor__text__vecttext__ngram_range': [(1, 1), (1, 2)],
        #  'clf__random_state':[42]
        #  },
        {'clf': [MultiOutputClassifier(LinearSVC())],
         'clf__estimator__C': [1.0, 10.0, 100.0, 1000.0],
         'clf__estimator__max_iter': [5000],
         # 'preprocessor__text__vecttext__max_df': [0.5, 0.75, 1.0],
         # 'preprocessor__text__vecttext__ngram_range': [(1, 1), (1, 2)],
         'clf__estimator__random_state': [42]
         },
        {'clf': [MultiOutputClassifier(MultinomialNB())]
         #'preprocessor__text__vecttext__max_df': [0.5, 0.75, 1.0],
         #'preprocessor__text__vecttext__ngram_range': [(1, 1), (1, 2)]
         }
    ]

    rkf = RepeatedKFold(
        n_splits=3,
        n_repeats=1,
        random_state=42
    )

    cv = GridSearchCV(
        pipeline,
        parameters,
        cv=rkf,
        scoring=['f1_weighted', 'f1_micro', 'f1_samples'],
        refit='f1_weighted',
        n_jobs=-1)

    return cv


def evaluate_model(model):
    print('Best score:{}'.format(model.best_score_))
    print('Best parameters set:{}'.format(
        model.best_estimator_.get_params()['clf']))
    df = pd.DataFrame.from_dict(model.cv_results_)
    print('mean_test_f1_weighted:{}'.format(df['mean_test_f1_weighted']))
    print('mean_test_f1_micro:{}'.format(df['mean_test_f1_micro']))
    print('mean_test_f1_micro:{}'.format(df['mean_test_f1_micro']))
    print('mean_test_f1_samples:{}'.format(df['mean_test_f1_samples']))
    # print('mean_test_roc_auc:{}'.format(df['mean_test_roc_auc']))


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)

def main():
    if len(sys.argv) == 3:
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
