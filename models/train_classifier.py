from sklearn.naive_bayes import GaussianNB
import sklearn.svm
from sklearn.pipeline import Pipeline
import sys
import argparse
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
import string
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import RandomizedSearchCV

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('maxent_ne_chunker')

parser = argparse.ArgumentParser(description='Processes the data.')

parser.add_argument('database_filepath', action='store', metavar="['/path/to/database.db']",
                    help='Provide the location of the database')
parser.add_argument('model_filepath', action='store', metavar="['/path/to/model']",
                    help='Provide the destination of the produced pickle file')


def load_data(database_filepath):
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('data', engine)
    X = df['message']
    Y = df.drop(['id', 'message', 'original'], axis=1)
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
        ('clf', RandomForestClassifier()),
    ])

    parameters = [{
        [{"clf": [RandomForestClassifier()],
          # "clf__penalty": ['l1', 'l2'],
          # "classifier__C": np.logspace(0, 4, 10)
          },
         [{"clf": [sklearn.svm.LinearSVC()],
           # "classifier__penalty": ['l1', 'l2'],
           # "classifier__C": np.logspace(0, 4, 10)
           },
          [{"clf": [GaussianNB()],
            # "classifier__penalty": ['l1', 'l2'],
            # "classifier__C": np.logspace(0, 4, 10)
            },
           'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        'features__text_pipeline__vect__max_df': (0.5, 0.75, 1.0),
        'features__text_pipeline__vect__max_features': (None, 5000, 10000),
        'features__text_pipeline__tfidf__use_idf': (True, False),
        'clf__n_estimators': [50, 100, 200],
        'clf__min_samples_split': [2, 3, 4],
        'features__transformer_weights': (
            {'text_pipeline': 1, 'starting_verb': 0.5},
            {'text_pipeline': 0.5, 'starting_verb': 1},
            {'text_pipeline': 0.8, 'starting_verb': 1},
        )
    }]
    # param_dist = {"max_depth": [3, None],
    #               "max_features": sp_randint(1, 11),
    #               "min_samples_split": sp_randint(2, 11),
    #               "bootstrap": [True, False],
    #               "criterion": ["gini", "entropy"]}

    n_iter_search= 20

    rskf= RepeatedStratifiedKFold(n_splits=10, n_repeats=2, random_state=1337, shuffle = True)

    cv= RandomizedSearchCV(pipeline, param_grid=parameters, n_iter=n_iter_search, cv=rskf)

    cv.fit(data.data, data.target)


def evaluate_model(model, X_test, Y_test, category_names):
    pass


def save_model(model, model_filepath):
    pass


def main():
    if len(sys.argv) == 3:
        # database_filepath, model_filepath = sys.argv[1:]
        args = parser.parse_args()

        database_filepath = args.database_filepath
        model_filepath = args.model_filepath

        print('Loading data...\n    DATABASE: {}'.format(database_filepath))

        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1337)

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
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
