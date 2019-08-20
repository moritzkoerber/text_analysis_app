import sys
import argparse
import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_evaluation import train_test_split

parser = argparse.ArgumentParser(description='Processes the data.')

parser.add_argument('database_filepath', action='store', metavar="['/path/to/database.db']",help='Provide the location of the database')
parser.add_argument('model_filepath', action='store', metavar="['/path/to/model']",help='Provide the destination of the produced pickle file')

def load_data(database_filepath):
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('data', engine)
    X = df['message']
    Y = df.drop(['id', 'message', 'original'], axis = 1)
    category_names = Y.columns
    return X, Y, category_names



def tokenize(text):
    pass


def build_model():
    pass


def evaluate_model(model, X_test, Y_test, category_names):
    pass


def save_model(model, model_filepath):
    pass

def main():
    if len(sys.argv) == 3:
        #database_filepath, model_filepath = sys.argv[1:]
        args = parser.parse_args()

        database_filepath = args.database_filepath
        model_filepath = args.model_filepath

        print('Loading data...\n    DATABASE: {}'.format(database_filepath))

        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2 , random_state = 1337)
        
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