import sys
import argparse
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

parser = argparse.ArgumentParser(description='Processes the data.')

parser.add_argument('messages_filepath', action='store', metavar="['/path/to/messages_data_file.csv']",help='Provide the location of the messages .csv file')
parser.add_argument('categories_filepath', action='store', metavar="['/path/to/categories_data_file.csv']",help='Provide the location of the categories .csv file')
parser.add_argument('database_filepath', action='store', metavar="['/path/to/database.db']",help='Provide the location of the database where the cleaned data file will be stored')

def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df_merged = messages.merge(categories)
    return df_merged

def clean_data(df):
    # concat
    cats = df['categories'].str.split(pat = ';', expand = True)
    cats.columns =  cats.loc[0].apply(lambda x: x.replace(x[-2:],''))
    df = df.drop('categories', axis = 1)
    df = pd.concat([df, cats], axis = 1)

    # clean df
    df.dropna(how = 'all', axis = 1, inplace = True)
    df.dropna(how = 'all', axis = 0, inplace = True)

    for col in df[cats.columns]:
        df[col] = np.where(df[col].str.contains('1'), 1, 0)
        # if df[col].sum() == 0:
        #     df.drop(col, axis = 1, inplace = True)
        # if df[col].sum() == 1:
        #     df.drop(col, axis = 1, inplace = True)

    print("{} rows have duplicates and will be deleted.".format(df.duplicated().sum()))
    df.drop_duplicates(inplace=True)

    # drop rows that are completely empty
    df.dropna(how = 'all', axis = 0, inplace=True)

    return df


def save_data(df, database_filepath):
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df.to_sql('data', engine, index=False)

def main():
    if len(sys.argv) == 4:

        #messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        args = parser.parse_args()

        messages_filepath = args.messages_filepath
        categories_filepath = args.categories_filepath
        database_filepath = args.database_filepath

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')

if __name__ == '__main__':
    main()