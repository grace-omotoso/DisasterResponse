import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
  '''
  INPUT - messages_filepath: the filepath of the file that contains messages
        - categories_filepath: the filepath of the file that contains categories

  OUTPUT - df - a dataframe that contains a merge of messages and categories
  '''
  messages = pd.read_csv(messages_filepath)
  categories = pd.read_csv(categories_filepath)

  # merge messages and categories using id
  df = pd.merge(messages, categories, on='id', sort=True)

  return df

def process_categories(categories):
  '''
  INPUT - categories:  A multi column dataframe that contains unprocessed categories of messages
  OUTPUT - categories: A multicolumn dataframe that contains binary processed categories of messages
  '''
  # select the first row of the categories dataframe
  row = categories.iloc[0]
  # use this row to extract a list of new column names for categories.
  # one way is to apply a lambda function that takes everything 
  # up to the second to last character of each string with slicing
  category_colnames = row.apply(lambda x: x[:-2])
  categories.columns = category_colnames

  #Iterate through the category columns in df to keep only the last 
  #character of each string (the 1 or 0).
  for column in categories:
    # set each value to be the last character of the string
    categories[column] = categories[column].astype(str).str[-1]
    # convert column from string to numeric
    categories[column] = categories[column].astype(int)

  return categories

def clean_data(df):
  '''
  INPUT - df: dataframe that contains messages and categories
       
  OUTPUT - df_clean: a new dataframe that has been cleaned and ready for our model
  '''

  #Split the values in the categories column on the ; character so 
  #that each value becomes a separate column
  categories = df['categories'].str.split(pat=';', expand=True)
  categories = process_categories(categories)
  
  # drop the original categories column from `df`
  df = df.drop(columns=["categories"])
  # concatenate the original dataframe with the new `categories` dataframe
  df = pd.concat([df, categories], axis=1, sort=False)

  # drop columns with non binary values
  valid_vals = [0, 1]
  df = df[df['related'].isin(valid_vals)]

  # drop duplicate
  df = df.drop_duplicates()

  return df


def save_data(df, database_filename):
  '''
    INPUT - database_filename: the name of sqlite database file 
          - df: the dataframe to load to sql
  '''
  engine = create_engine('sqlite:///'+database_filename)
  df.to_sql('disaster_response', engine, if_exists='replace', index=False)  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

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