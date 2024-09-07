# import libraries
import sys
from sqlalchemy import create_engine
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import FeatureUnion
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from custom_transformer import StartingVerbExtractor
import xgboost as xgb
import pickle
# download necessary NLTK data
import nltk
nltk.download(['punkt', 'wordnet'])


def load_data(database_filepath):

    '''
        INPUT - database_filepath: The DB filepath of the sqlite database that has our data
        OUTPUT - X: Model data input
                 y: Model data output
                 category_names: names of output categories
    '''

    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('disaster_response', engine)

    # drop columns with non binary values
    valid_vals = [0, 1]
    df = df[df['related'].isin(valid_vals)]
    X = df['message']
    y = df.iloc[:,4:]
    category_names = y.columns.values

    return X, y, category_names


def tokenize(text):
    tokens = word_tokenize(text)
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        
        # lemmatize, normalize case, and remove leading/trailing white space
        lemmed = lemmatizer.lemmatize(tok)
        clean_tok = lemmed.lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens 


def build_model():
    model = Pipeline([
    ('features', FeatureUnion([

        ('text_pipeline', Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer())
        ])),
        ('starting_verb', StartingVerbExtractor())
    ])),
    ('clf', xgb.XGBClassifier())

])
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(Y_test.values, y_pred))
    print("Classification Report:\n", classification_report(Y_test.values, y_pred, target_names=category_names))

def save_model(model, model_filepath):
    filename = 'classifier.pkl'
    pickle.dump(model_filepath, open(model_filepath, 'wb'))
    
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