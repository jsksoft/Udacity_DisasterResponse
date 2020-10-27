import sys
import pandas as pd
from sqlalchemy import create_engine
import re

import nltk
nltk.download(['punkt', 'wordnet'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle

def load_data(database_filepath):
    """Load the data from database

    Parameters:
    database_filepath: sqllite file containing the clean dataset, the file is created by process_data.py

    Returns:
    X: input data containing sent messages
    Y: categories to be predicted
    category_names: names of the categories saves in variable Y
    """
    # load data from database
    database_filepath = 'sqlite:///{}'.format(database_filepath)
    engine = create_engine(database_filepath)
    # read data using pandas read_sql
    df = pd.read_sql("SELECT * FROM DisasterResponseMessages", engine)
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis = 1)
    # get category names which are the columns of Y
    category_names = Y.columns.values
    # return values
    return X, Y, category_names

def tokenize(text):
    """Tokenize input text

    Parameters:
    text: input text

    Returns:
    clean_tokens: list containing the normalized, tokenized and lemmazied words
    """
    # remove punctuation and use lowercase letters
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # tokentize words
    tokens = word_tokenize(text)
    # create word lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # lemmatize words
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """Build machine learning model using pipeline and GridSearch

    Returns:
    model: machine learning model with best parameters using GridSearch
    """
    # create pipeline
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    # parameter dictionary
    parameters =  {
        'vect__max_df': (0.5, 1.0),
        'tfidf__use_idf': (True, False),
        'clf__estimator__max_features': ['auto', 'sqrt'],
        'clf__estimator__max_depth': [5,10, 20,None]
        }
    # use GridSearchCV for finding optimal parameters
    model = GridSearchCV(pipeline, param_grid=parameters, verbose=10)
    # return model
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluate the machine learning model using X_test, Y_test and category_names

    Parameters:
    model: machine learning model with best parameters using GridSearch
    X_test: X values of the test subset
    Y_test: Y values of the test subset
    category_names: category names
    """
    # predict results
    Y_pred = model.predict(X_test)

    # print report using sklearn classification_report
    print('\n', classification_report(Y_test, Y_pred, target_names=category_names))


def save_model(model, model_filepath):
    """Save the machine learning model to a pickle file

    Parameters:
    model: machine learning model with best parameters using GridSearch
    model_filepath: filename the model is saved to
    """
    pickle.dump(model, open(model_filepath, 'wb'))


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