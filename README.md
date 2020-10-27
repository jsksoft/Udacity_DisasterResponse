# Udacity_DisasterResponse
Analyzing disaster messages using a machine learning model via a web app <br>
Udacity Nanodegree: Date Science

## Objective
In this project, I built a model for an API that classifies disaster messages. I used a dataset of real disaster messages from Figure Eight (https://www.figure-eight.com/) that were sent during disaster events to train the model. My machine learning (ML) pipeline using NLTK, scikit-learn and GridSearchCV categorizes these messages to be able to analyze and categorize new, so far unknown messages. A web app will be the interface between a potential emergency worker entering the message and the machine learning model to categorize the message for easier and faster help. 

## Instructions
1. Set up the below listed file and folder structure

2. Run the following command in the project's root directory to set up the database
    - To run the ETL pipeline that cleans the data and store them in the database file data/DisasterResponse.db<br>
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

3. Run the following command in the project's root directory to set up the machine learning model.
    - To run the ML pipeline that trains the classifier and saves it to models/classifier.pkl <br>
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

4. Run the following command in the app's directory to run your web app. <br>
    `python run.py`

## Libraries
The following python libraries are used for the project:
- sys
- pandas
- sqlalchemy
- re
- nltk
- sklearn
- pickle

## Added Jupyter Notebooks
- ETL Pipeline Preparation.ipynb: This notebook has been used to develop the code for ETL pipeline (Extract, Transform, and Load process)<br>
- ML Pipeline Preparation.ipynb: This notebook has been used to develop the code for machine learning pipeline (split the data, build the machine learning model, traing and evaluate the model)<br>

## File structure
The python files for the web app and the model file classifier.pkl are structured in the following way
- app<br>
| - template<br>
| |- master.html  # main page of web app<br>
| |- go.html      # classification result page of web app<br>
|- run.py         # Flask file that runs app<br>
<br>
- data<br>
|- disaster_categories.csv  # data to process (not included due to license issues)<br>
|- disaster_messages.csv    # data to process (not included due to license issues)<br>
|- process_data.py          # python file which performs the ETL pipeline (Extract, Transform, and Load process)<br>
|- DisasterResponse.db      # database file containing the cleaned messages and categories (not included due to license issues)<br>
<br>
- models<br>
|- train_classifier.py      # python file which performs the machine learning pipeline (split the data, build the machine learning model, traing and evaluate the model)<br>
|- classifier.pkl           # saved ML model <br>

## additional programs necessary to run the web app
Flask is used to use the web app.
