# Classifying disaster response messages

This repository contains a web app that classifies the content of messages that are usually sent during disasters such as earthquakes. Labels are, for example, *medical* or *request*.

## How to use the app
 
To run the app, run './app/run.py'. If you use this app on your local machine, you can visit it at [http://localhost:3001/](http://localhost:3001/). Run './data/process_data.py' to preprocess the data. Please provide the file paths of the messages and categories datasets as the first and second argument respectively, as well as the filepath of the database to save the cleaned data to as the third argument. Example: `python ./data/process_data.py './data/disaster_messages.csv' './data/disaster_categories.csv' './data/database.db'`. The predictive model has been Run './models/train_classifier.py' to retrain the model. Please provide the filepath of the disaster messages database as the first argument and the filepath of the pickle file to save the model to as the second argument. Example: `python ./models/train_classifier.py './data/database.db' './models/model.pkl'`.

## Folder structure

'./data/' contains the underlying data in .csv-format. './models/' contains the predictive model in .pkl-format. './app' contains all files of the app itself.  

## Requirements

The required packages are and their corresponding version are stored in './requirements.txt'.

## Further information

This app has been built on top of material provided by [Udacity](http://udacity.com), with modifications from my side. The process_data.py and train_classifier.py were written by myself; feel free to use it or develop it further.