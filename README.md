This repository contains a web app that classifies the content of messages that are usually sent during disasters such as earthquakes. Labels are, for example, 'medical' or 'request'.

To run the app, run `./app/run.py`

The required packages are and their corresponding version are stored in ./requirements.txt

If you use this app on your local machine, you can visit it at localhost...

Run ./data/process_data.py to preprocess the data. Please provide the filepaths of the messages and categories datasets as the first and second argument respectively, as well as the filepath of the database to save the cleaned data to as the third argument. Example: `python ./data/process_data.py './data/disaster_messages.csv' './data/disaster_categories.csv' './data/database.db'`

Run ./models/train_classifier.py to retrain the model. Please provide the filepath of the disaster messages database as the first argument and the filepath of the pickle file to save the model to as the second argument. Example: `python ./models/train_classifier.py './data/database.db' './models/model.pkl'`

This app has been built on top of files provided by[Udacity](http://udacity.com) that were modified by me. The process_data.py and train_classifier.py were written by me. 

2019-08-30 Moritz Koerber 
CC-