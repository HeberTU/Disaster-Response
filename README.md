# Disaster Response Pipeline Project

This project provides a web interface to interact in real-time with a Natural Language Processing (NLP) model that aims to categorize messages from real-life disaster events into 36 categories.  

![image](https://user-images.githubusercontent.com/28582065/108807124-dcd50880-75a3-11eb-874e-48cff81bc407.png)


## Getting Started

### Dependencies

* Python 3.6+
* Pandas 1.2.1
* Sqlalchemy 1.3.23
* nltk 3.5
* scikit-learn 0.23.2
* flask 1.1.2

### Installing

#### Clone the github repository

` git clone https://github.com/HeberTU/Disaster-Response.git`

#### Execution Programs

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


## License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


## Acknowledgements

This Project is part of Data Science Nanodegree Program by [Udacity](https://www.udacity.com/) in collaboration with [Figure Eight](https://www.figure-eight.com/).


## Appendix

Messages categories distribution from the training set

![newplot](https://user-images.githubusercontent.com/28582065/108807292-4b19cb00-75a4-11eb-822b-d3fc7519bcd3.png)

Model response example

![image](https://user-images.githubusercontent.com/28582065/108807373-86b49500-75a4-11eb-8c6f-c3df8b338668.png)
