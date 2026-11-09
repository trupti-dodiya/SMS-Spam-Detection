
# SMS Spam Detection

## Project Overview

SMS Spam Detection is a Machine Learning project that
classifies SMS messages into two categories:

- Spam
- Ham (Genuine Message)

The project uses Natural Language Processing (NLP)
techniques and a Machine Learning classification algorithm.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Streamlit
- Joblib
- TF-IDF Vectorization
- Multinomial Naive Bayes

## Project Workflow

1. Load the SMS dataset.
2. Clean and preprocess the text data.
3. Split the dataset into training and testing sets.
4. Convert text into numerical features using TF-IDF.
5. Train a Multinomial Naive Bayes model.
6. Evaluate model performance.
7. Build a web application using Streamlit.

## Project Structure

- app.py - Streamlit application
- train_model.py - Model training
- evaluate_model.py - Model evaluation
- predict.py - Predict new SMS messages
- clean_data.py - Data preprocessing
- spam_model.pkl - Saved trained model
- requirements.txt - Required packages

## Installation

Clone the repository:

git clone https://github.com/trupti-dodiya/SMS-Spam-Detection.git

Move into the project folder:

cd SMS-Spam-Detection

Install dependencies:

pip install -r requirements.txt

## Run the Application

streamlit run app.py

## Machine Learning Algorithm

Multinomial Naive Bayes

## Feature Extraction

TF-IDF (Term Frequency-Inverse Document Frequency)

## Dataset

SMS Spam Collection Dataset from the UCI Machine
Learning Repository.

## Future Improvements

- Compare multiple classification algorithms.
- Improve precision and recall.
- Add a more detailed user interface.
- Deploy the application online.

## Author

Trupti Dodiya
