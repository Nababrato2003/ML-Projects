# Twitter Sentiment Analysis - YBI Foundation Internship

## Overview
This repository contains a Machine Learning project developed during my internship at the YBI Foundation. The objective of this project is to perform Sentiment Analysis on Twitter data, accurately classifying tweets as either positive or negative.

## Dataset
The model is trained on a Kaggle Twitter Sentiment Dataset containing 1.6 million processed tweets. 
* **0**: Negative Tweet
* **1** (mapped from 4): Positive Tweet

## Technologies and Libraries Used
* **Programming Language:** Python 3
* **Data Manipulation:** Pandas, NumPy
* **Natural Language Processing (NLP):** NLTK (Natural Language Toolkit), Regular Expressions (`re`)
* **Machine Learning:** Scikit-Learn (scikit-learn)
* **Model Serialization:** Pickle

## Methodology
1.  **Data Preprocessing:** * Loaded the dataset with 1.6 million entries and mapped the positive target labels from '4' to '1' for standard binary classification.
    * Applied **Stemming** using the `PorterStemmer` to reduce words to their root forms (e.g., acting, actor -> act).
    * Removed English stopwords and non-alphabetic characters using regex to reduce noise in the text data.
2.  **Feature Extraction:**
    * Utilized a **TF-IDF Vectorizer** (Term Frequency-Inverse Document Frequency) to convert the cleaned textual data into numerical feature vectors suitable for the machine learning algorithm.
3.  **Model Training:**
    * Split the dataset into training (80%) and testing (20%) sets using stratified sampling.
    * Trained a **Logistic Regression** model on the vectorized data.
4.  **Evaluation & Export:**
    * Evaluated the model using standard accuracy scores.
    * Exported the final trained model as `trained_model.sav` using `pickle` to allow for future predictions without retraining.

## Results
The Logistic Regression model achieved the following performance metrics:
* **Training Data Accuracy:** ~79.8%
* **Test Data Accuracy:** ~77.8%

## How to Run
1. Clone the repository to your local machine.
2. Ensure you have the required dependencies installed:
   ```bash
   pip install numpy pandas nltk scikit-learn
