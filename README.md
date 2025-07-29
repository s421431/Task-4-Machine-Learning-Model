Of course! Here is a comprehensive README file for your Python script. It explains the project's purpose, the methodology, how to run it, and what to expect from the output.

Spam Email Classification Comparison

This project builds, trains, and evaluates five different machine learning models to classify emails as spam or not spam. It uses the well-known "Spambase" dataset from the UCI Machine Learning Repository.

The primary goal is to compare the performance of these classifiers on the same dataset using standard evaluation metrics like accuracy, precision, recall, and F1-score.

Table of Contents

Project Overview

Dataset

Methodology

Requirements

How to Run

Results

License

Project Overview

The script performs a complete machine learning workflow:

Data Loading: Fetches the Spambase dataset directly from the UCI repository.

Preprocessing: Splits the data into training and testing sets and standardizes the features using StandardScaler.

Model Training: Trains five distinct classification models on the prepared data.

Evaluation: Evaluates each model's performance on the unseen test set.

Reporting: Prints a detailed classification report and a confusion matrix for each model.

Visualization: Generates and displays a heatmap of the confusion matrix for each model, providing a clear visual representation of its performance.

The models compared are:

Logistic Regression

Gaussian Naive Bayes

Support Vector Machine (SVM)

Decision Tree Classifier

K-Nearest Neighbors (KNN)

Dataset

The project uses the Spambase Data Set from the UCI Machine Learning Repository.

Source: https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data

Description: The dataset consists of 4601 emails, each represented by 57 features. The features include the frequency of certain words (e.g., "money", "free"), the frequency of specific characters (e.g., '!', '$'), and statistics on capital letters.

Target Variable: The last column, spam, indicates whether the email was considered spam (1) or not (0).

Methodology

The script follows these key steps:

Load Data: The spambase.data file is loaded into a pandas DataFrame, and appropriate column names are assigned.

Feature and Target Separation: The DataFrame is split into features (X) and the target variable (y).

Train-Test Split: The data is divided into an 80% training set and a 20% testing set using a random_state for reproducibility.

Feature Scaling: StandardScaler is used to standardize the features by removing the mean and scaling to unit variance. This is crucial for distance-based algorithms like SVM and KNN and beneficial for Logistic Regression.

Iterative Modeling: The script iterates through a dictionary of the five models. In each iteration:

The model is trained on the scaled training data (X_train_scaled).

Predictions are made on the scaled test data (X_test_scaled).

Key performance metrics are calculated and stored.

Display Results: For each model, the script prints a formatted summary of its performance metrics and displays a confusion matrix heatmap using seaborn and matplotlib.

Requirements

To run this script, you need Python 3 and the following libraries. You can install them using pip:

Generated bash
pip install pandas scikit-learn seaborn matplotlib numpy


Alternatively, you can create a requirements.txt file with the following content:

Generated code
pandas
scikit-learn
seaborn
matplotlib
numpy
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
IGNORE_WHEN_COPYING_END

And install them all with:

Generated bash
pip install -r requirements.txt
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END
How to Run

Save the code as a Python file (e.g., spam_classifier.py).

Make sure you have an active internet connection, as the script downloads the dataset from a URL.

Run the script from your terminal:

Generated bash
python spam_classifier.py
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

The script will print the performance reports to the console and display a confusion matrix plot for each of the five models one by one. You will need to close each plot window to see the next one.

Results

The script outputs a detailed performance analysis for each classifier. The output for each model includes:

Accuracy: The overall proportion of correctly classified instances.

Precision: The proportion of positive predictions that were actually correct.

Recall: The proportion of actual positives that were correctly identified.

F1-Score: The harmonic mean of precision and recall.

Classification Report: A detailed, per-class breakdown of precision, recall, and F1-score.

Confusion Matrix: A table showing the number of true positives, true negatives, false positives, and false negatives.

Example Output (Logistic Regression)
Generated code
--- Logistic Regression ---
Accuracy: 0.9262
Precision: 0.9213
Recall: 0.8995
F1-Score: 0.9102
Classification Report:
              precision    recall  f1-score   support

           0       0.93      0.95      0.94       531
           1       0.92      0.90      0.91       390

    accuracy                           0.93       921
   macro avg       0.93      0.92      0.92       921
weighted avg       0.93      0.93      0.93       921

Confusion Matrix:
[[502  29]
 [ 39 351]]

==============================
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
IGNORE_WHEN_COPYING_END

After the text output for each model, a plot like this will be displayed:

This visual representation helps in quickly assessing how well the model distinguishes between "Spam" and "Not Spam".
