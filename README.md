
# Student Dropout Prediction

## Project Overview

This project develops a machine learning system to predict whether a student belongs to the dropout group.

The project uses Logistic Regression as a binary classification model and includes data preprocessing, exploratory data analysis, model training, evaluation, and a Gradio-based prediction application.

## Problem Statement

Student dropout is an important educational issue. The objective of this project is to use student demographic, enrollment, academic, and related information to predict whether a student belongs to the dropout class.

The prediction can potentially be used as part of an educational early-warning system to help identify students who may require additional support.

## Dataset

The dataset contains:

- 4,424 student records
- 36 input features
- Student demographic information
- Enrollment information
- Academic information
- Economic indicators

The original target contains three categories:

- Graduate
- Enrolled
- Dropout

For this project, the target was converted into a binary classification problem:

- 1 = Dropout
- 0 = Not Dropout

The Graduate and Enrolled categories were grouped into the Not Dropout class.

## Data Preprocessing

The dataset was checked for:

- Missing values
- Duplicate records
- Data types
- Target values

The dataset contained:

- 0 missing values
- 0 duplicate records

The target variable was converted into a binary target.

The final dataset contained:

- 4,424 records
- 36 input features

The data was then divided into training and testing sets.

## Exploratory Data Analysis

Exploratory Data Analysis was performed to understand:

- Dropout distribution
- Academic performance
- Approved curricular units
- Feature relationships
- Correlations with the target

Visualizations were created using Matplotlib and Seaborn.

## Machine Learning Model

### Logistic Regression

Logistic Regression was selected because this project is a binary classification problem.

The dataset was divided into training and testing data before model training.

Feature scaling was performed using StandardScaler.

The Logistic Regression model was then trained using the training data.

## Model Evaluation

The model was evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

## Prediction Application

A Gradio-based application was developed to use the trained model for student risk prediction.

The application allows student information to be entered and produces:

- Predicted class
- Dropout probability
- Risk level

Risk levels are categorized based on the predicted dropout probability.

## Project Workflow

Student Dataset
        ↓
Data Cleaning
        ↓
Preprocessing
        ↓
Exploratory Data Analysis
        ↓
Train/Test Split
        ↓
Feature Scaling
        ↓
Logistic Regression
        ↓
Model Evaluation
        ↓
Dropout Probability
        ↓
Risk Level
        ↓
Gradio Application

## Tools and Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Gradio
- Google Colab
- GitHub

## Project Results

Add the actual evaluation results here:

- Accuracy : 0.8859
- Precision: 0.8894
- Recall   : 0.7359
- F1 Score : 0.8054
- ROC-AUC  : 0.9266

Add the confusion matrix and screenshots of the prediction application.

## Potential Application

This project demonstrates how machine learning can be used as part of an educational early-warning system.

Such a system could help identify students who may need additional academic or institutional support.

The model should be treated as a decision-support tool rather than a replacement for human judgment.

## Future Improvements

Possible improvements include:

- Testing additional classification algorithms
- Hyperparameter tuning
- More detailed feature engineering
- Improving the user interface
- Deploying the application publicly
- Evaluating the model on additional datasets
