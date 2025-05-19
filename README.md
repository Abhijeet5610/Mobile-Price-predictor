# Mobile Price Prediction Using Machine Learning Models

## Project Overview
This project focuses on building machine learning models to predict the price of a Mobile Phone based on given attributes. The dataset contains various attributes, and the goal is to classify a mobile phone falls under which price range out of four(Multiclass Classification).

## Models Implemented
- Logistic Regression **Final chosen model**
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest 
- Support Vector Machine (SVM)
- Softmax Regression
- Neural network

## Dataset
The data consists of clinical parameters such as battery_power,blue,clock_speed,dual_sim,fc,four_g,int_memory,m_dep,mobile_wt,n_cores,pc,px_height,px_width,ram,sc_h,sc_w,talk_time and more.

## Data Preprocessing
- Normalized numerical features
- Split dataset into training (80%) and testing (20%) sets

## Model Selection and Evaluation
Each model was trained and evaluated using metrics:
- Accuracy
- Precision
- Recall
- F1 Score

After extensive hyperparameter tuning and evaluation, the **Logistic Regression** model demonstrated the best overall performance with an F1 score of **0.9825** on the test data.
