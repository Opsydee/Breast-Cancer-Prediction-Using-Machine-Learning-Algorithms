# Project Topic:
Breast-Cancer-Prediction-Using-Machine-Learning-Algorithms.

# Project Overview:
This project aims to build a machine learning model to predict whether a breast tumor is benign or malignant based on diagnostic features extracted from breast tissue samples. The project demonstrates the process of data preprocessing, feature selection, model development, and evaluation to support healthcare decision-making using data-driven insights.

# Table of Contents
- Project Overview
- Installation and Setup
- Data
  - Source Data
  - Data Acquisition
  - Data Preprocessing
- Code Structure
- Usage
- Results and Evaluation
- Future Work
- Acknowledgments.

# Installation and Setup:
   - Install dependencies:
   - Install the required Python libraries by running:
     - import numpy as np
     - import pandas as pd
     - import matplotlib.pyplot as plt
     - import seaborn as sns
     - from sklearn.model_selection import train_test_split
     - from sklearn.linear_model import LogisticRegression
     - from sklearn.metrics import accuracy_score,confusion_matrix, classification_report, mean_squared_error, r2_score,ConfusionMatrixDisplay
     - from sklearn.model_selection import cross_val_score

# Data:
 - Data Acquisition:
    The dataset consists of diagnostic features for breast tumors, including attributes such as:
     - Mean radius
     - Texture
     - Smoothness
     - Compactness
     - Target labels (Benign or Malignant)
  - Data Preprocessing
    - Handling Missing Values: No missing values were detected in the dataset.
    - Feature Scaling: Features were normalized for better model performance.
    - Data Splitting: The dataset was divided into training and testing sets.

# Code Structure
 - data/: Contains raw and processed datasets.
 - notebooks/: Jupyter notebook for model exploration and evaluation.
 - src/: Scripts for data preprocessing, training, and evaluation.
 - results/: Stores evaluation metrics and visualizations.

# Usage:
  - Preprocess the dataset by running
  - Train the machine learning model
  - Evaluate the model

# Results and Evaluation:
  - The best-performing model achieved an accuracy of 95% on the test set and it is GradientBoosting Classifier.
  - The classification report and confusion matrix are available for a detailed breakdown of performance.
  - Key visualizations highlight feature importance and decision boundaries

# Future Work:
  - Model Optimization:Experiment with hyperparameter tuning for better performance.
  - Algorithm Selection:Test advanced models like Random Forest, XGBoost, or Neural Networks.
  - Deployment: Deploy the model as a web application using Streamlit or Flask.
  - Data Augmentation: Explore additional features or datasets to improve generalization.

# Acknowledgments:
Special thanks to the UCI Machine Learning Repository for the dataset and the open-source community for providing essential tools like Scikit-learn, Pandas, and Matplotlib. Thanks to Skillharvest and Miss chinazom for the knowledge impact for me to be able to solve this problem.
