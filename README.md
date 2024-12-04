# Bank-Marketing-Anaylsis
This project uses machine learning models (KNN and SVM) to predict whether customers will subscribe to a term deposit after a marketing call. The models achieved decent accuracy but struggled with recall due to class imbalance. Improvements are needed in balancing the data and fine-tuning the models for better predictions.

# Bank Marketing Prediction

## Project Overview
This project uses machine learning models (KNN and SVM) to predict whether customers will subscribe to a term deposit after a marketing phone call. The dataset contains customer information and the outcome of the marketing campaigns. The goal is to predict customer subscriptions using classification models.

## Dataset
The dataset is based on a marketing campaign conducted by a Portuguese bank, where phone calls were made to potential clients to offer term deposits. The target variable is whether the customer subscribed to the term deposit (`y`). The dataset includes features like age, job, marital status, education, and contact details.

## Models Used
- **K-Nearest Neighbors (KNN)**: Tested with different K values (1 to 20).
- **Support Vector Machines (SVM)**: Tested with Linear, Polynomial, and Gaussian kernels.

## Results
- **KNN**: The best performance was with **K=20**, achieving an accuracy of **89.5%**. However, recall was low across all values, indicating difficulty predicting subscribers.
- **SVM**: The **Gaussian kernel** performed best with **89.5% accuracy**, but again, recall was low, highlighting a bias toward predicting non-subscribers.

## Key Insights
- The models performed well at predicting non-subscribers but struggled with predicting subscribers.
- There is a class imbalance in the dataset, which may be affecting recall and causing the models to favor non-subscribers.

## Recommendations for Improvement
- **SMOTE** or other methods to address the class imbalance.
- **Hyperparameter tuning** for better performance.
- Exploring additional **feature engineering** to improve predictive power.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/bank-marketing-prediction.git

Usage:
Run the Jupyter notebooks or Python scripts to preprocess the data, train the models, and evaluate the results.
