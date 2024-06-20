**Breast Cancer Detection Project :=**

This project uses various machine learning models to classify breast cancer as either malignant (M) or benign (B) based on the provided dataset.

**Aim**

To build a machine learning model that accurately classifies breast cancer as malignant or benign using features extracted from breast cancer cell images.


**Project Overview :=**

1. Import Libraries

Import necessary libraries for data manipulation, model training, and evaluation.

2. Load and Inspect Data


Load the dataset using Pandas and inspect its structure, including the first and last few rows, shape, info, and statistical summary.

3. Data Mapping

Map the diagnosis column values to numerical values: Malignant (M) to 1 and Benign (B) to 0.

4. Feature Selection

Select features (X) and target (y) for the model. Here, the features are columns 2 to 32 and the target is the diagnosis column.

5. Data Visualization

Visualize the dataset using Seaborn pair plots to understand the distribution of features.

6. Data Preprocessing

Split the dataset into training and testing sets. Standardize the feature values using StandardScaler.

7. Model Training and Evaluation

Train multiple machine learning models (Logistic Regression, Decision Tree, SVM, Naive Bayes, KNN, LDA) and evaluate their performance using cross-validation.

8. Model Selection and Prediction

Train a Logistic Regression model on the training data, make predictions on the test data, and evaluate the model's performance using classification report, accuracy score, and R2 score.

**Conclusion :=**


This project demonstrates the application of various machine learning models to classify breast cancer. The Logistic Regression model, among others, is trained and evaluated for its accuracy and performance.
