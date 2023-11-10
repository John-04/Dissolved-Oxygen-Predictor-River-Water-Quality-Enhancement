# Dissolved-Oxygen-Predictor-River-Water-Quality-Enhancement
A machine learning project predicting river water dissolved oxygen. It includes data preprocessing, feature engineering, outlier detection, and uses Random Forest with GridSearchCV for optimization. The goal is to enhance water quality assessment in rivers.

## Data Loading and Preprocessing:

Imported necessary libraries for data manipulation, machine learning, and visualization.
Loaded the original dataset ("sample_submission.csv") using the pandas library.
Conducted Exploratory Data Analysis (EDA) to understand the distribution of the target variable (dissolved oxygen) and visualized it using histograms and kernel density plots.
Handled missing values by imputing them with the mean of each column using SimpleImputer from scikit-learn.

## Feature Engineering:
Calculated the mean and standard deviation for groups of variables related to O2, NH4, NO2, NO3, and BOD5.
Computed the total sum of O2, NH4, NO2, NO3, and BOD5, creating new features representing cumulative values.
Removed redundant or unnecessary features to simplify the dataset.

## Outlier Detection and Handling:
Visualized outliers using box plots.
Used the Z-score method to detect and remove outliers, ensuring a cleaner dataset.

## Balancing the Dataset:
Applied Synthetic Minority Over-sampling Technique (SMOTE) to address class imbalance, resulting in a more balanced distribution of target categories.

## Correlation Analysis:
Calculated and visualized the correlation matrix to understand relationships between different features.

##Model Training and Hyperparameter Tuning:
Used GridSearchCV to perform hyperparameter tuning for a Random Forest model.
Trained a Random Forest model on the preprocessed and balanced dataset.

## Model Evaluation:
Made predictions on the validation set and evaluated the model using Mean Squared Error (MSE) as the performance metric.

## Iteration and Refinement:
Emphasized the iterative nature of model refinement based on evaluation results.
Encouraged continuous improvement through fine-tuning hyperparameters, exploring advanced feature engineering, or considering alternative algorithms.
Finally, the script concluded by saving the cleaned and enhanced dataset to a CSV file named "cleaned_enhanced_dataset.csv." This dataset is expected to be more suitable for training and evaluating machine learning models compared to the original dataset.
