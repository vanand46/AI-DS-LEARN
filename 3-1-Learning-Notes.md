# Learning Notes: Assignment 3.1 Concepts and Definitions

1) Supervised learning
Definition: Model learns from labeled data (features + known target).
In this assignment: We learn from housing features and known prices.

2) Regression
Definition: Predicting a continuous numeric value.
In this assignment: Predicting median_house_value.

3) Features vs target
Definition: Features are inputs, target is the output we predict.
Examples: Features = longitude, latitude, median_income. Target = median_house_value.

4) Missing values (NaN) and imputation
Definition: Missing values are empty cells in data.
Imputation: Filling missing values (median for numeric).
In this assignment: total_bedrooms missing values filled with median.

5) Categorical feature
Definition: A column with text categories.
In this assignment: ocean_proximity is categorical (NEAR OCEAN, INLAND).

6) One-hot encoding
Definition: Convert each category into a 0/1 column.
In this assignment: ocean_proximity_NEAR_OCEAN = 1 or 0.
Reason: Models only work with numbers.

7) Train/validation/test split
Definition: Split data into three parts for learning and checking.
In this assignment: 70% train, 15% validation, 15% test.

8) Linear regression (OLS)
Definition: Fits a straight-line equation between features and target.
Equation: y = w1*x1 + w2*x2 + ... + b
In this assignment: statsmodels OLS is used.

9) Predictions
Definition: Model outputs estimated values for target.
In this assignment: predict house value for train, validation, test.

10) Evaluation metrics
MAE (Mean Absolute Error): average absolute error.
RMSE (Root Mean Squared Error): punishes big errors more.
R2 (R-squared): how much variance is explained.

11) Overfitting
Definition: Very good on train, poor on validation/test.
Signal: Train error low, validation error high.

12) Underfitting
Definition: Model too simple, poor on both train and validation.
Signal: Both train and validation errors are high.

13) Model quality status
Definition: Simple label based on train vs validation RMSE gap.
Rule used: within 10% = Good, within 25% = Moderate, else Needs Improvement.

14) Skewed data
Definition: Many low values and few high values.
In this assignment: house values are right-skewed.

15) Target capping
Definition: Max value is capped in dataset.
In this assignment: median_house_value is capped at 500001.

16) JSON summary output
Definition: A saved file with key results and metrics.
In this assignment: Anand_V_Assignment-3-1-Output-Summary.json.
