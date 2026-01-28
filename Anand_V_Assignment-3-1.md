# Assignment 3.1

Name: Anand V  
Dataset: Anand_V_Assignment-3-1.csv  
Goal: Predict median_house_value

## 1) Problem
- It is a regression problem because the target is a number (median_house_value).
- We use features like income, rooms, and location to predict the price.
- Example: Higher median_income usually means higher house value.
- Reason: We need a numeric prediction for house price.
```
# Load data
base = os.path.dirname(__file__)
df = pd.read_csv(os.path.join(base, "housing.csv"))

# Clean and encode
df["total_bedrooms"] = df["total_bedrooms"].fillna(df["total_bedrooms"].median())
df = pd.get_dummies(df, columns=["ocean_proximity"], drop_first=True)
df = df.apply(pd.to_numeric, errors="coerce")
df = df.fillna(df.median(numeric_only=True))
target = "median_house_value"
```

## 2) Data preparation
- Missing values: total_bedrooms has empty values, so I filled with median.
- Categorical encoding: ocean_proximity is text, so I used one-hot encoding.
- Example: NEAR OCEAN becomes a new column with values 0 or 1.
- Reason: Missing values and text will break the model if not fixed. A 0/1 column lets the model read category as numbers.
```
# Clean and encode
df["total_bedrooms"] = df["total_bedrooms"].fillna(df["total_bedrooms"].median())
df = pd.get_dummies(df, columns=["ocean_proximity"], drop_first=True)
df = df.apply(pd.to_numeric, errors="coerce")
df = df.fillna(df.median(numeric_only=True))
target = "median_house_value"
```

## 3) Model training and testing
- Split: 70% train, 15% validation, 15% test.
- Train data is used to learn, validation is used to check, test is final check.
- Model: statsmodels OLS (Ordinary Least Squares, simple linear regression).
- Metrics: RMSE (Root Mean Squared Error), MAE (Mean Absolute Error), R2 (R-squared).
- Example: If validation RMSE is close to train RMSE, model is stable.
- Reason: Splits give a fair check, and metrics show accuracy.
- Model quality status: Moderate (train and validation errors are close, but linear model is simple).
- Model quality status rule: If validation RMSE is within 10% of train RMSE = Good, within 25% = Moderate, else Needs Improvement.
- Output summary file: Anand_V_Assignment-3-1-Output-Summary.json
```
# Split train/val/test (70/15/15)
np.random.seed(42)
idx = np.random.permutation(len(df))
n_train = int(0.70 * len(df))
n_val = int(0.15 * len(df))
train_df = df.iloc[idx[:n_train]]
val_df = df.iloc[idx[n_train : n_train + n_val]]
test_df = df.iloc[idx[n_train + n_val :]]

X_train = train_df.drop(columns=[target])
y_train = train_df[target]
X_val = val_df.drop(columns=[target])
y_val = val_df[target]
X_test = test_df.drop(columns=[target])
y_test = test_df[target]

# force numeric types for statsmodels
X_train = X_train.astype(float)
X_val = X_val.astype(float)
X_test = X_test.astype(float)
y_train = y_train.astype(float)
y_val = y_val.astype(float)
y_test = y_test.astype(float)

# Train OLS regression
X_train_c = sm.add_constant(X_train)
X_val_c = sm.add_constant(X_val, has_constant="add")
X_test_c = sm.add_constant(X_test, has_constant="add")
X_train_c = X_train_c.astype(float)
X_val_c = X_val_c.astype(float)
X_test_c = X_test_c.astype(float)
model = sm.OLS(y_train, X_train_c).fit()

# Predict
train_pred = model.predict(X_train_c)
val_pred = model.predict(X_val_c)
test_pred = model.predict(X_test_c)

# Metrics
def get_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return mae, rmse, r2

train_mae, train_rmse, train_r2 = get_metrics(y_train, train_pred)
val_mae, val_rmse, val_r2 = get_metrics(y_val, val_pred)
test_mae, test_rmse, test_r2 = get_metrics(y_test, test_pred)

# Model quality status (simple rule)
rmse_gap = abs(val_rmse - train_rmse) / train_rmse
if rmse_gap <= 0.10:
    model_quality = "Good"
elif rmse_gap <= 0.25:
    model_quality = "Moderate"
else:
    model_quality = "Needs Improvement"
```

## 4) Overfitting / underfitting
- Overfitting: train error low but validation error high.
- Underfitting: both train and validation errors high.
- Example: Train RMSE (Root Mean Squared Error) 40k and Val RMSE (Root Mean Squared Error) 90k = overfitting.
- In this work, train and validation errors are close, so overfitting is small.
- Reason: This check tells if model is too complex or too simple.
- Techniques to reduce overfitting: cross-validation, regularization (Ridge/Lasso), simpler model.
- Techniques to reduce underfitting: better features, non-linear model, more training data.
```
# Model quality status (simple rule)
rmse_gap = abs(val_rmse - train_rmse) / train_rmse
if rmse_gap <= 0.10:
    model_quality = "Good"
elif rmse_gap <= 0.25:
    model_quality = "Moderate"
else:
    model_quality = "Needs Improvement"
```

## 5) Reflection
- Data is skewed and target is capped at 500001, so high prices are hard to learn.
- Linear model is simple, so it cannot capture complex patterns.
- There is a learning curve for understanding concepts, terms, and numpy.


## Output summary JSON (generated python program)
```json
{
  "rows": 20640,
  "columns": 13,
  "missing_total_bedrooms": 0,
  "train_size": 14447,
  "val_size": 3096,
  "test_size": 3097,
  "model": "Statsmodels OLS",
  "model_quality_status": "Good",
  "metrics": {
    "train_rmse": 68393.3593436973,
    "val_rmse": 67668.17398620748,
    "test_rmse": 71483.88300249705,
    "train_mae": 49778.903266655565,
    "val_mae": 49482.23306798515,
    "test_mae": 50843.52059139623,
    "train_r2": 0.6480757459736415,
    "val_r2": 0.6502260378269678,
    "test_r2": 0.6254234270484975
  }
}
```
