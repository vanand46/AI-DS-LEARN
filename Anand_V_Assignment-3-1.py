import os
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Load data
base = os.path.dirname(__file__)
df = pd.read_csv(os.path.join(base, "housing.csv"))

# Clean and encode
df["total_bedrooms"] = df["total_bedrooms"].fillna(df["total_bedrooms"].median())
df = pd.get_dummies(df, columns=["ocean_proximity"], drop_first=True)
df = df.apply(pd.to_numeric, errors="coerce")
df = df.fillna(df.median(numeric_only=True))
target = "median_house_value"

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

# Save summary
summary = {
    "rows": int(df.shape[0]),
    "columns": int(df.shape[1]),
    "missing_total_bedrooms": int(df["total_bedrooms"].isna().sum()),
    "train_size": int(len(X_train)),
    "val_size": int(len(X_val)),
    "test_size": int(len(X_test)),
    "model": "Statsmodels OLS",
    "model_quality_status": model_quality,
    "metrics": {
        "train_rmse": float(train_rmse),
        "val_rmse": float(val_rmse),
        "test_rmse": float(test_rmse),
        "train_mae": float(train_mae),
        "val_mae": float(val_mae),
        "test_mae": float(test_mae),
        "train_r2": float(train_r2),
        "val_r2": float(val_r2),
        "test_r2": float(test_r2),
    },
}
with open(os.path.join(base, "Anand_V_Assignment-3-1-Output-Summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
