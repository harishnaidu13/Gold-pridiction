from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from custom_transformers import CorrelationSelector, MissingValueSelector

BASE_DIR = Path(__file__).parent
CSV_PATH = BASE_DIR / "financial_regression (1).csv"
MODEL_PATH = BASE_DIR / "model.joblib"

# 1. Load dataset
df = pd.read_csv(CSV_PATH)

# 2. Remove rows where all numeric values are missing
cols = df.select_dtypes(include="number").columns
df.dropna(subset=cols, how="all", inplace=True)
df.drop(columns=["date"],inplace=True)

# # 3. Convert date column into features
# df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
# df["day"] = df["date"].dt.day
# df["month"] = df["date"].dt.month
# df["year"] = df["date"].dt.year
# df.drop(columns=["date"], inplace=True)

 
df = df.dropna(subset=["gold close"])
num=df.select_dtypes(include="number").columns.tolist()
per=df[num].isnull().mean()*100
val=per[per>80].index.tolist()
df=df.drop(columns=val)

# 5. Separate X and y
X = df.drop(columns=["gold close"])
y = df["gold close"]
X = X.select_dtypes(include="number")


corr=X.corr().abs()
upper=corr.where(np.triu(np.ones(corr.shape),k=1).astype(bool))
drop_col=[col for col in upper.columns if (upper[col]>0.95).any()]
X=X.drop(columns=drop_col)

# 6. Train/test split (time-ordered, no shuffle)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, shuffle=False
)

# 7. Pipeline
pipeline = Pipeline([
    ("missing_selector", MissingValueSelector()),
    ("imputer", SimpleImputer(strategy="median")),
    ("correlation", CorrelationSelector(threshold=0.85)),
    ("scaler", StandardScaler()),
    ("model", LinearRegression()),
])

pipeline.fit(X_train, y_train)

# 8. Evaluation
y_pred = pipeline.predict(X_test)
metrics = {
    "mae": float(mean_absolute_error(y_test, y_pred)),
    "mse": float(mean_squared_error(y_test, y_pred)),
    "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
    "r2": float(r2_score(y_test, y_pred)),
}
for name, value in metrics.items():
    print(f"{name.upper()}: {value}")

# 9. Save the pipeline and the exact input columns it expects
joblib.dump(
    {"pipeline": pipeline, "features": list(X.columns), "metrics": metrics},
    MODEL_PATH,
)
print(f"Saved model to {MODEL_PATH}")



# import numpy as np
# import pandas as pd

# from sklearn.pipeline import Pipeline
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import (
#     mean_absolute_error,
#     mean_squared_error,
#     r2_score
# )

# from sklearn.base import BaseEstimator, TransformerMixin


# # 1. Load dataset
# df = pd.read_csv(r"C:\Users\harish\OneDrive\Desktop\Linear regression.py\financial_regression (1).csv")

# # 2. Remove rows where all numeric values are missing
# cols = df.select_dtypes(include="number").columns
# df.dropna(subset=cols, how="all", inplace=True)


# # 3. Convert date column into features
# df["date"] = pd.to_datetime(
#     df["date"], errors="coerce", dayfirst=True
# )

# df["day"] = df["date"].dt.day
# df["month"] = df["date"].dt.month
# df["year"] = df["date"].dt.year

# df.drop(columns=["date"], inplace=True)


# # 4. Remove rows where target is missing
# df = df.dropna(subset=["gold close"])


# # 5. Separate X and y
# X = df.drop(columns=["gold close"])
# y = df["gold close"]

# # Keep numerical features
# X = X.select_dtypes(include="number")


# # 6. Train test split
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y,
#     test_size=0.30,
#     shuffle=False
# )


# # 7. Custom transformer for removing
# # features with more than 80% missing values

# class MissingValueSelector(BaseEstimator, TransformerMixin):

#     def fit(self, X, y=None):
#         missing_percent = pd.DataFrame(X).isnull().mean()
#         self.selected_features_ = missing_percent[
#             missing_percent <= 0.80
#         ].index.tolist()
#         return self

#     def transform(self, X):
#         return pd.DataFrame(X).loc[:, self.selected_features_]


# # 8. Custom transformer for correlation
# # feature selection

# class CorrelationSelector(BaseEstimator, TransformerMixin):

#     def __init__(self, threshold=0.85):
#         self.threshold = threshold

#     def fit(self, X, y=None):
#         X = pd.DataFrame(X)
#         corr = X.corr().abs()

#         upper = corr.where(
#             np.triu(np.ones(corr.shape), k=1).astype(bool)
#         )

#         self.selected_features_ = [
#             col for col in upper.columns
#             if not any(upper[col] > self.threshold)
#         ]

#         return self

#     def transform(self, X):
#         return pd.DataFrame(X).loc[:, self.selected_features_]


# # 9. Create Pipeline

# pipeline = Pipeline([
#     ("missing_selector", MissingValueSelector()),

#     ("imputer", SimpleImputer(strategy="median")),

#     ("correlation", CorrelationSelector(threshold=0.85)),

#     ("scaler", StandardScaler()),

#     ("model", LinearRegression())
# ])


# # 10. Train model
# pipeline.fit(X_train, y_train)


# # 11. Prediction
# y_pred = pipeline.predict(X_test)


# # 12. Evaluation
# print("MAE:", mean_absolute_error(y_test, y_pred))

# print("MSE:", mean_squared_error(y_test, y_pred))

# print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# print("R2 Score:", r2_score(y_test, y_pred))
