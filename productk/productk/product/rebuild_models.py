
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge, Lasso

BASE = Path(__file__).parent
DATA_FILE = BASE / "product demand.csv"

print("Using scikit-learn:", sklearn.__version__)
print("Loading dataset...")

df = pd.read_csv(DATA_FILE)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["store_nbr", "family", "date"]).copy()

# Calendar features
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["day_of_week"] = df["date"].dt.dayofweek
df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

# Historical sales features, calculated separately for each store/family
groups = df.groupby(["store_nbr", "family"], sort=False)["sales"]
df["lag_1"] = groups.shift(1)
df["lag_7"] = groups.shift(7)
df["rolling_mean_7"] = groups.transform(
    lambda s: s.shift(1).rolling(7, min_periods=7).mean()
)

categorical_features = ["family", "city", "state", "type"]
numerical_features = [
    "store_nbr", "onpromotion", "cluster",
    "year", "month", "day", "day_of_week",
    "week_of_year", "is_weekend",
    "lag_1", "lag_7", "rolling_mean_7"
]
feature_cols = categorical_features + numerical_features

df = df.dropna(subset=feature_cols + ["sales"]).copy()

# Fit the final models using April-November data.
# December remains held out for testing.
train = df[
    (df["date"] >= "2016-04-01") &
    (df["date"] < "2016-12-01")
].copy()

if train.empty:
    raise ValueError(
        "No training rows found for April-November 2016. "
        "Check the dates in product demand.csv."
    )

X_train = train[feature_cols]
y_train = train["sales"]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        ("numerical", StandardScaler(), numerical_features)
    ]
)

print("Training rows:", len(train))
X_processed = preprocessor.fit_transform(X_train)

ridge = Ridge(alpha=0.0001)
lasso = Lasso(alpha=50, max_iter=100000, tol=1e-4)

print("Training Ridge...")
ridge.fit(X_processed, y_train)

print("Training Lasso...")
lasso.fit(X_processed, y_train)

# Back up existing files before replacing them
for name in ["preprocessor.pkl", "ridge_model.pkl", "lasso_model.pkl"]:
    path = BASE / name
    if path.exists():
        backup = BASE / (name + ".colab_backup")
        if not backup.exists():
            shutil.copy2(path, backup)

joblib.dump(preprocessor, BASE / "preprocessor.pkl")
joblib.dump(ridge, BASE / "ridge_model.pkl")
joblib.dump(lasso, BASE / "lasso_model.pkl")

with open(BASE / "feature_columns.json", "w", encoding="utf-8") as f:
    json.dump(feature_cols, f, indent=2)

# Verify that the new files load in this environment
joblib.load(BASE / "preprocessor.pkl")
joblib.load(BASE / "ridge_model.pkl")
joblib.load(BASE / "lasso_model.pkl")

print("\nSUCCESS: All three model files were saved and loaded.")
print("Preprocessed training shape:", X_processed.shape)
print("You can now start the FastAPI website.")
