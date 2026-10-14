
from pathlib import Path
from datetime import date, timedelta
import json

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware


# ==================================================
# 1. PROJECT PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

DATA_PATH = BASE_DIR / "product demand.csv"
RIDGE_PATH = BASE_DIR / "ridge_model.pkl"
LASSO_PATH = BASE_DIR / "lasso_model.pkl"
PREPROCESSOR_PATH = BASE_DIR / "preprocessor.pkl"
FEATURES_PATH = BASE_DIR / "feature_columns.json"

required_files = [
    DATA_PATH,
    RIDGE_PATH,
    LASSO_PATH,
    PREPROCESSOR_PATH,
    FEATURES_PATH,
    STATIC_DIR / "style.css",
    STATIC_DIR / "script.js",
    BASE_DIR / "index.html",
]

missing_files = [p.name for p in required_files if not p.exists()]

if missing_files:
    raise FileNotFoundError(
        f"Missing project files: {missing_files}"
    )


# ==================================================
# 2. LOAD DATA AND TRAINED MODELS
# ==================================================

df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])

with open(FEATURES_PATH, "r", encoding="utf-8") as file:
    feature_cols = json.load(file)

preprocessor = joblib.load(PREPROCESSOR_PATH)
ridge_model = joblib.load(RIDGE_PATH)
lasso_model = joblib.load(LASSO_PATH)

df = df.sort_values(
    ["store_nbr", "family", "date"]
).reset_index(drop=True)

required_columns = {
    "date", "store_nbr", "family", "sales", "onpromotion",
    "city", "state", "type", "cluster"
}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Dataset is missing columns: {sorted(missing_columns)}"
    )


# ==================================================
# 3. FASTAPI APPLICATION
# ==================================================

app = FastAPI(
    title="Product Demand Forecasting API",
    description=(
        "Product demand predictions using Ridge and Lasso Regression"
    ),
    version="1.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

app.mount(
    "/static",
    StaticFiles(directory=STATIC_DIR),
    name="static"
)


# ==================================================
# 4. REQUEST SCHEMA
# ==================================================

class PredictionRequest(BaseModel):
    store_nbr: int = Field(gt=0)
    family: str = Field(min_length=1)
    forecast_date: date
    onpromotion: int = Field(default=0, ge=0)


# ==================================================
# 5. HOMEPAGE
# ==================================================

@app.get("/")
def home():
    return FileResponse(BASE_DIR / "index.html")


# ==================================================
# 6. OPTIONS FOR THE WEBSITE
# ==================================================

@app.get("/api/options")
def get_options():
    store_info = (
        df.groupby("store_nbr")
        .agg(
            city=("city", "first"),
            state=("state", "first"),
            type=("type", "first"),
            cluster=("cluster", "first")
        )
        .reset_index()
    )

    stores = [
        {
            "value": int(row["store_nbr"]),
            "label": (
                f"Store {int(row['store_nbr'])} — "
                f"{row['city']}, {row['state']}"
            ),
            "store_nbr": int(row["store_nbr"]),
            "city": str(row["city"]),
            "state": str(row["state"]),
            "type": str(row["type"]),
            "cluster": int(row["cluster"])
        }
        for _, row in store_info.iterrows()
    ]

    return {
        "stores": stores,
        "families": sorted(df["family"].dropna().unique().tolist()),
        "models": ["Ridge", "Lasso", "Combined"],
        "data_start": str(df["date"].min().date()),
        "data_end": str(df["date"].max().date())
    }


# ==================================================
# 7. BUILD FEATURES FOR ONE PREDICTION
# ==================================================

def build_prediction_features(request: PredictionRequest):
    store_history = df[
        (df["store_nbr"] == request.store_nbr)
        & (df["family"] == request.family)
        & (df["date"] < pd.Timestamp(request.forecast_date))
    ].sort_values("date")

    if store_history.empty:
        raise HTTPException(
            status_code=400,
            detail="No historical sales found for this store and family."
        )

    latest_date = df["date"].max().date()

    if request.forecast_date > latest_date + timedelta(days=1):
        raise HTTPException(
            status_code=400,
            detail=(
                "This version supports dates in the dataset and "
                "one-day-ahead forecasts only."
            )
        )

    if len(store_history) < 7:
        raise HTTPException(
            status_code=400,
            detail="At least 7 historical observations are required."
        )

    sales_history = store_history["sales"].astype(float)

    # Uses the last seven available observations before the forecast date.
    lag_1 = float(sales_history.iloc[-1])
    lag_7 = float(sales_history.iloc[-7])
    rolling_mean_7 = float(sales_history.iloc[-7:].mean())

    store_rows = df[df["store_nbr"] == request.store_nbr]

    if store_rows.empty:
        raise HTTPException(
            status_code=400,
            detail="Unknown store number."
        )

    if request.family not in set(df["family"].dropna()):
        raise HTTPException(
            status_code=400,
            detail="Unknown product family."
        )

    store = store_rows.iloc[0]
    forecast_day = pd.Timestamp(request.forecast_date)

    row = {
        "store_nbr": request.store_nbr,
        "family": request.family,
        "onpromotion": request.onpromotion,
        "city": store["city"],
        "state": store["state"],
        "type": store["type"],
        "cluster": int(store["cluster"]),
        "year": forecast_day.year,
        "month": forecast_day.month,
        "day": forecast_day.day,
        "day_of_week": forecast_day.dayofweek,
        "week_of_year": int(forecast_day.isocalendar().week),
        "is_weekend": int(forecast_day.dayofweek >= 5),
        "lag_1": lag_1,
        "lag_7": lag_7,
        "rolling_mean_7": rolling_mean_7
    }

    missing_features = set(feature_cols) - set(row)

    if missing_features:
        raise HTTPException(
            status_code=500,
            detail=f"Missing model features: {sorted(missing_features)}"
        )

    input_df = pd.DataFrame(
        [[row[column] for column in feature_cols]],
        columns=feature_cols
    )

    try:
        processed = preprocessor.transform(input_df)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Feature preprocessing failed: {str(exc)}"
        ) from exc

    return processed, {
        "lag_1": round(lag_1, 2),
        "lag_7": round(lag_7, 2),
        "rolling_mean_7": round(rolling_mean_7, 2)
    }


# ==================================================
# 8. PREDICT USING BOTH MODELS
# ==================================================

@app.post("/api/predict")
def predict_demand(request: PredictionRequest):
    processed, historical_features = build_prediction_features(request)

    try:
        ridge_raw = float(ridge_model.predict(processed)[0])
        lasso_raw = float(lasso_model.predict(processed)[0])
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Model prediction failed: {str(exc)}"
        ) from exc

    # Keep raw estimates for transparency; sales predictions are nonnegative.
    ridge_prediction = max(0.0, ridge_raw)
    lasso_prediction = max(0.0, lasso_raw)
    combined_prediction = (ridge_prediction + lasso_prediction) / 2.0

    return {
        "store_nbr": request.store_nbr,
        "family": request.family,
        "forecast_date": str(request.forecast_date),
        "predicted_sales": round(combined_prediction, 2),
        "ridge_prediction": round(ridge_prediction, 2),
        "lasso_prediction": round(lasso_prediction, 2),
        "combined_prediction": round(combined_prediction, 2),
        "models_used": ["Ridge", "Lasso"],
        "historical_features": historical_features,
        "note": (
            "Combined prediction is the average of the Ridge and Lasso "
            "estimates. Averaging does not guarantee improved accuracy."
        )
    }
