"""
ARIMA modeling template for atmospheric CO₂ concentration forecasting.

Usage:
    python src/model_arima.py \
        --data_path data/co2.csv \
        --date_col date \
        --value_col co2 \
        --order 1 1 1 \
        --test_size 0.2

Outputs (saved under results/):
    - forecast.csv           : holds train/test split and predictions
    - metrics.json           : MSE and meta info
    - forecast_plot.png      : line plot of actual vs. forecast on test set
"""

import argparse
import os
import json
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

def load_series(
    path: str,
    date_col: Optional[str] = None,
    value_col: Optional[str] = None
) -> pd.Series:
    """
    Load a time series from CSV and return a pandas Series with a DateTimeIndex.
    The function attempts a robust auto-detection when columns are not provided.
    """
    df = pd.read_csv(path)
    # Try to auto-detect date column
    if date_col is None:
        candidates = [c for c in df.columns if str(c).lower() in {"date", "month", "time", "timestamp"}]
        if not candidates:
            # Try common combinations like Year, Month
            if {"year","month"}.issubset({c.lower() for c in df.columns}):
                df["date"] = pd.to_datetime(dict(year=df.filter(regex="(?i)^year$").iloc[:,0],
                                                 month=df.filter(regex="(?i)^month$").iloc[:,0], day=1))
                date_col = "date"
            else:
                # Fallback: first column as date
                date_col = df.columns[0]
        else:
            date_col = candidates[0]
    # Try to parse dates
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col)

    # Detect value column
    if value_col is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            value_col = numeric_cols[-1]  # pick the last numeric column by default
        else:
            raise ValueError("No numeric column found for value_col. Please specify --value_col.")

    s = pd.Series(df[value_col].values, index=pd.DatetimeIndex(df[date_col], name=date_col), name=value_col)
    s = s.asfreq(pd.infer_freq(s.index) or "MS")  # infer frequency; default to month start if unknown
    # Simple forward-fill for missing values
    s = s.ffill()
    return s

def train_test_split_series(s: pd.Series, test_size: float = 0.2) -> Tuple[pd.Series, pd.Series]:
    n = len(s)
    n_test = max(1, int(n * test_size))
    return s.iloc[:-n_test], s.iloc[-n_test:]

def fit_arima(train: pd.Series, order: Tuple[int,int,int]) -> ARIMA:
    model = ARIMA(train, order=order, enforce_stationarity=False, enforce_invertibility=False)
    return model.fit()

def evaluate_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))

def plot_forecast(train: pd.Series, test: pd.Series, pred: pd.Series, out_path: str) -> None:
    plt.figure()  # single-plot figure
    train.plot(label="Train")
    test.plot(label="Test")
    pred.plot(label="Forecast")
    plt.legend()
    plt.title("CO₂ Forecast (ARIMA)")
    plt.xlabel("Date")
    plt.ylabel("CO₂ concentration")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/co2.csv")
    parser.add_argument("--date_col", type=str, default=None)
    parser.add_argument("--value_col", type=str, default=None)
    parser.add_argument("--order", type=int, nargs=3, default=[1,1,1], help="ARIMA(p,d,q)")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    series = load_series(args.data_path, args.date_col, args.value_col)
    train, test = train_test_split_series(series, test_size=args.test_size)
    model_fit = fit_arima(train, tuple(args.order))

    # Dynamic forecast over the test horizon
    forecast = model_fit.forecast(steps=len(test))
    forecast.index = test.index  # align index

    mse = evaluate_mse(test.values, forecast.values)

    # Save artifacts
    out_csv = os.path.join(args.results_dir, "forecast.csv")
    pd.DataFrame({
        "date": test.index,
        "actual": test.values,
        "forecast": forecast.values
    }).to_csv(out_csv, index=False)

    out_metrics = os.path.join(args.results_dir, "metrics.json")
    with open(out_metrics, "w", encoding="utf-8") as f:
        json.dump({
            "model": "ARIMA",
            "order": args.order,
            "test_size": args.test_size,
            "mse": mse,
            "n_train": int(len(train)),
            "n_test": int(len(test))
        }, f, indent=2)

    out_png = os.path.join(args.results_dir, "forecast_plot.png")
    plot_forecast(train, test, forecast, out_png)

    print(f"[OK] Saved: {out_csv}")
    print(f"[OK] Saved: {out_metrics}")
    print(f"[OK] Saved: {out_png}")
    print(f"MSE: {mse:.4f}")

if __name__ == "__main__":
    main()