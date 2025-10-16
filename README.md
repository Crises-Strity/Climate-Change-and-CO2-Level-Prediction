# Climate Change and CO₂ Level Prediction (ARIMA)

A research-grade repository that analyzes and forecasts atmospheric CO₂ concentration using a **statistical time-series model (ARIMA)**.  
This project develops a quantitative framework applying **data preprocessing, visualization, and stationarity tests** to identify trend and seasonality patterns in a dataset spanning **1958–2001 (2,284 observations)**, and fits an **ARIMA(1,1,1)** model to forecast future CO₂ levels. The model achieves a **Mean Squared Error (MSE) as low as 0.57** on the held-out evaluation, indicating strong predictive performance.

> Notebook: `notebooks/Climate Change and CO2 Level Prediction Project.ipynb`  
> Raw data: `data/co2.csv`

---

## 📦 Repository Structure
```
climate-co2-prediction/
├── data/
│   └── co2.csv                   # Raw dataset
├── notebooks/
│   └── Climate Change and CO2 Level Prediction Project.ipynb
├── src/
│   └── model_arima.py            # Runnable ARIMA training & forecasting script
├── results/                      # Auto-generated: metrics, plots, forecasts
├── requirements.txt
├── .gitignore
├── LICENCE
└── README.md
```

## 🔬 Methodology
- **Model**: ARIMA(1,1,1) (p=1, d=1, q=1)
- **Pipeline**:
  1. Data loading & preprocessing (date parsing, frequency inference, missing value handling)
  2. Exploratory visualization & stationarity checks (in the notebook)
  3. Train/test split on chronological order
  4. Model fitting and multi-step forecasting
  5. Evaluation with MSE and visual comparison

## 📊 Results (Reported)
- **MSE**: *≈ 0.57* on the evaluation split
- Visual inspection shows the forecast tracks the underlying seasonal-trend structure (see `results/forecast_plot.png` after running).

> Note: Your exact score may differ depending on the chosen split and any additional preprocessing.

## 🧰 Installation
Create a virtual environment and install dependencies:
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## ▶️ Quick Start
Run the ARIMA script directly (adjust column names as needed):
```bash
python src/model_arima.py   --data_path data/co2.csv   --date_col date   --value_col co2   --order 1 1 1   --test_size 0.2
```

Outputs (created under `results/`):
- `forecast.csv` – dates, actuals, forecasts on the test set  
- `metrics.json` – MSE and run metadata  
- `forecast_plot.png` – line plot of Train/Test/Forecast

### Common adjustments
- If your CSV uses different column names, pass `--date_col` / `--value_col`.
- If your dataset is monthly but lacks an explicit frequency, the script will infer it (defaults to `MS`).
- For alternative ARIMA orders, change `--order p d q`.

## 🗂️ Dataset
- Time coverage: **1958–2001** (2,284 observations)  
- File: `data/co2.csv`  
- Please cite or reference the original data provider if applicable.

## 🚀 Future Work
- Add **seasonal ARIMA (SARIMA)** with seasonal order search
- Robust **model selection** via information criteria grid search
- **Exogenous regressors (ARIMAX)** using climate covariates
- **Residual diagnostics** and formal stationarity tests in code
- Reproducible **conda environment** and CI workflow

## 📝 Citation
If you use this repository in academic work, please cite it as:
```
Cris Wang. Climate Change and CO₂ Level Prediction (ARIMA). GitHub repository, 2025.
```
