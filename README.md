# 📈 linear_regression_model_for_SPY

A Python project that analyzes and forecasts short‑term SPY ETF movements (tracking the S&P 500) using data from major global indices.

---

## 🔍 Project Overview

1. **Data Ingestion**  
   - Load historical OHLCV data for SPY and eight global indices (S&P 500, NASDAQ, DJIA, CAC 40, DAX, Nikkei, HSI, AORD) from CSV files.  
   - Compute daily returns as the difference between today’s open and the previous day’s open (or close for Asian markets).  
   - Consolidate into a single “panel” DataFrame and forward‑fill missing values.

2. **Modeling**  
   - Split data into **Train** and **Test** sets.  
   - Train an **Ordinary Least Squares (OLS)** regression to predict next‑day SPY returns using:  
     - SPY’s 1‑day lagged return  
     - Current returns of all other indices  
   - Evaluate performance with **R²**, **Adjusted R²** and **RMSE** on both splits.

3. **Strategy Simulation**  
   - **Long/Short Rule**:  
     - Go **Long** when predicted return > 0  
     - Go **Short** when predicted return < 0  
   - Compute **cumulative wealth** and compare against a **Buy & Hold** benchmark.  
   - Measure risk‑adjusted performance via **daily & annualized Sharpe Ratios**.  
   - Calculate **Maximum Drawdown** to assess peak‑to‑trough exposure.

4. **Outputs**  
   - Scatter‑matrix and actual vs. predicted plots  
   - Cumulative returns charts (strategy vs. buy‑&‑hold)  
   - Performance summary tables  
   - All figures saved under `plots/`

---

## 🚀 Getting Started

### 1. Prerequisites  
- Python 3.8 or higher  
- Git (optional, for cloning)

### 2. Installation  

```bash
git clone https://github.com/username/your-repo.git
cd your-repo
pip install -r requirements.txt
`````

### 3. Fetch Data

```bash
python request_data.py
`````

### 4. Train & Evaluate

```bash
python linear_regression.py
`````
This script will:

- Build and fit the OLS model

- Print R², Adjusted R², RMSE for Train/Test

- Simulate the long/short strategy and Buy & Hold

- Save all plots in plots/