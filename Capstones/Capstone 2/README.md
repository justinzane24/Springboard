# Time Series Forecasting with Sales Data

## Overview

This project explores the use of time series forecasting techniques to predict future sales based on historical transactional data. The goal was to evaluate classical statistical models (SARIMA, SARIMAX) and deep learning models (LSTM) to determine which approach delivers the most accurate and practical forecasts.

## Business Objective

Accurate sales forecasts can help drive better decisions around inventory management, staffing, promotions, and revenue planning. This project aims to recommend the best-performing model that can reliably forecast monthly or weekly sales, potentially informing strategy in a retail or e-commerce setting.

## Data

- **Source:** Global Superstore dataset (sales transactions across multiple years)
- **Time Frame:** 4 years of daily data
- **Key Variables:**
  - Sales
  - Order Date
  - Country, Region
  - Segment, Category, Sub-Category
- **Target Variable:** Sales (transformed using log scale for modeling)

## Methodology

1. **Data Preprocessing:**
   - Handled missing values and outliers
   - Resampled data at monthly and weekly frequencies
   - Engineered time-based features (year, month, quarter, etc.)

2. **Exploratory Data Analysis:**
   - Seasonal patterns and autocorrelation identified
   - ADF test confirmed stationarity of the log-transformed monthly sales

3. **Models Used:**
   - **SARIMA**: Seasonal ARIMA with (p,d,q)x(P,D,Q,12)
   - **SARIMAX**: Extension of SARIMA with exogenous variables
   - **LSTM**: Sequence-based neural network (PyTorch)

4. **Hyperparameter Tuning:**
   - SARIMA/SARIMAX tuned manually using AIC and residual analysis
   - LSTM tuned using Ray Tune (limited by hardware)

5. **Evaluation Metrics:**
   - MAPE (Mean Absolute Percentage Error)
   - RMSE
   - Visual inspection of forecast vs. actual

## Results

- **SARIMA**: Best overall performance with ~11% MAPE on monthly data. Simple to implement and computationally efficient.
- **SARIMAX**: Slightly underperformed SARIMA but showed potential if more meaningful exogenous features were added.
- **LSTM**: MAPE of 36.48%, underperformed likely due to:
  - Training on only 10% of data
  - Weekly resampling (noisier)
  - No intermediate validation due to time constraints

## LSTM Notes

Due to hardware and time constraints, only ~10% of the dataset was used to train the LSTM model. Evaluation checkpoints were omitted, as each would have added ~2.5 hours per training run. Despite this, the model demonstrated learning progress, though its MAPE was higher than classical models. The weekly resampling likely introduced more noise compared to the monthly SARIMA inputs.

## Recommendations

- **Deploy SARIMA** as the current best option — strong performance, low complexity, and no need for external data.
- **Re-evaluate SARIMAX and LSTM** with:
  - Daily or weekly granularities
  - Additional features like promotions, holidays, pricing
  - More computing power for full dataset training and evaluation
- **Future Work:** Explore Prophet, transformer-based models, or hybrid ensembles for long-range forecasting.
