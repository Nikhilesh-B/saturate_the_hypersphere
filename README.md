# Analytics in Finance

This repository contains a collection of analytical techniques, statistical models, and computational tools used to study financial data. It is designed to showcase practical implementations of concepts taught in advanced finance and analytics coursework, with an emphasis on reproducible workflows, clear methodology, and applied insights.

The analyses in this repository are implemented in Python using libraries commonly used in quantitative research, including NumPy, pandas, SciPy, statsmodels, scikit-learn, and arch.

---

## Contents

### 1. Time-Series Modeling and Forecasting
A set of models used to analyze and forecast financial time-series such as returns, volatility, and macroeconomic indicators.

Includes:
- AR, MA, ARMA, and ARIMA models with order selection (AIC/BIC) and full diagnostic workflows  
- Stationarity testing (ADF, KPSS), autocorrelation analysis, PACF/ACF interpretation  
- Rolling forecast frameworks for out-of-sample evaluation  
- Use cases: return prediction, seasonality detection, shock response analysis  

---

### 2. Volatility Modeling
Models focused on estimating and forecasting conditional variance, a core component of risk management and derivatives pricing.

Includes:
- GARCH(1,1) implementation using maximum-likelihood estimation  
- Residual diagnostics and goodness-of-fit evaluation  
- Rolling volatility forecasts and comparison to historical volatility  
- Optional extensions: Student-t innovations, EGARCH, and volatility clustering analysis  

---

### 3. Cross-Sectional Analysis and Factor Modeling
Tools for analyzing cross-sectional behavior of securities and understanding sources of return variation.

Includes:
- Construction of factor regressors using financial and accounting data  
- Ordinary least squares regressions with statistical inference  
- Residual analysis for alpha estimation  
- Application examples: testing factor exposures, examining return drivers, and building simple ranking signals  

---

### 4. Regression and Predictive Modeling
Applications of machine learning and statistical modeling to financial datasets.

Includes:
- Regularized regression (Ridge, Lasso) for prediction of continuous financial variables  
- Train/test splits respecting time-order constraints  
- Error metrics appropriate for financial data (MAE, RMSE, directional accuracy)  
- Use cases: earnings prediction, forecasting fundamentals, modeling spread behavior  

---

### 5. Portfolio Analytics
Analytical tools for understanding portfolio construction and risk.

Includes:
- Expected return and variance estimation  
- Covariance matrices and shrinkage techniques  
- Efficient frontier visualization  
- Simple optimization examples (mean–variance, minimum variance, and constrained optimization)  

---

## Repository Structure
