# aemo-qld-price-forecasting
Synthetic AEMO-style electricity price forecasting project.

**AEMO-style Electricity Price Forecasting (Synthetic Data)**
📌 **Overview**
This project simulates an Australian NEM (AEMO)–style electricity price dataset for Queensland (QLD1) and builds a forecasting model to predict 30-minute spot prices.
The data contains realistic patterns including demand cycles, temperature effects, renewable penetration, and scarcity price spikes.

**🎯 Objectives**
--Generate synthetic but realistic AEMO-style electricity price & demand data
--Perform exploratory data analysis (EDA)
--Engineer time-series features
--Compare forecasting models:
--Naive (lag-1)
--Linear Regression
--Random Forest Regressor
--Interpret model errors and feature importance

**Data Description**

| Column            | Description                                            |
| ----------------- | ------------------------------------------------------ |
| `datetime`        | 30-min timestamp (2018–2022)                           |
| `region`          | QLD1                                                   |
| `demand_mw`       | Electricity demand with daily + yearly seasonality     |
| `temperature_c`   | Synthetic QLD temperature with seasonal pattern        |
| `renewable_share` | Solar + renewable mix (%)                              |
| `price`           | Final spot price driven by demand, renewables & spikes |


**👩‍💻 Feature Engineering**
--price_lag_1, price_lag_48
--demand_lag_1
--Hour, day of week, month
--Demand quantile bucket
--Rolling & seasonal signals

**Key Insights**
--Demand and renewable share are the strongest drivers of price.
--Forecasting is hardest during morning/evening peaks and during synthetic price spikes.
--Random Forest handles non-linear patterns in demand and temperature better than Linear Regression.


