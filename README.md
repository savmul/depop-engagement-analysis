# Depop Listing Engagement Analysis

## View the Project
- GitHub-rendered report: `reports/01_eda_engagement.md`
- Notebook source (may not render on GitHub): `notebooks/01_eda_engagement.ipynb`

## Project Overview
This project analyzes public Depop listings to understand what factors 
drive
user engagement (likes) for vintage women’s t-shirts. Engagement was 
modeled
as both a regression problem (predicting likes) and a classification 
problem
(predicting high vs low engagement).

## Data
- Publicly available Depop listing data
- Features include price, brand, condition, and days since posting
- Engagement measured using number of likes
- Listings labeled as high or low engagement using the median likes 
threshold

## Methods
- Exploratory data analysis (EDA)
- Data cleaning and preprocessing
- Feature engineering
- Linear regression to predict engagement
- Logistic regression to classify high vs low engagement
- Feature importance analysis using model coefficients

## Results

### Regression
The regression model predicts listing engagement (likes) with a mean 
absolute
error (MAE) of approximately 23 likes. Given the long-tailed and noisy 
nature
of marketplace engagement data, this provides a reasonable baseline rather
than precise prediction.

### Classification
The classifier achieved approximately 63% accuracy, outperforming a naive
baseline (~60%). This indicates that listing metadata contains meaningful
signal for predicting engagement.

### Feature Importance
Brand recognition and item condition were the strongest predictors of
engagement. Listings associated with recognizable brands consistently 
showed
higher engagement, while higher prices generally reduced engagement 
likelihood.

## Tools Used
- Python
- pandas, NumPy
- scikit-learn
- Matplotlib
- Jupyter Notebook

## View the Analysis
GitHub may not reliably render `.ipynb` notebooks.

- HTML report (recommended): `reports/01_eda_engagement.html`
- Notebook source: `notebooks/01_eda_engagement.ipynb`


