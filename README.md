Depop Listing Engagement Analysis
📌 Project Overview

User engagement is a key driver of visibility and sales in marketplace platforms.
This project analyzes public Depop listings to understand which listing attributes are most strongly associated with higher engagement (measured by likes) for vintage women’s t-shirts.

The goal is not perfect prediction, but to identify actionable signals that could inform:

seller listing strategies

marketplace ranking logic

experimentation hypotheses

❓ Key Question

What listing characteristics are most associated with high engagement, and how much signal exists in basic listing metadata?

📊 Data

Publicly available Depop listing data

Features include:

price

brand

condition

time since posting

Engagement measured as number of likes

Listings labeled as high vs low engagement using the median likes threshold

🧠 Approach

Exploratory data analysis (EDA) to understand distributions and relationships

Data cleaning and feature engineering

Regression modeling to estimate engagement magnitude

Classification modeling to identify high-engagement listings

Feature importance analysis using model coefficients

🔍 Key Findings

Brand recognition is the strongest predictor of engagement

Item condition is consistently associated with higher likes

Higher prices generally reduce engagement likelihood

Even simple listing metadata contains meaningful signal, despite the noisy and long-tailed nature of marketplace engagement data

📈 Model Performance (Context, Not the Point)

Regression MAE ≈ 23 likes, serving as a reasonable baseline given engagement volatility

Classification accuracy ≈ 63%, outperforming a naive baseline (~60%)

These results suggest the models are useful for directional insight, not precise prediction.

💡 How This Could Be Used

Help sellers optimize listings for visibility

Inform marketplace experiments around ranking or recommendations

Generate hypotheses for A/B testing (e.g., brand weighting, price sensitivity)

🔜 Next Steps

If extended further, this analysis could incorporate:

image-based features

seller history

time-of-day or recency effects

experimental validation via A/B testing

🛠️ Tools

Python (pandas, NumPy, scikit-learn)
Matplotlib
Jupyter Notebook

📂 View the Analysis

HTML report (recommended): reports/01_eda_engagement.html

Notebook source: notebooks/01_eda_engagement.ipynb

nbviewer:
https://nbviewer.org/github/savmul/depop-engagement-analysis/blob/main/notebooks/01_eda_engagement.ipynb
- **Notebook source (GitHub)**: `notebooks/01_eda_engagement.ipynb`




