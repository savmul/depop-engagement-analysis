## Results

### Regression
A linear regression model was used to predict listing engagement (likes).
The model achieved a mean absolute error (MAE) of approximately 23 likes.
Given the long-tailed and noisy nature of marketplace engagement data, this
provides a reasonable baseline rather than precise prediction.

### Classification
Listings were classified into high vs low engagement using the median number
of likes as a threshold. A logistic regression model achieved ~63% accuracy,
outperforming a naive baseline (~60%), indicating that listing metadata
contains meaningful predictive signal.

### Feature Importance
Brand and condition were the strongest predictors of engagement. Listings
associated with recognizable brands and better condition showed a higher
likelihood of strong engagement, while higher prices generally reduced
engagement probability.
