# Airbnn in D.C. Area
# Data Description

* File Name: listings.csv
* Description: Detailed Listings data for Washington, D.C.
* Data Size: 9,330 rows × 106 columns
* It contains 9329 accommodations with 106 different kinds of detailed features, which includes not only the basic information, but other detailed features that help us analyze deeper and wider: guests reviews scores (the total number of reviews, overall review scores, scores in various aspects), description (amenities, house rules, room space), time series data (the first host date, the first reviews date, the last reviews date), performance rate (response rate, acceptance rate, is a super host or not), just to name a few.

# Conclusion
The outcome is not ideal as we thought. We choose SVR-ploy as our best model as it has highest cross-validation score, which is 0.6282, but mean squared error is way too large and prediction is not useful. It maybe resulted by redundant features, which lead to overfitting.
