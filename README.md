House-prices-kaggle competition
===============================

This repository contains my work for the kaggle competition involving the housing prices in Ames, Iowa based on 79 different variables. My objective is to build a good regression model that generalizes well to unseen data. Further themes that will probably come up as I go deep into the project are: feature engineering, XGBoosting, multivariate regression. The metric that I wish to use for quantifying the model performance the mean squared error. 

Exploratory Data Analysis
-------------------------

### Sale price distribution
Here, we look at the distribution of the sale prices of the house. The log transformed sale price (it seems to be more 'normal') is also shown.  
![](https://github.com/FyzHsn/House-prices-kaggle/blob/master/Figs/PriceDistribution.png?raw=true)  
Since regression works best on unskewed normal datasets, we will consider applying log transforms to reduce the skewness of the feature vectors.  

Linear Regression Model
-----------------------

### saleprice vs overallqual
We apply single variable linear regression to `overallqual` and `saleprice`. `overallqual` has the highest correlation with `saleprice` with a value of 0.82. The second highest is `grlivarea` with a correlation value of 0.73.
