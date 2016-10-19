# -*- coding: utf-8 -*-
"""


Author: Faiyaz Hasan
Date: October 17, 2016
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.mlab as mlab

from sklearn.model_selection import train_test_split
from scipy.stats import skew

###################
# 0. LOAD DATASET #
###################
train_file_loc = r'C:\Users\Windows\Dropbox\AllStuff\Housing_price_regression\Data\train.csv'
df = pd.read_csv(train_file_loc)

##############################################################
# 1. STRUCTURE OF DATASET - DIMENSIONS, TARGET VARIABLE ETC. #
##############################################################
df.shape # Data frame has 1460 rows and 81 feature vectors
df.columns # Column names - SalePrice is the target variable

df.columns = [colnames.lower() for colnames in df.columns] # lowercase

#########################
# 2. DATA PREPROCESSING #
#########################
X = df.iloc[:, :-1].values # Matrix containing feature vectors
y = np.log(df['saleprice'].values) # Target variable - log of sale price of house


################################
# 3. EXPLORATORY DATA ANALYSIS #
################################
plt.figure()
plt.subplot(211)
plt.hist(y, bins=30, facecolor='green', alpha=0.75)
plt.title('Sale price distribution')
plt.xlabel('Sale price (USD)')
plt.ylabel('# of sold houses')

plt.subplot(212)
plt.hist(np.log(y), bins=30, facecolor='blue', alpha=0.75)
plt.xlabel('Log of sale price')
plt.ylabel('# of sold houses')
#plt.savefig('PriceDistribution.png')
#plt.clf()
plt.show()

# find features that are numeric
numeric_feats = df.dtypes[df.dtypes != 'object'].index
skewness = df[numeric_feats].apply(lambda x: skew(x.dropna()))












