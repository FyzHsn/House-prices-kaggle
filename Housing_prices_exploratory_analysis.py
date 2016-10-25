# -*- coding: utf-8 -*-
"""
In this script, I perform exploratory data analysis on the Ames, Iowa housing
data set. This is part of a kaggle competition. I follow various kernels on the
kaggle website which offer a lot of useful lessons.

Author: Faiyaz Hasan
Date: October 17, 2016
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.mlab as mlab

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
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

#####################################################
# 2. EXPLORATORY DATA ANALYSIS + DATA PREPROCESSING #
#####################################################
y = df['saleprice'].values

plt.figure()
plt.subplot(211)
plt.hist(y, bins=30, facecolor='green', alpha=0.75)
plt.title('Sale price distribution')
plt.xlabel('Sale price (USD)')
plt.ylabel('# of sold houses')

plt.subplot(212)
plt.hist(np.log1p(y), bins=30, facecolor='blue', alpha=0.75)
plt.xlabel('Log of sale price')
plt.ylabel('# of sold houses')
plt.tight_layout()
#plt.savefig('PriceDistribution.png')
#plt.clf()
plt.show()

df_new = df.copy() # Create new set of log transformed data
df_new['saleprice'] = np.log1p(df['saleprice'])


# find features that are numeric
numeric_feats = df.dtypes[df.dtypes != 'object'].index
skewness = df[numeric_feats].apply(lambda x: skew(x.dropna()))
skewed_feats = skewness[skewness > 0.75].index

# log transform skewed feature vectors
df_new[skewed_feats] = np.log1p(df[skewed_feats])

# break down numeric features into 4 batches for visibility - 32 cols.
numeric_feats = list(numeric_feats.difference(['id', 'saleprice']))
batch1 = numeric_feats[0:9]
batch2 = numeric_feats[9:18]
batch3 = numeric_feats[18:27]
batch4 = numeric_feats[27:36]
batch1.append('saleprice')
batch2.append('saleprice')
batch3.append('saleprice')
batch4.append('saleprice')

# correlation matrix of numeric features
cor_mat = np.corrcoef(df_new[batch4].values.T)
sns.set(font_scale=1.5)
heat_map = sns.heatmap(cor_mat,
                       cbar=True,
                       annot=True,
                       fmt='.2f',
                       annot_kws={'size': 12},
                       yticklabels=batch4,
                       xticklabels=batch4)
plt.title('Correlation Matrix - Batch 4 columns')
plt.show()

"""
Variables of interest in prediction model based on correlation coeffs.
1. overallqual - 0.82
2. grlivarea - 0.73
3. garagecars - 0.68
4. garagearea - 0.65
5. 1stflrsf - 0.61
6. fullbath - 0.59
7. yearbuilt - 0.59
8. yearremodadd - 0.57
9. totrmsabvgrd - 0.53
10. fireplaces - 0.49
11. openporchsf - 0.46
12. lotarea - 0.40
13. totalbsmtsf - 0.37
14. wooddecksf - 0.34
15. halfbath - 0.31

"""

# get dummy variables for nominal variables
df_new = pd.get_dummies(df_new)

X = \
df_new.loc[:, [i for i in skewed_feats.difference(['saleprice'])]].values # Matrix containing log transformed skewed features
y = df_new['saleprice'].values # Target variable - log of sale price of house

# missing values - Impute via mean
df_new[numeric_feats].isnull().sum()
imr = Imputer(missing_values='NaN', strategy='mean', axis=0)
df_new = imr.fit_transform(df_new[numeric_feats])







