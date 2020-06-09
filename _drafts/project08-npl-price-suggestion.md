---
title: "Predicting Item Prices From Descriptions"
excerpt: "This project is my entry in Kaggle's Mercari Price Suggestion Challenge,  <br/><img src='/images/mercari/brand_ecdf.png'>"
collection: portfolio
---



<img src="/images/toxic/wordcloud.png" style="width:100%" />
# I. Definition
## Project Overview


```python
import os
os.chdir(r'D:\Python\Spyder\Kaggle Projects\Mercari')

import pandas as pd
import operator
import matplotlib.pyplot as plt
import numpy as np
import nltk
import time

nltk.data.path.append(r'D:\Python\Data Sets\nltk_data')
```


```python
start = time.time()
def print_time(start):
    time_now = time.time() - start
    minutes = int(time_now / 60)
    seconds = int(time_now % 60)
    if seconds < 10:
        print('Elapsed time was %d:0%d.' % (minutes, seconds))
    else:
        print('Elapsed time was %d:%d.' % (minutes, seconds))
```


```python
df = pd.read_csv('train.tsv', sep='\t')
df_sub = pd.read_csv('test.tsv', sep='\t')

submission = pd.DataFrame()
submission['test_id'] = df_sub.test_id.copy()

y_target = list(df.price)
```

## Impute Missing Values


```python
def null_percentage(column):
    df_name = column.name
    nans = np.count_nonzero(column.isnull().values)
    total = column.size
    frac = nans / total
    perc = int(frac * 100)
    print('%d%% or %d missing from %s column.' %
          (perc, nans, df_name))

def check_null(df, columns):
    for col in columns:
        null_percentage(df[col])

check_null(df, df.columns)

```

    0% or 0 missing from train_id column.
    0% or 0 missing from name column.
    0% or 0 missing from item_condition_id column.
    0% or 6327 missing from category_name column.
    42% or 632682 missing from brand_name column.
    0% or 0 missing from price column.
    0% or 0 missing from shipping column.
    0% or 4 missing from item_description column.



```python
def merc_imputer(df_temp):
    df_temp.brand_name = df_temp.brand_name.replace(np.nan, 'no_brand')
    df_temp.category_name = df_temp.category_name.replace(np.nan, 'uncategorized/uncategorized')
    df_temp.item_description = df_temp.item_description.replace(np.nan, 'No description yet')
    df_temp.item_description = df_temp.item_description.replace('No description yet', 'no_description')
    return df_temp

df = merc_imputer(df)
df_sub = merc_imputer(df_sub)
```


```python
print('Training Data')
check_null(df, df.columns)
print('Submission Data')
check_null(df_sub, df_sub.columns)
```

    Training Data
    0% or 0 missing from train_id column.
    0% or 0 missing from name column.
    0% or 0 missing from item_condition_id column.
    0% or 0 missing from category_name column.
    0% or 0 missing from brand_name column.
    0% or 0 missing from price column.
    0% or 0 missing from shipping column.
    0% or 0 missing from item_description column.
    Submission Data
    0% or 0 missing from test_id column.
    0% or 0 missing from name column.
    0% or 0 missing from item_condition_id column.
    0% or 0 missing from category_name column.
    0% or 0 missing from brand_name column.
    0% or 0 missing from shipping column.
    0% or 0 missing from item_description column.


# EDA

## Shipping


```python
df.shipping.value_counts()
```




    0    819435
    1    663100
    Name: shipping, dtype: int64




```python
print('%.1f%% of items have free shipping.' % ((663100 / len(df))*100))
```

    44.7% of items have free shipping.


Free shipping items should be priced higher because shipping is included in the price.

### Price


```python
df.columns
```




    Index(['train_id', 'name', 'item_condition_id', 'category_name', 'brand_name',
           'price', 'shipping', 'item_description'],
          dtype='object')




```python
print('$1 items: ' + str(df.price[df.price == 1].count()))
print('$2 items: ' + str(df.price[df.price == 2].count()))
print('$3 items: ' + str(df.price[df.price == 3].count()))
```

    $1 items: 0
    $2 items: 0
    $3 items: 18703


There is a minimum price of $3.


```python
plt.figure('Training Price Dist', figsize=(30,10))
plt.title('Price Distribution for Training - 3 Standard Deviations', fontsize=32)
plt.hist(df.price.values, bins=145, normed=False,
         range=[0, (np.mean(df.price.values) + 3 * np.std(df.price.values))])
plt.axvline(df.price.values.mean(), color='b', linestyle='dashed', linewidth=2)
plt.xticks(fontsize=24)
plt.yticks(fontsize=26)
plt.show()

print('Line indicates mean price.')
```


![png](/images/mercari/price_distribution.png)


    Line indicates mean price.


Most prices are on the lower end of the spectrum, and items priced above 145 are outliers that make up less that 0.3% of the data. Are there free items?


```python
print('Free items: %d, representing %.5f%% of all items.' %
      (df.price[df.price == 0].count(),
        (df.price[df.price == 0].count() / df.shape[0])))
```

    Free items: 874, representing 0.00059% of all items.


What does free even mean here?


```python
print('Free items where seller pays shipping: %d.' %
      df.price[operator.and_(df.price == 0, df.shipping == 1)].count())
```

    Free items where seller pays shipping: 315.


This is a tiny outlier. And it seems like some items the sellers actually paid to give away. I'd like to see how many items are listed for a low price but the seller is actually making money off shipping to avoid fees, a common eBay practice. Unfortunately, without data about the actual shipping price, we can't extrapolate any insights here. My approach would be to look at items that are priced lower than average yet have higher than average shipping prices for their name and descriptions.


```python
print('No description:', str(df.item_description[df.item_description == 'no_description'].count()))
print('Uncategorized:',str(df.category_name[df.category_name == 'uncategorized/uncategorized'].count()))
```

    No description: 82493
    Uncategorized: 6327


 Many items lack a description, but few lack a category.

### Category Name


```python
cat_counts = np.sort(df.category_name.value_counts())
print(str(len(cat_counts)) + ' categories total.')
print(str(df.shape[0]) + ' records total.')
print('Category frequency percentiles, marked by lines: \n25%%: %d, 50%%: %d, 75%%: %d, 95%%: %d, 97.5%%: %d.' %
     (cat_counts[int(len(cat_counts)*0.25)],
      cat_counts[int(len(cat_counts)*0.5)],
      cat_counts[int(len(cat_counts)*0.75)],
      cat_counts[int(len(cat_counts)*0.9)],
      cat_counts[int(len(cat_counts)*0.95)]))

title = 'Category Quantity ECDF Without Top 15 Outliers'
plt.figure(title, figsize=(30,10))
plt.title(title, fontsize=32)
x = np.sort(df.category_name.value_counts())
x = x[0:-15]
y = np.arange(1, len(x) + 1) / len(x)
plt.plot(x, y, marker='.', linestyle='none')
plt.xticks(fontsize=24)
plt.yticks(fontsize=26)
plt.axvline(x=x[int(len(x)*0.25)], linewidth=1, color='b')
plt.axvline(x=x[int(len(x)*0.5)], linewidth=1, color='b')
plt.axvline(x=x[int(len(x)*0.75)], linewidth=1, color='b')
plt.axvline(x=x[int(len(x)*0.95)], linewidth=1, color='b')
plt.axvline(x=x[int(len(x)*0.975)], linewidth=1, color='b')
plt.show()
```

    1288 categories total.
    1482535 records total.
    Category frequency percentiles, marked by lines:
    25%: 10, 50%: 76, 75%: 593, 95%: 2509, 97.5%: 5708.



![png](/images/mercari/category_ecdf.png)



```python
print('The top 75%% of categories represent %.1f%% of the dataset, and the top 50%% represent %.1f%%.' %
      ((sum([count for count in cat_counts if count > 10]) / len(df))*100,
       (sum([count for count in cat_counts if count > 76]) / len(df))*100))
```

    The top 75% of categories represent 99.9% of the dataset, and the top 50% represent 99.2%.


There are a lot of uncommon or unique categories that make up a small percentage of the data. If dimensionality reduction needs to happen here, I think it would be safe to keep only the top half of category names and the remaining ~10th of a percent of data will be grouped together as items with an uncommon category.


```python
title = 'Top 35 Categories'
plt.figure(title, figsize=(30,10))
df.category_name.value_counts()[0:35].plot(kind='bar')
plt.title(title, fontsize=30)
plt.yticks(fontsize=18)
plt.xticks(fontsize=18, rotation=35, ha='right')
plt.show()
```


![png](/images/mercari/top_35_cat.png)


## Brand Name


```python
brand_counts = np.sort(df.brand_name.value_counts())
print(str(len(brand_counts)) + ' brands total.')
print(str(df.shape[0]) + ' records total.')
print('Category frequency percentiles, marked by lines: \n25%%: %d, 50%%: %d, 75%%: %d, 95%%: %d, 97.5%%: %d.' %
     (brand_counts[int(len(brand_counts)*0.25)],
      brand_counts[int(len(brand_counts)*0.5)],
      brand_counts[int(len(brand_counts)*0.75)],
      brand_counts[int(len(brand_counts)*0.9)],
      brand_counts[int(len(brand_counts)*0.95)]))

title = 'Brand Quantity ECDF Without Top 25 Outliers'
plt.figure(title, figsize=(30,10))
plt.title(title, fontsize=32)
x = np.sort(df.brand_name.value_counts())
x = x[0:-25]
y = np.arange(1, len(x) + 1) / len(x)
plt.plot(x, y, marker='.', linestyle='none')
plt.xticks(fontsize=24)
plt.yticks(fontsize=26)
plt.axvline(x=x[int(len(x)*0.25)], linewidth=1, color='b')
plt.axvline(x=x[int(len(x)*0.5)], linewidth=1, color='b')
plt.axvline(x=x[int(len(x)*0.75)], linewidth=1, color='b')
plt.axvline(x=x[int(len(x)*0.95)], linewidth=1, color='b')
plt.axvline(x=x[int(len(x)*0.975)], linewidth=1, color='b')
plt.show()
```

    4810 brands total.
    1482535 records total.
    Category frequency percentiles, marked by lines:
    25%: 1, 50%: 4, 75%: 23, 95%: 131, 97.5%: 396.



![png](/images/mercari/brand_ecdf.png)



```python
print('The top 75%% of categories represent %.1f%% of the dataset, and the top 50%% represent %.1f%%.' %
      ((sum([count for count in brand_counts if count > 1]) / len(df))*100,
       (sum([count for count in brand_counts if count > 4]) / len(df))*100))
```

    The top 75% of categories represent 99.9% of the dataset, and the top 50% represent 99.7%.


A story similar to category_name.


```python
print('%d items, or %.2f%%, are missing a brand name.' %
      (len(df[df.brand_name == 'no_brand']),
       len(df[df.brand_name == 'no_brand']) / len(df)))
```

    632682 items, or 0.43%, are missing a brand name.



```python
title = 'Top 35 Brands'
plt.figure(title, figsize=(30,10))
df.brand_name.value_counts()[1:70].plot(kind='bar')
plt.title(title, fontsize=30)
plt.yticks(fontsize=18)
plt.xticks(fontsize=18, rotation=45, ha='right')
plt.show()
```


![png](/images/mercari/top_35_brand.png)


The most popular brands, PINK and Nike, are an order of magnitude less frequent than unbranded items. It seems there's a mix of company brands and individual product line brands, as we can see both Victoria's Secret and Pink as well as Nintendo and Pokemon.


```python
title = 'Top Half of Brands'
plt.figure(title, figsize=(30,10))
df.brand_name.value_counts()[50:2500].plot(kind='bar')
plt.title(title, fontsize=30)
plt.yticks(fontsize=18)
plt.xticks(fontsize=0, rotation=45, ha='right')
plt.show()
```


![png](/images/mercari/top_half_brands.png)


An exponential growth curve that explodes at the end. I just like making huge charts like this.


```python
df.columns
```




    Index(['train_id', 'name', 'item_condition_id', 'category_name', 'brand_name',
           'price', 'shipping', 'item_description'],
          dtype='object')



# Preprocessing

## Natural Language Processing


```python
import nltk
nltk.data.path.append(r'D:\Python\Data Sets\nltk_data')
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse
```

### Category Name

This is pretty straightforward. Make dummy categories for each categorical value, with uncommon values just zero. But since this dataset is big and there's a large number of categories, the best way to use this is to use CountVectorizer() because it returns a sparse matrix instead of a dense one.


```python
cat_vec = CountVectorizer(stop_words=[stopwords, string.punctuation], max_features=int(len(cat_counts)*0.75))
cat_matrix = cat_vec.fit_transform(df.category_name)
cat_matrix_sub = cat_vec.transform(df_sub.category_name)
```


```python
# For exploring the tokens. The array is an array inside of an array of one, ravel pulls it out.
cat_tokens = list(zip(cat_vec.get_feature_names(), np.array(cat_matrix.sum(axis=0)).ravel()))
```

### Brand Name


```python
brand_vec = CountVectorizer(stop_words=[stopwords, string.punctuation], max_features=int(len(brand_counts)*0.75))
brand_matrix = brand_vec.fit_transform(df.brand_name)
brand_matrix_sub = brand_vec.transform(df_sub.brand_name)
```


```python
brand_tokens = list(zip(brand_vec.get_feature_names(), np.array(brand_matrix.sum(axis=0)).ravel()))
```

### Item Name

Item name and description are more complicated. As they are phrases and sentences, the number of words is going to be exponentially larger and the words themselves don't hold equal weight. I'm going to use a statistical method called Term Frequency - Inverse Document Frequency (TF-IDF) that combines the bag of words approach with a weight adjustment based on the overall frequency of each term in the dataset.


```python
name_vec = TfidfVectorizer(min_df=15, stop_words=[stopwords, string.punctuation])
name_matrix = name_vec.fit_transform(df.name)
name_matrix_sub = name_vec.transform(df_sub.name)
```


```python
print('Kept %d words.' % len(name_vec.get_feature_names()))
```

    Kept 14407 words.


### Description


```python
desc_vec = TfidfVectorizer(max_features=100000,
                           stop_words=[stopwords, string.punctuation])
desc_matrix = desc_vec.fit_transform(df.item_description)
desc_matrix_sub= desc_vec.transform(df_sub.item_description)
```

### Condition and Shipping


```python
cond_matrix = sparse.csr_matrix(pd.get_dummies(df.item_condition_id, sparse=True, drop_first=True))
cond_matrix_sub = sparse.csr_matrix(pd.get_dummies(df_sub.item_condition_id, sparse=True, drop_first=True))
```


```python
ship_matrix = sparse.csr_matrix(df.shipping).transpose()
ship_matrix_sub = sparse.csr_matrix(df_sub.shipping).transpose()
```

### Combine Sparse Matrices


```python
sparse_matrix = sparse.csr_matrix(sparse.hstack([cat_matrix, brand_matrix, name_matrix, desc_matrix,
                               cond_matrix, ship_matrix]))
sparse_matrix_sub = sparse.csr_matrix(sparse.hstack([cat_matrix_sub, brand_matrix_sub, name_matrix_sub,
                                   desc_matrix_sub, cond_matrix_sub, ship_matrix_sub]))
```


```python
if sparse_matrix.shape[1] == sparse_matrix_sub.shape[1]:
    print('Features check out.')
else:
    print("The number of features in training and test set don't match.")
```

    Features check out.


### Garbage Collection


```python
import gc
del(cat_matrix, brand_matrix, name_matrix, desc_matrix, cond_matrix, ship_matrix)
del(cat_matrix_sub, brand_matrix_sub, name_matrix_sub, desc_matrix_sub, cond_matrix_sub, ship_matrix_sub)
del(df, df_sub)
gc.collect()
```




    169




```python
print_time(start)
```

    Elapsed time was 2:01.


# Training


```python
def rmsle(true, pred):
    assert len(pred) == len(true)
    return np.sqrt(np.mean(np.power(np.log1p(pred)-np.log1p(true), 2)))
```

Take the log of the target data to boost training accuracy.


```python
y_target = np.log10(np.array(y_target) + 1)
```

Split training and test set


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(sparse_matrix, y_target, test_size = 0.1)
```

### Ridge Regression


```python
start = time.time()
from sklearn.linear_model import Ridge
reg_ridge = Ridge(solver='sag', alpha=5)
reg_ridge.fit(X_train, y_train)
y_pred = reg_ridge.predict(X_test)
print(rmsle(10 ** y_test - 1, 10 ** y_pred - 1))
print_time(start)
```

    0.476321830094
    Elapsed time was 0:38.


### Lasso


```python
'''
start = time.time()
from sklearn.linear_model import Lasso
reg_lasso = Lasso(alpha=1.0)
reg_lasso.fit(X_train, y_train)
y_pred = reg_lasso.predict(X_test)
print(rmsle(10 ** y_test - 1, 10 ** y_pred - 1))
print_time(start)
'''
```




    '\nstart = time.time()\nfrom sklearn.linear_model import Lasso\nreg_lasso = Lasso(alpha=1.0)\nreg_lasso.fit(X_train, y_train)\ny_pred = reg_lasso.predict(X_test)\nprint(rmsle(10 ** y_test - 1, 10 ** y_pred - 1))\nprint_time(start)\n'



### Random Forest


```python
'''start = time.time()
from sklearn.ensemble import RandomForestRegressor
reg_rf = RandomForestRegressor(n_estimators=800, max_depth=10)
reg_rf.fit(X_train, y_train)
y_pred = reg_rf.predict(X_test)
print(rmsle(10 ** y_test - 1, 10 ** y_pred - 1))
print_time(start)
'''
```




    'start = time.time()\nfrom sklearn.ensemble import RandomForestRegressor\nreg_rf = RandomForestRegressor(n_estimators=800, max_depth=10)\nreg_rf.fit(X_train, y_train)\ny_pred = reg_rf.predict(X_test)\nprint(rmsle(10 ** y_test - 1, 10 ** y_pred - 1))\nprint_time(start)\n'



### LightGBM


```python
import lightgbm as lgb
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
```


```python
def rmsle_lgb(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    return np.sqrt(np.mean(np.power(np.log1p(y_true + 1) - np.log1p(y_pred + 1), 2)))
```


```python
start = time.time()
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    'verbose': 0,
    'num_leaves': 31,
    'n_estimators': 1000,
    'learning_rate': 0.5,
    'max_depth': 10,
}

reg_lgbm = lgb.LGBMRegressor(**params)
reg_lgbm.fit(X_train, y_train)
y_pred = reg_lgbm.predict(X_test)

print(rmsle(10 ** y_test - 1, 10 ** y_pred - 1))
print_time(start)
```

    0.45267428996
    Elapsed time was 10:19.


### Stacking

http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html

Stacking Function. This function returns dataframes of predictions from each input model that can be merged into the train, test, and submission datasets. I used scikit-learn's 'clone' method, so all the weights will be stripped from the input models. This lets you input either fresh models or previously used models without worrying about it. A cloned model is generated for each training fold. I learned how to do this here, and it's an article worth reading:
http://blog.kaggle.com/2016/12/27/a-kagglers-guide-to-model-stacking-in-practice/

In my experiments, I've gotten 2-3% better performance from stacking a few models together than the best model does on it's own. Interestingly, even adding a crap model (like the Lasso regression with a score of 0.75) increases the overall performance. The stacked models seem to work better with tree based models than regression models.


```python
print('hi')
```

    hi



```python
from sklearn.model_selection import KFold
from sklearn.base import clone
def stack_predictions(X_train, y_train, X_test, submit, K, *models):
    train_preds = pd.DataFrame(index=np.array(range(X_train.shape[0])))
    test_preds = pd.DataFrame(index=np.array(range(X_test.shape[0])))
    submit_preds = pd.DataFrame(index=np.array(range(submit.shape[0])))
    folds = KFold(n_splits=K, shuffle=True)

    fold_n = 0
    train_folds = np.zeros(len(train_preds))
    for train_index, test_index in folds.split(X_train):
        train_folds[test_index] = fold_n
        fold_n += 1

    fold_n = 0
    test_folds = np.zeros(len(test_preds))
    for train_index, test_index in folds.split(X_test):
        test_folds[test_index] = fold_n
        fold_n += 1

    fold_n = 0
    submit_folds = np.zeros(len(submit_preds))
    for train_index, test_index in folds.split(submit):
        submit_folds[test_index] = fold_n
        fold_n += 1

    for m, model in enumerate(models):
        print('Selecting model %d.' % (m+1))
        col = 'pred_col_' + str(m)
        train_preds[col] = np.nan
        test_preds[col] = np.nan
        submit_preds[col] = np.nan

        for fold in range(K):
            print('Processing a fold...')
            current_model = clone(model)
            current_model.fit(X_train[np.where(train_folds!=fold)], y_train[np.where(train_folds!=fold)])

            train_preds[col].iloc[np.where(train_folds==fold)] = current_model.predict(
                X_train[np.where(train_folds==fold)])

            test_preds[col].iloc[np.where(test_folds==fold)] = current_model.predict(
                X_test[np.where(test_folds==fold)])

            submit_preds[col].iloc[np.where(submit_folds==fold)] = current_model.predict(
                submit[np.where(submit_folds==fold)])  

    return train_preds, test_preds, submit_preds
```


```python
from sklearn.model_selection import KFold
from sklearn.base import clone

def stack_predictions(X_train, y_train, X_test, submit, K, *models):
    train_preds = pd.DataFrame(index=np.array(range(X_train.shape[0])))
    test_preds = pd.DataFrame(index=np.array(range(X_test.shape[0])))
    submit_preds = pd.DataFrame(index=np.array(range(submit.shape[0])))
    folds = KFold(n_splits=K, shuffle=True)

    fold_n = 0
    train_folds = np.zeros(len(train_preds))
    for train_index, test_index in folds.split(X_train):
        train_folds[test_index] = fold_n
        fold_n += 1

    fold_n = 0
    test_folds = np.zeros(len(test_preds))
    for train_index, test_index in folds.split(X_test):
        test_folds[test_index] = fold_n
        fold_n += 1

    fold_n = 0
    submit_folds = np.zeros(len(submit_preds))
    for train_index, test_index in folds.split(submit):
        submit_folds[test_index] = fold_n
        fold_n += 1

    for m, model in enumerate(models):
        print('Selecting model %d.' % (m+1))
        col = 'pred_col_' + str(m)
        train_preds[col] = np.nan
        test_preds[col] = np.nan
        submit_preds[col] = np.nan

        for fold in range(K):
            print('Processing a fold...')
            current_model = clone(model)
            current_model.fit(X_train[np.where(train_folds!=fold)], y_train[np.where(train_folds!=fold)])

            train_preds[col].iloc[np.where(train_folds==fold)] = current_model.predict(
                X_train[np.where(train_folds==fold)])

            test_preds[col].iloc[np.where(test_folds==fold)] = current_model.predict(
                X_test[np.where(test_folds==fold)])

            submit_preds[col].iloc[np.where(submit_folds==fold)] = current_model.predict(
                submit[np.where(submit_folds==fold)])  

    return train_preds, test_preds, submit_preds
```

Create models


```python
reg_ridge = Ridge(solver='sag', alpha=5)
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    'verbose': 0,
    'num_leaves': 31
}

reg_lgbm = lgb.LGBMRegressor(**params)
```


```python
start = time.time()
train_preds, test_preds, sub_preds = stack_predictions(X_train, y_train, X_test,
                                                       sparse_matrix_sub, 10,
                                                       reg_ridge, reg_lgbm)
print_time(start)
```


    Elapsed time was 9:26.



```python
X_train_stacked = sparse.csr_matrix(sparse.hstack([X_train, sparse.csr_matrix(train_preds)]))
X_test_stacked = sparse.csr_matrix(sparse.hstack([X_test, sparse.csr_matrix(test_preds)]))
sub_stacked = sparse.csr_matrix(sparse.hstack([sparse_matrix_sub, sparse.csr_matrix(sub_preds)]))
```


```python
start = time.time()
from sklearn.linear_model import Ridge
reg_ridge = Ridge(solver='sag', alpha=5)
reg_ridge.fit(X_train_stacked, y_train)
y_pred = reg_ridge.predict(X_test_stacked)
print(rmsle(10 ** y_test - 1, 10 ** y_pred - 1))
print_time(start)
```

    0.473045784616
    Elapsed time was 0:51.



```python
start = time.time()
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    'verbose': 0,
    'num_leaves': 31,
    'n_estimators': 1000,
    'learning_rate': 0.5,
    'max_depth': 10,
    'random_seed'
}

reg_lgbm = lgb.LGBMRegressor(**params)
reg_lgbm.fit(X_train_stacked, y_train)
r_pred = reg_lgbm.predict(X_test_stacked)

print(rmsle(10 ** y_test - 1,10 ** r_pred - 1))
print_time(start)
```

    0.442803533309
    Elapsed time was 9:12.


Another LGBM model trained with the additional features from the input Ridge and LGBM models. I think it can evaluate how accurate the LGBM and Ridge predictions are along with the context of all the previous features, and make a more informed prediction.



```python
pred_sub = reg_lgbm.predict(sub_stacked)
lightgbm_submission = submission.copy()
lightgbm_submission['price'] = pd.DataFrame(10 ** pred_sub - 1)

lightgbm_submission.to_csv('lightgbm_test_2.csv', index=False)
```

My final leaderboard score was 0.44072, which placed me at 420 out of 2,382. I was pretty happy with this because I don't really do Kaggle challenges to win, but rather to experiment with fun strategies.
