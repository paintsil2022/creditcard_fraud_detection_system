import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, f1_score
from sklearn.linear_model import LinearRegression
import math
from itertools import combinations
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def preparing_data():
    data1 = pd.read_csv('fraudTest.csv')
    data2 = pd.read_csv('fraudTrain.csv')

    data = pd.concat([data1, data2])

    data = data.sample(frac=1, random_state=1).reset_index()
    data = data.head(n=100000)

    # doing some prelimenary adjustments to data set
    data = data.drop('Unnamed: 0', axis=1)
    data = data.drop('trans_num', axis=1)
    data = data.drop('first', axis=1)
    data = data.drop('last', axis=1)
    data = data.drop('street', axis=1)
    data = data.drop('city', axis=1)

    # calculating distance between credit card holder location and location of merchant
    data['distance'] = np.sqrt((data['lat'] - data['merch_lat'])**2 + (data['long'] - data['merch_long'])**2)

    # converting to date time
    data['trans_date_trans_time'] = pd.to_datetime(data['trans_date_trans_time'])
    data['hour'] = data['trans_date_trans_time'].apply(pull_hour)
    return data

# pulling the hour for a variable
def pull_hour(ts):
    return ts.hour



# using unix time we are going to calculate the sum of transaction amounts in past 30 days
# we will create two variables from this, first sum of transactions in 30 days
# next will be a interaction based variable between current purchase amount / 30 day total
# this will help measure if this transaction is out of the ordinary


# function to calculate last 30 day spending
def sum_30_day(unixtime, cc_num, data):
    unixstamp = unixtime
    minus30 = unixstamp - 2629743
    ccnum = cc_num
    sumtable = data.loc[(data["cc_num"] == ccnum) & (data['unix_time'] < unixstamp) & (data['unix_time'] > minus30)]
    history30 = sumtable['amt'].sum()
    return history30


def finalizing_data(data):
# running function and creating a new variable for it
    data['history_30'] = data.apply(lambda x: sum_30_day(x.unix_time, x.cc_num, data), axis=1)

# measuring interaction effect with amt in new variable
    data['interaction_30'] = data['history_30'] / data['amt']


# dropping non categorical variables in preperation for regression modeling

    data = data.drop('trans_date_trans_time', axis=1)
    data = data.drop('state', axis=1)
    data = data.drop('merchant', axis=1)
    data = data.drop('job', axis=1)
    data = data.drop('dob', axis=1)
    data = data.drop('category', axis=1)
    data = data.drop('gender', axis=1)
    data = data.drop('index', axis=1)

# creating a correlation heatmap
# using this we will check for any multicollinearity issues
# multicollinearity is when two variables have a correlation >0.7 with eachother


    # fig, ax = plt.subplots(figsize=(20,10))
# sns.heatmap(data.corr(),annot=True).set_title('Correlation')


# there is multicollinearity issues with our non generated predictors such as lat longs, we will drop all of these

    data = data.drop('cc_num', axis=1)
    data = data.drop('zip', axis=1)
    data = data.drop('lat', axis=1)
    data = data.drop('long', axis=1)
    data = data.drop('unix_time', axis=1)
    data = data.drop('merch_lat', axis=1)
    data = data.drop('merch_long', axis=1)
    data = data.drop('city_pop', axis=1)

    # fig, ax = plt.pltsubplots(figsize=(20,10))


    # all the multicollinearity issues are fixed, we are going to begin our fitting process with the data

    # we are going to use a logistic regression algorithim for this binary classification

    # we will then measure using an accuracy score to see if our model is working

    y = data['is_fraud']
    x = data.drop('is_fraud', axis=1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

    model = LogisticRegression()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    y_pred_proba = model.predict_proba(x_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    return (accuracy, x_test)



def train_test_model():
    data = preparing_data()
    return finalizing_data(data)




# print(accuracy)
# print(x_test)