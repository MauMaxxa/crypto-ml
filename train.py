# -*- coding: utf-8 -*-
"""

Use this class to train a basic machine learning model. 
Or modify it to incorporate your own machine learning models or pipelines using other libraries like XGBoost, Keras or strategies like Spiking Neural Networks!

"""

import numpy as np
from numpy import *
import pandas as pd

from binance.client import Client
from binance.enums import *

import matplotlib.pyplot as plt

import CoreFunctions as cf

from sklearn.ensemble import GradientBoostingClassifier
import sklearn.model_selection as ms
import sklearn.preprocessing as pp
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, mean_squared_error
from joblib import dump, load

import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
plt.rcParams.update({'figure.figsize':(7,5), 'figure.dpi':100})

def binanceDataFrame(klines):
    df = pd.DataFrame(np.array(klines).reshape(-1,12),dtype=float, columns = ('Open Time',
                                                                    'Open',
                                                                    'High',
                                                                    'Low',
                                                                    'Close',
                                                                    'Volume',
                                                                    'Close time',
                                                                    'Quote asset volume',
                                                                    'Number of trades',
                                                                    'Taker buy base asset volume',
                                                                    'Taker buy quote asset volume',
                                                                    'Ignore'))

    df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms')
    return df

def percentage(percent, whole):
  return (percent * whole) / 100.0


#You do not need to enter your key / secret to obtain exchange data, they are only needed for transactions in the TradingBot.py class..
api_key = 'ghsXM7BN8z1FmsZhlHqLyHZR804qYXGqv4hjBHjAidRpynXdAaFEHod9H4P8A9EE'
api_secret = 'BTcxCD5hJ5TrjAFIVAFarWRXgItxX9LpHIgWYrVFM9mxq7IFk8DNWPkDu87Mskm5'
client = Client(api_key, api_secret)

symbol = "BTCUSDT"
#candles = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1MINUTE, "28 Nov, 2020", "29 Nov, 2020")
candles = np.array(client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1HOUR, "28 Nov, 2020", "29 Nov, 2020"))
df = binanceDataFrame(candles)


X = df.to_numpy()

# Impute missing
imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
#imp = SimpleImputer(missing_values=np.nan, copy=False, strategy="mean", )
#imp = pp.Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
imp = imp.fit(X)
X = imp.transform(X)
scaler = pp.StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

print(X)


'''
dataset_length = len(candles)

train_percentage = 70
validation_percentage = 10
test_percentage = 100 - train_percentage - validation_percentage
train_length = int(percentage(train_percentage, dataset_length))
validation_length = int(percentage(validation_percentage, dataset_length))
test_length = int(dataset_length - train_length - validation_length)

#Convert the raw exchange data into a more user-friendly form, with some basic resource creation
x = cf.FeatureCreation(candles)
#Create our goals
y = cf.CreateTargets(candles,1)

# removes the main elements of the resources and targets - this is for certain resources that are not compatible with the main
# for example, SMA27 would have 27 entries that would be incompatible / incomplete and would need to be discarded
y = y[94:]
x = x[94:len(candles)-1]

# produce sets, avoiding overlaps!
# data is separated temporarily instead of randomly
# this prevents the model from learning things it wouldn't know - also known as a leak - which can give us false positive models
trny = y[:train_length-1]
trnx = x[:train_length-1]

# The validation set is not used in this initial model, but it should be used if you are using other libraries that support early stop.
valy = y[train_length:train_length+validation_length-1]
valx = x[train_length:train_length+validation_length-1]

tsty = y[train_length+validation_length:]
tstx = x[train_length+validation_length:]

model = GradientBoostingClassifier(n_estimators=20, learning_rate = 0.5, max_features=2, max_depth = 2, random_state = 0) 
#print("MODEL: " + str(model))
model.fit(trnx,trny)

preds = model.predict(tstx)

# Some basic tests to let us know the performance of our model on invisible data - "modern".
# Help with fine-tuning features and model parameters
accuracy = accuracy_score(tsty, preds)
mse = mean_squared_error(tsty, preds)

print("Accuracy = " + str(accuracy))
print("MSE = " + str(mse))

falsePos = 0
falseNeg = 0
truePos = 0
trueNeg = 0
total = len(preds)

for i in range(len(preds)):
    
    if preds[i] == tsty[i] and tsty[i] == 1:
        truePos +=1
        
    elif preds[i] == tsty[i] and tsty[i] == 0:
        trueNeg +=1
        
    elif preds[i] != tsty[i] and tsty[i] == 1:
        falsePos +=1
        
    elif preds[i] != tsty[i] and tsty[i] == 0:
        falseNeg +=1
        
print("False Pos = " + str(falsePos/total))
print("False Neg = " + str(falseNeg/total))
print("True Pos = " + str(truePos/total))
print("True Neg = " + str(trueNeg/total))

# How important are resources - help in selecting and creating resources!
results = pd.DataFrame()
results['names'] = trnx.columns
results['importance'] = model.feature_importances_
print(results.head)

out = binanceDataFrame(candles)
#out2 = binanceDataFrame(model)


# Plot
''''''
PLOT EXAMPLE
#plt.plot(tstx, tsty)
#plt.plot(preds, tstx)
plt.plot(out['Close'])


# Decorate
plt.xlabel('TEMPO')
plt.title('TRAIN VS PREDICTION')
#plt.xlim(1,10)
#plt.ylim(-1.0 , 2.5)
plt.show()
''''''

plt.style.use('ggplot')

fig, ax = plt.subplots()

y_val = tsty
x_val = tstx
y_val_pred = model.predict(x_val)


ax.plot(y_val, y_val_pred, 'b.')
ax.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'k--')
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
ax.set_title('Ridge')
plt.show()


# save our model to the system for use in the bot
dump(model, open("Models/model.mdl", 'wb'))
'''







