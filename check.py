# -*- coding: utf-8 -*-
import datetime
from binance.client import Client
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import sys

#client = Client('API_KEY', 'SECRET_KEY')
client = Client('', '')

crypto = client.get_historical_klines(symbol=sys.argv[1], interval=Client.KLINE_INTERVAL_30MINUTE, start_str="15 Nov, 2020")
ds_length = len(crypto)
ds_perc = 80
trn_length = int(ds_length/100*ds_perc)

crypto = pd.DataFrame(crypto, columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'])


crypto['Open time'] = pd.to_datetime(crypto['Open time'], unit='ms')

crypto.set_index('Open time', inplace=True)

crypto['Close']=crypto['Close'].astype(float)

data = crypto.iloc[:,3:4].astype(float).values

scaler= MinMaxScaler()
data= scaler.fit_transform(data)

training_set = data[:trn_length-1]
test_set = data[trn_length:]

X_train = training_set[0:len(training_set)-1]
y_train = training_set[1:len(training_set)]

X_test = test_set[0:len(test_set)-1]
y_test = test_set[1:len(test_set)]

X_train = np.reshape(X_train, (len(X_train), 1, X_train.shape[1]))
X_test = np.reshape(X_test, (len(X_test), 1, X_test.shape[1]))


model = Sequential()
model.add(LSTM(256, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(256))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, epochs=50, batch_size=16, shuffle=False)

model.save("models/"+sys.argv[1]+'_model.h5')


predicted_price = model.predict(X_test)
predicted_price = scaler.inverse_transform(predicted_price)
real_price = scaler.inverse_transform(y_test)

preds = predicted_price
#accuracy = accuracy_score(y_test, predicted_price)
mse = mean_squared_error(y_test, predicted_price)

#print("Accuracy = " + str(accuracy))
print("MSE = " + str(mse))

falsePos = 0
falseNeg = 0
truePos = 0
trueNeg = 0
total = len(preds)

for i in range(len(preds)):
    
    if preds[i] == y_test[i] and y_test[i] == 1:
        truePos +=1
        
    elif preds[i] == y_test[i] and y_test[i] == 0:
        trueNeg +=1
        
    elif preds[i] != y_test[i] and y_test[i] == 1:
        falsePos +=1
        
    elif preds[i] != y_test[i] and y_test[i] == 0:
        falseNeg +=1
        
print("False Pos = " + str(falsePos/total))
print("False Neg = " + str(falseNeg/total))
print("True Pos = " + str(truePos/total))
print("True Neg = " + str(trueNeg/total))


plt.figure(figsize=(10,4))
red_patch = mpatches.Patch(color='red', label='Predicted Price of '+sys.argv[1])
blue_patch = mpatches.Patch(color='blue', label='Real Price of '+sys.argv[1])
plt.legend(handles=[blue_patch, red_patch])
plt.plot(predicted_price, color='red', label='Predicted Price of '+sys.argv[1])
plt.plot(real_price, color='blue', label='Real Price of '+sys.argv[1])
plt.title('Predicted vs. Real Price of '+sys.argv[1])
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()