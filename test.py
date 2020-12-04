# How to Design Intraday Algo-Trading Model for Cryptocurrencies 
# using Bitcoin-based Signals?
# 
# (c) 2020 QuantAtRisk.com, by Pawel Lachowicz
 
import ccrypto as cc
 
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import plotly.graph_objects as go  # good for candlestick charts
import pickle  # for storing Python's dictionaries in a file
 
import warnings
warnings.filterwarnings("ignore")
 
# matplotlib color codes
blue, orange, red, green = '#1f77b4', '#ff7f0e', '#d62728', '#2ca02c'
grey8 = (.8,.8,.8)

# coin selection
coins = ['BTC', 'ETH', 'XRP', 'BCH', 'LTC', 'EOS', 'XTZ', 'LINK', 
         'XLM', 'DASH', 'ETC', 'ATOM']

database = {}
for coin in coins:
    print(coin + '...', end=" ")
    try:
        database[coin] = cc.getCryptoSeries(coin, freq='m', ohlc=True, exch='Coinbase')
        print('downloaded')
    except:
        print('unsuccessful')
    
# save dictionoary with time-series in a file (binary)
with open('timeseries_20200420_20200427.db', 'wb') as handle:
    pickle.dump(database, handle)

# load time-series from database
with open('timeseries_20200420_20200427.db', 'rb') as handle:
    ts = pickle.load(handle)
    
print(ts.keys())  # get the keys of available time-series

dict_keys(['BTC', 'ETH', 'XRP', 'BCH', 'LTC', 'EOS', 'XTZ', 'LINK', 'XLM', 'DASH', 
           'ETC', 'ATOM'])

cc.displayS([ts['BTC'].head(), ts['ETH'].head()], ['Bitcoin (start)','Ethereum (start)'])
cc.displayS([ts['BTC'].tail(), ts['ETH'].tail()], ['Bitcoin (end)','Ethereum (end)'])

# candlestick chart for OHLC time-series available as pandas' DataFrames
# employing plotly
 
# color codes for plotly
whiteP, blackP, redP, greyP = '#FFFFFF', '#000000', '#FF4136', 'rgb(150,150,150)'
 
fig = go.Figure(data=go.Candlestick(x     = ts['BTC'].index, 
                                    open  = ts['BTC'].iloc[:,0], 
                                    high  = ts['BTC'].iloc[:,1],
                                    low   = ts['BTC'].iloc[:,2],
                                    close = ts['BTC'].iloc[:,3],)
               )
fig.update(layout_xaxis_rangeslider_visible=False)
fig.update_layout(plot_bgcolor=whiteP, width=500)
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=greyP)
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=greyP)
fig.update_yaxes(title_text='BTC/USD')
 
# update line and fill colors
cs = fig.data[0]
cs.increasing.fillcolor, cs.increasing.line.color = blackP, blackP
cs.decreasing.fillcolor, cs.decreasing.line.color = redP, redP
 
fig.show()

btc = ts['BTC']  # assign BTC data to a new temporary variable
 
btc = btc[['BTCUSD_O', 'BTCUSD_C']]  # limit to Open and Close series
 
# add a new column with Close price 1 min ago
btc['BTCUSD_C_LAG1'] = btc['BTCUSD_C'].shift(1)
 
# define a supporting function
def rr(z):
    '''Calculates rate of return [percent].
       Works with two DataFrame's columns as an input.
    '''
    x, y = z[0], z[1]
    return 100*(x/y-1)
 
 
# calculate rate of return between:
btc['rate_of_reutrn'] = btc[['BTCUSD_C', 'BTCUSD_C_LAG1']].apply(rr, axis=1)
 
display(btc)
 
# get rid of NaN rows
btc = btc.dropna()

# select a threshold for triggers
thr = 0.5  # 1-min rate of return greater than 'thr' percent
 
tmp = btc[btc.rate_of_reutrn > thr]
 
fig, ax = plt.subplots(1,1,figsize=(15,5))
ax.plot((btc.BTCUSD_C), color=grey8)
ax.plot([tmp.index, tmp.index], [tmp.BTCUSD_O, tmp.BTCUSD_C], color=red)
ax.grid()
ax.legend(['BTCUSD Close Price', 'Triggers'])
plt.gcf().autofmt_xdate()
myFmt = mdates.DateFormatter('%Y-%m-%d %H:%M')
plt.gca().xaxis.set_major_formatter(myFmt)

print(tmp.index)


# time index when a new position will be opened
ind_buy = tmp.index + pd.Timedelta(minutes = 1)
print(ind_buy)

def check_stoploss(z, thr1=-0.15, thr2=-0.15):
    p1, p2 = z
    if p1 < thr1 or p2 < thr2:
        return False  # close position
    else:
        return True  # hold position open


backtested_coins = ['BTC']
 
results = {}
 
for coin in backtested_coins:
 
    # read OHLC price time-series
    df = ts[coin]
 
    tradePnLs = list()
 
    for ib in range(len(ind_buy)):
        i = ind_buy[ib]
        try:
            op = df.loc[i][0]
 
            # Trade No. 'ib' DataFrame
            tmp = df[df.index >= i]
            tmp['open_price'] = op  # trade's open price
            tmp['current_price'] = df[coin + 'USD_C']
            tmp['pnl'] = tmp.current_price / op - 1
 
            fi = True
            out1 = list()
            out2 = list()
            for j in range(tmp.shape[0]):
                if fi:
                    maxPnL = tmp.pnl[j]
                    maxClose = tmp.iloc[j, 3]
                    fi = False
                else:
                    if tmp.pnl[j] > maxPnL:
                        maxPnL = tmp.pnl[j]
                        maxClose = tmp.iloc[j, 3]
                out1.append(maxPnL)
                out2.append(maxClose)  # close price
 
            tmp['maxPnL'] = out1
            tmp['maxClose'] = out2
            tmp['drawdown'] = tmp.current_price / tmp.maxClose - 1
            tmp['hold'] = tmp[['pnl', 'drawdown']].apply(check_stoploss, axis=1)
 
            # execute selling if detected
            sell_executed = True
            try:
                sell_df = tmp[tmp.hold == 0]
                sell_time, close_price = sell_df.index[0], sell_df.current_price[0]
                tmpT = tmp[tmp.index <= sell_time]
            except:
                sell_executed = False
 
            #display(tmp.iloc[:,:].head(10))
 
            plt.figure(figsize=(15,4))
            plt.grid()
            plt.plot(tmp.pnl, color=grey8, label = "Rolling trade's PnL (open trade)")
            if sell_executed:
                plt.plot(tmpT.pnl, color=blue, label = "Rolling trade's PnL (closed)")
                plt.title("Trade's final PnL = %.2f%%" % (100*tmpT.iloc[-1,6]))
                tradePnLs.append(tmpT.iloc[-1,6])
            else:
                plt.title("Current trade's PnL = %.2f%%" % (100*tmp.iloc[-1,6]))
                tradePnLs.append(tmp.iloc[-1,6])
            plt.plot(tmp.maxPnL, color=orange, label = "Rolling maximal trade's PnL")
            plt.plot(tmp.index, np.zeros(len(tmp.index)), '--k')
            plt.suptitle('Trade No. %g opened %s @ %.2f USD' % (ib+1, i, df.loc[i][0]))
            plt.legend()
            locs, labels = plt.xticks()
            plt.xticks(locs, [len(list(labels))*""])
            plt.show()
 
            plt.figure(figsize=(14.85,1.5))
            plt.grid()
            plt.plot(tmp.drawdown, color=red, label = "Rolling trade's drawdown")
            plt.plot(tmp.index, np.zeros(len(tmp.index)), '--k')
            plt.gcf().autofmt_xdate()
            myFmt = mdates.DateFormatter('%Y-%m-%d %H:%M')
            plt.gca().xaxis.set_major_formatter(myFmt)
            plt.legend()
            plt.show()
 
            print("\n\n")
        except:
            pass

c = 1000  # initial investment (fixed; per each trade)
tradePnLs = np.array(tradePnLs)
n_trades = len(tradePnLs)
res = pd.DataFrame(tradePnLs, columns=['Trade_PnL'])
res['Investment_USD'] = c
res['Trade_ROI_USD'] = np.round(c * (tradePnLs + 1),2)
res.index = np.arange(1,n_trades+1)
res.index.name = 'Trade_No'
ROI = res.Trade_ROI_USD.sum() - (n_trades * c)
ROI_pct = 100 * (res.Trade_ROI_USD.sum() / (n_trades * c) - 1)
tot_pnl = res.Trade_ROI_USD.sum()
res.loc[res.shape[0]+1] = ['', np.round(n_trades * c,2), '']
res.rename(index = {res.index[-1] : "Total Investment (USD)"}, inplace=True)
res.loc[res.shape[0]+1] = ['', '', np.round(tot_pnl,2)]
res.rename(index = {res.index[-1] : "Total PnL (USD)"}, inplace=True)
res.loc[res.shape[0]+1] = ['', '', np.round(ROI,2)]
res.rename(index = {res.index[-1] : "Total ROI (USD)"}, inplace=True)
res.loc[res.shape[0]+1] = ['', '', np.round(ROI_pct,2)]
res.rename(index = {res.index[-1] : "Total ROI (%)"}, inplace=True)

results[coin] = res

display(results['BTC'])

backtested_coins = ['BTC', 'ETH', 'XTZ', 'DASH', 'LINK']

cc.displayS([results['ETH'], results['XTZ']], ['Trading ETH/USD', 'Trading XTZ/USD'])
cc.displayS([results['DASH'], results['LINK']],['Trading DASH/USD', 'Trading LINK/USD'])