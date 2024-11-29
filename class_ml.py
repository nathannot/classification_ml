import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import plotly.graph_objects as go
import datetime

st.header('ML Classification For Next Day Gain/Loss')
# Stock selection
stocks = st.selectbox('Select from the following stocks', ('Apple', 'Google', 'Tesla', 'Meta', 'Amazon', 'Microsoft'))
code = {'Apple': 'AAPL', 'Google': 'GOOG', 'Tesla': 'TSLA', 'Meta': 'META', 'Amazon': 'amzn', 'Microsoft': 'MSFT'}
ticker = code[stocks]

# Date selection
st.write('Select date range for historical data')
start = st.date_input('Start date', datetime.date(2020, 1, 1), max_value = datetime.date(2022,11,28))
end = st.date_input('End date', datetime(2024,11,1), min_value = start+datetime.timedelta(days=730))
st.write('Using a larger time frame should produce better results')
def lag_function(x, p):
        x = x.copy()
        for i in range(1,p+1):
            x[f'lag_{i}'] = x['Adj Close'].shift(i)
        x = x.dropna()
        return x
da = yf.download(ticker, start,end)[['Adj Close']]
if da.empty:
    st.warning("No data available for the selected date range. Please adjust the date range.")
    st.stop()
df = lag_function(da, 20)
def bbands(x, window):
    x = x.copy()
    
    rolling_ma = x['Adj Close'].rolling(window).mean()
    rolling_std = x['Adj Close'].rolling(window).std()
    x['bb_upper'] = rolling_ma + 2*rolling_std
    x['bb_lower'] = rolling_ma - 2*rolling_std
    #x[f'ma_{window}'] = rolling_ma
    return x
        
def get_rsi(x, target):
    x = x.copy()
    dP = x[target].diff(1)
    gain = dP.where(dP>0,0)
    loss = -dP.where(dP<0,0)
    avg_gain = gain.rolling(14, min_periods=1).mean()
    avg_loss = loss.rolling(14, min_periods=1).mean()
    smooth_gain = np.zeros(len(avg_gain))
    smooth_loss = np.zeros(len(avg_gain))
    n = len(avg_gain)
    for i in range(1,n):
        smooth_gain[i] = (avg_gain.iloc[i-1]*(n-1)+gain.iloc[i])/n
        smooth_loss[i] = (avg_loss.iloc[i-1]*(n-1)+loss.iloc[i])/n
    RS = smooth_gain/(smooth_loss+1e-8)
    RSI = 100-100/(1+RS)
    x['RSI'] = RSI
    return x

df = bbands(df, 21)
data = df.diff(1).dropna()

Y = data.iloc[:,0]
y = np.where(Y>0,1,0)
X = data.iloc[:,1:].values
train = int(0.8*len(y))
X1 = np.column_stack([np.ones(len(X)),X])
Xtrain = X1[:train]
Xtest = X1[train:]
ytrain = y[:train]
ytest = y[train:]


def loss(b, Xtrain, ytrain):
    power = np.clip(Xtrain.dot(b), -709, 709)    
    p = 1/(1+np.exp(-power))
    loss = ytrain@np.log(p+1e-8)+(1-ytrain)@np.log(1-p+1e-8)
    return -loss

init = [0.1]*Xtrain.shape[1]
result = minimize(loss, init, args=(Xtrain, ytrain))
beta = result.x

power = Xtest.dot(beta)
power_clipped = np.clip(power, -709, 709)

p = 1/(1+np.exp(-power_clipped))
ypred = np.where(p>=0.5,1,0)
rdf = pd.DataFrame({'ytest':ytest,
              'ypred':ypred})
diff = ytest-ypred
rdf['diff']=diff
acc = (np.sum(np.where(diff==0,1,0))/len(ytest)).round(4)
TP = np.sum(np.where((diff==0) & (ytest == 1) ,1,0))
TN = np.sum(np.where((diff==0) & (ytest == 0) ,1,0))
FP = np.sum(np.where((diff==-1) ,1,0))
FN = np.sum(np.where( diff == 1,1,0))

recall1 = TP/(TP+FN)
precision1 = TP/(TP+FP)
F11 = 2/(1/recall1+1/precision1)
supp1 = TP+FN
recall0 = TN/(TP+FN)
precision0 = TN/(TN+FN)
F10 = 2/(1/recall0+1/precision0)
supp0 = TN+FP

tot = len(ytest)
recall_avg = supp1*recall1/tot + supp0*recall0/tot
precision_avg = supp1*precision1/tot + supp0*precision0/tot
F1_avg = supp1*F11/tot + supp0*F10/tot
print(acc,recall_avg, precision_avg, F1_avg)

next = np.hstack([1,data.iloc[-1,:][:-1].values])
power = next.dot(beta)
power_clipped = np.clip(power, -709, 709)
p = 1/(1+np.exp(-power_clipped))
fut = np.where(p>0.5,1,0).reshape(-1)
pred = np.where(fut==1, 'UP', 'DOWN').reshape(-1)[0]

columns = ['recall','precision','F1','Support']
index = ['Down','Up','weighted average']
recalls = [recall0,recall1,recall_avg]
pres = [precision0,precision1,precision_avg]
F1s = [F10,F11,F1_avg]
supp = [supp0,supp1,tot]
summ = pd.DataFrame({'recall':recalls, 'precision':pres, 'F1':F1s,
              'Support':supp}, index=index)
st.write('Classification Model Performance Summary')
st.write(summ)
st.write(f'Model accuracy is {acc}')
st.write(f'The prediction of next day is {pred}')
d = yf.download(ticker, end-datetime.timedelta(days=3),end+datetime.timedelta(days=4))[['Adj Close']]
st.write('The table below shows last price in model and next day prices so we can see if the model prediction is right')
st.write(d)
st.write('The last date the model uses is the last business day before end date.')
st.write('Find the next business date to see actual price of future date for the model')
fig = go.Figure()
fig.add_trace(go.Scatter(x=da.index[train:],y=da['Adj Close'].iloc[train:]))
fig.update_layout(hovermode='x', title = f'Daily price chart of {ticker}')

st.plotly_chart(fig)
st.write('Combine the prediction with the graph to visualise price direction')
