import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
import numpy as np
from plotly import graph_objs as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import LSTM


st.title('Stock Predictor')

st.write('Shown are the stock price data for query companies!')
st.markdown('''
**Credits**
- App built by Nehal, Sara, Alok
- Built in `Python` using `streamlit`,`yfinance`, `pandas` and `datetime`
''')
st.write('---')

# Sidebar
st.sidebar.subheader('Choose Your Query Parameter ')
start_date = st.sidebar.date_input("Start Date", datetime.date(1999,1,1))
end_date = datetime.date.today().strftime("%Y-%m-%d")

stocks = ('AAPL','GOOGL', 'MSFT')
tickerSymbol = st.sidebar.selectbox('Select',stocks)

@st.cache
def load_data(ticker):
  data = yf.download(ticker,start_date,end_date)
  data.reset_index(inplace=True)
  return data


data_load_state = st.text("Load data...")
data = load_data(tickerSymbol)
data_load_state.text("Loading data.... Done!")

st.subheader('Raw Data')
st.write(data)

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Stock-Close'))
    fig.update_layout(title="This is the Trend in the Raw Data")
    st.plotly_chart(fig)
  
  
plot_raw_data()
data_close = data['Close']
scaler = MinMaxScaler(feature_range = (0,1))
data_close = scaler.fit_transform(np.array(data_close).reshape(-1,1))

#Split the data into train and test split
training_size = int(len(data_close)*0.75)
test_size = len(data_close)-training_size
train_data, test_data = data_close[0:training_size,:], data_close[training_size:len(data_close)]

def create_dataset(dataset, time_step = 1):
    dataX, dataY = [], []
    for i in range(len(ataset) - time_step-1):
        a = dataset[i:(i+time_step),0]
        dataX.append(a)
        dataY.append(dataset[i+time_step, 0]
        return np.array(dataX), np.array(dataY)
                     
time_step = 100
x_train, y_train = create_dataset(train_data, time_step)
x_test, y_test = create_dataset(test_data, time_step)
                     
#reshape the input to be [sample, time steps, features] which is the requirement of LSTM
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
st.write(x_train)
#Create the LSTM model

                     
                     
                     
                     
    
    
    
    
    
