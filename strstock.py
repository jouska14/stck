import streamlit as st
import yfinance as yf
import altair as alt
import pandas as pd
import datetime
import numpy as np
import plotly.express as px
from plotly import graph_objs as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import LSTM
import math

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
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'],  name='Stock Close'))
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

@st.cache
def create_dataset(dataset, time_step = 1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i+time_step),0]
        dataX.append(a)
        dataY.append(dataset[i+time_step, 0])
    return np.array(dataX), np.array(dataY)
                     
time_step = 100
x_train, y_train = create_dataset(train_data, time_step)
x_test, y_test = create_dataset(test_data, time_step)
                     
#reshape the input to be [sample, time steps, features] which is the requirement of LSTM
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

                     
#Create the LSTM model
model = Sequential() 
model.add(LSTM(50 ,return_sequences = True, input_shape = (100,1)))
model.add(LSTM(50, return_sequences= True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss = 'mean_squared_error', optimizer = 'adam')
model.summary()   

model.fit(x_train,y_train, validation_data=(x_test,y_test), epochs=4, batch_size=64, verbose=1)

#Lets predict and check performance metrics
train_predict = model.predict(x_train)
test_predict = model.predict(x_test)

#Transform back to original form
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

#Calculate RMSE performance metrics
math.sqrt(mean_squared_error(y_train, train_predict))
#Plotting

#Shift train prediction for plotting
look_back = 100
trainPredictPlot = np.empty_like(data_close)
trainPredictPlot[:,:] = np.nan
trainPredictPlot[look_back:len(train_predict) + look_back, :] = train_predict

#Shift test prediction for plotting
testPredictPlot = np.empty_like(data_close)
testPredictPlot[:,:] = np.nan
testPredictPlot[len(train_predict) + (look_back * 2)+1:len(data_close) - 1, :] = test_predict
itdc = pd.DataFrame(scaler.inverse_transform(data_close))
st.write(itdc)
#Plot baseline and predictions
tpp = pd.DataFrame(trainPredictPlot)
st.write(tpp)

tepp = pd.DataFrame(testPredictPlot)
st.write('This is how we split the Data')
fig = px.line(data_close ,x=data['Date'] , y=itdc[0], labels={'x':'Year' , 'y':'Closing price'})
st.write(fig)
fig1 = px.line(data_close ,x=data['Date'] , y=tpp[0], labels= {'x':'Year' , 'y':'Closing price'})
st.write(fig1)
fig2 = px.line(data_close ,x=data['Date'] , y=tepp[0], labels= {'x':'Year', 'y':'Closing price'})
st.write(fig2)


x_input = test_data[1367:].reshape(-1,1)

temp_input=list(x_input)
temp_input=temp_input[0].tolist()

lst_output=[]
n_steps=100
i=0
while(i<60):
    
    if(len(temp_input)>100):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        st.write("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        st.write("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        st.write(yhat[0])
        temp_input.extend(yhat[0].tolist())
        st.write(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
    
day_new=np.arange(1,1407)
day_pred=np.arange(1407,1467)

dp1 = scaler.inverse_transform(data_close[5768:])
dp2 = scaler.inverse_transform(lst_output)

fig = go.Figure()
fig.add_trace(go.Scatter(x= 'day_new', y='dp1', mode='lines'))
fig.add_trace(go.Scatter(a = 'day_pred', y='dp2' ,mode = 'lines'))
fig.show()
