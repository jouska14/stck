import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
import numpy as np
from plotly import graph_objs as go


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
start_date = st.sidebar.date_input("Start Date", datetime.date(2019,1,1))
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
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='Stock-Open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Stock-Close'))
    fig.layouts.update(title_text="Time Series Data")
    st.plotly_chart(fig)
  
  
plot_raw_data()
