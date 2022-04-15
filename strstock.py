import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
import numpy as np

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
end_date = st.sidebar.date_input("End Date" , datetime.date(2022,4,15))

ticker_list = pd.read_csv('AAPL','GOOGL')
tickerSymbol = st.sidebar.selectbox('Stock Ticker',ticker_list)
