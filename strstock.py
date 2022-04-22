import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
import altair as alt

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
end_date = datetime.date.today()

stocks = ('AAPL','GOOGL', 'MSFT')
tickerSymbol = st.sidebar.selectbox('Select',stocks)
tickerData = yf.Ticker(tickerSymbol) #get ticker data
tickerDf = tickerData.history(period = "1mo", start = start_date, end = end_date)#getting historical price

#ticker info
string_logo = '<img src=%s>' % tickerData.info['logo_url']
st.markdown(string_logo, unsafe_allow_html=True)

string_name = tickerData.info['longName']
st.header('**%s**' % string_name)

string_summary = tickerData.info['longBusinessSummary']
st.info(string_summary)

df = pd.DataFrame(tickerDf)

chck = st.checkbox('Show dataframe')
if chck:
    st.write(df)
    
st.header('**Trends in Historical Data**')
st.line_chart(tickerDf)

'''# Shown are the Stock *closing price* #'''
st.line_chart(tickerDf.Close)

