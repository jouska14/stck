import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
import numpy as np

st.title('Stock Predictor')

st.markdown('''
          # Stock Price App
Shown are the stock price data for query companies!
**Credits**
- App built by Nehal, Sara, Alok
- Built in `Python` using `streamlit`,`yfinance`, `pandas` and `datetime`
''')
st.write('---')


