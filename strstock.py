import streamlit as st
import yfinance as yf

def get_ticker(name):
    company = yf.Ticker(name)
    return company


st.title("Build and Deploy Stock Market App Using Streamlit")
st.header("A Basic Data Science Web Application")
st.sidebar.header("Geeksforgeeks \n TrueGeeks")

company1 = get_ticker("AAPL")
company2 = get_ticker("GOOGL")

apple = yf.download("AAPL", start="2021-10-01", end="2021-10-01")
google = yf.download("GOOGL", start="2021-10-01", end="2021-10-01")

data1 = company1.history(period="3mo")
data2  = company2.history(period="3mo")

st.write(

)

st.write(company1.info['longBusinessSummary'])
st.write(apple)

st.line_chart(data1.values)

st.write(

)
st.write(company2.info['longBusinessSummary'], "\n", google)
st.line_chart(data2.values)