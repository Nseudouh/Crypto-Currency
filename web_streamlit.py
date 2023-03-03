import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import \
MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import yfinance as yf
#%matplotlib inline
from matplotlib import pyplot as plt
import statsmodels.api as smt
import seaborn as sns
import yfinance as yf
from sklearn.model_selection import train_test_split
import datetime as dt
from plotly import graph_objects as go

st.title('SOL INTELLIGENCE')
st.title('Cryptocurrency Prediction price')
cryp_download = ["BTC", "ETH", "LDO","WBTC", "AXS", "BNB", "MKR", "LTC", "XMR", "LUNA1", "ADA", "AVAX", "THETA", "TRX", "FTM", "KSM",
                 "MATIC", "DOGE", "LTC", "HNT", "KSM"]

choice = st.selectbox("Select your preferred coin", cryp_download)

def retrieve(abb):
    start = dt.datetime(2019, 1, 1)
    end = dt.datetime.now()
    df = yf.download(f'{abb}-USD', start=start, end=end, interval='1d')  #abb, start=start, end=end, interval='1d')
    df.reset_index(inplace=True)
    return df


download = retrieve(choice)

st.write(download.head())

def dataframe():
    new_data = pd.DataFrame(download)
    return new_data

def close_graph():
    figure = go.Figure()
    figure.add_trace(go.Scatter(x= dataframe()['Date'], y= dataframe()['Close']))
    figure.layout.update(title_text=f'Closing graph for {choice}', xaxis_rangeslider_visible=True)
    st.plotly_chart(figure)

close_graph()

days = (1,5,30,60,90)
future_days = st.selectbox("Please select the number of days", days)

def split_train_predict():
    x = np.array(dataframe().drop(['Close'], 1))

    y = np.array(dataframe()['Close'])

    # Split dataset into training set and test se
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,
                                                        random_state=0)  # 70% training and 30% test
    timestep = 30
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    btc_x_test = []
    btc_y_test = []
    btc_x_data = dataframe().drop(['Close'], 1).to_numpy()
    btc_x_data = btc_x_data.reshape((len(btc_x_data), 4))
    btc_y_data = dataframe()['Close']
    for x in range(future_days, len(x_train)):
        X_train.append(x_train[x - future_days:x, 3])
        Y_train.append(y_train[x])
    for x in range(future_days, len(x_test)):
        X_test.append(x_test[x - future_days:x, 3])
        Y_test.append(y_test[x])
    for x in range(future_days, len(btc_x_data)):
        btc_x_test.append(btc_x_data[x - future_days:x, 3])
        btc_y_test.append(btc_y_data[x])
    X_train, Y_train, X_test, Y_test, btc_x_test, btc_y_test = np.array(X_train), np.array(Y_train).reshape(-1,
                                                                                                            1), np.array(
        X_test), np.array(Y_test).reshape(-1, 1), np.array(btc_x_test), np.array(btc_y_test).reshape(-1, 1)

    scaler = MinMaxScaler()


    # scaling BTC data
    btc_scaled_x, btc_scaled_y = np.array(scaler.fit_transform(btc_x_test)), np.array(scaler.fit_transform(btc_y_test))

    rfr_model = RandomForestRegressor(n_estimators=300, max_depth=15)
    rfr_model.fit(btc_scaled_x, btc_scaled_y)
    rfr_prediction = rfr_model.predict(btc_scaled_x)
    # checking for model accuracy
    rmse = np.sqrt(mean_squared_error(btc_scaled_y, rfr_prediction))
    mae = mean_absolute_error(btc_scaled_y, rfr_prediction)
    rfr_prediction = rfr_model.predict(btc_scaled_x)
    return rfr_prediction

