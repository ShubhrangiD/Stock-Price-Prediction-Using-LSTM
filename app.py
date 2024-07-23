# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential  # type: ignore
from keras.layers import Dense, Dropout, LSTM  # type: ignore
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Define the date range
start = "2009-01-01"
end = pd.Timestamp.today().strftime('%Y-%m-%d')

st.title('Stock Closing Price Prediction')

# Reading data
user_input = st.text_input('Enter Stock Ticker', 'GOOG')
df = yf.download(user_input, start=start, end=end)

st.subheader('Dated from 1st Jan, 2009 to Today')
st.write(df.describe())

# First plot
st.subheader('Closing Price Vs Time Chart')
fig1 = plt.figure(figsize=(12, 6))
plt.plot(df['Close'])
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Closing Price Over Time')
st.pyplot(fig1)

# Moving averages
ma100 = df['Close'].rolling(100).mean()
ma200 = df['Close'].rolling(200).mean()

# Second plot
st.subheader('Closing Price Vs Time Chart with 100 days Moving Average')
fig2 = plt.figure(figsize=(12, 6))
plt.plot(df['Close'], 'r', label="Per Day Closing")
plt.plot(ma100, 'g', label="Moving Average 100")
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Closing Price and 100 Days Moving Average')
plt.legend()
st.pyplot(fig2)

# Third plot
st.subheader('Closing Price Vs Time Chart with 100 days and 200 days Moving Average')
fig3 = plt.figure(figsize=(12, 6))
plt.plot(df['Close'], 'r', label="Per Day Closing")
plt.plot(ma100, 'g', label="Moving Average 100")
plt.plot(ma200, 'b', label="Moving Average 200")
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Closing Price with Moving Averages')
plt.legend()
st.pyplot(fig3)

# Data preparation for training
train_df = pd.DataFrame(df['Close'][0: int(len(df)*0.85)])
test_df = pd.DataFrame(df['Close'][int(len(df)*0.85):int(len(df))])
scaler = MinMaxScaler(feature_range=(0, 1))
train_df_arr = scaler.fit_transform(train_df)

x_train = []
y_train = []
for i in range(100, train_df_arr.shape[0]):
    x_train.append(train_df_arr[i-100: i])    
    y_train.append(train_df_arr[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)

# Load the model
model = Sequential()

model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units=60, activation='relu', return_sequences=True))
model.add(Dropout(0.3))

model.add(LSTM(units=80, activation='relu', return_sequences=True))
model.add(Dropout(0.4))

model.add(LSTM(units=120, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units=1))

# Train the model
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs = 10)

# Prepare the data for prediction
past_100_days = train_df.tail(100)
final_df = pd.concat([past_100_days, test_df], ignore_index=True)

input_data = scaler.transform(final_df)

x_test = []
y_test = []
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])    
    y_test.append(input_data[i, 0])
x_test, y_test = np.array(x_test), np.array(y_test)

# Model prediction
y_pred = model.predict(x_test)
scale = scaler.scale_
scale_factor = 1 / scale[0]
y_pred = y_pred * scale_factor
y_test = y_test * scale_factor

# Final plot
st.subheader('Predicted Vs Original')
fig4 = plt.figure(figsize=(12, 6))
plt.plot(y_test, 'g', label="Original Price")
plt.plot(y_pred, 'r', label="Predicted Price")
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Predicted vs Original Prices')
plt.legend()
st.pyplot(fig4)

# Fetch real-time data
def fetch_data(ticker):
    end_date = pd.Timestamp.today().strftime('%Y-%m-%d')
    start_date = (pd.Timestamp.today() - pd.DateOffset(days=200)).strftime('%Y-%m-%d')
    return yf.download(ticker, start=start_date, end=end_date)

real_time_data = fetch_data(user_input)

st.subheader(f'Real-Time Stock Data for {user_input}')
st.write(real_time_data.tail(10))  # Display the last 10 rows of real-time data

# Prepare real-time data for prediction
real_time_close = real_time_data['Close'].values.reshape(-1, 1)
real_time_scaled = scaler.transform(real_time_close)

x_real_time = []
for i in range(100, len(real_time_scaled)):
    x_real_time.append(real_time_scaled[i-100:i])
x_real_time = np.array(x_real_time)

# Predict using the model
real_time_pred = model.predict(x_real_time)
real_time_pred = real_time_pred * scale_factor

# Plot real-time prediction
st.subheader(f'Real-Time Predicted Prices for {user_input}')
fig4 = plt.figure(figsize=(12, 6))
plt.plot(real_time_close[100:], 'g', label="Real Time Close Price")
plt.plot(real_time_pred, 'r', label="Predicted Price")
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Real-Time Predicted vs Actual Prices')
plt.legend()
st.pyplot(fig4)