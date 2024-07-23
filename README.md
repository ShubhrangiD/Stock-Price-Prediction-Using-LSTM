# Stock Price Prediction Using LSTM

This project focuses on predicting stock prices using historical data from Yahoo Finance. We utilize Long Short-Term Memory (LSTM) networks to forecast future stock prices and visualize the results using Streamlit.

## Table of Contents

1. [Introduction](#introduction)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Data Collection](#data-collection)
5. [Data Preprocessing](#data-preprocessing)
6. [Model Building](#model-building)
7. [Model Evaluation](#model-evaluation)
8. [Saving the Model](#saving-the-model)
9. [Running with Streamlit](#running-with-streamlit)
10. [Conclusion](#conclusion)

## Introduction

Predicting stock prices is a challenging task that can provide significant insights for investors. This project demonstrates how to build a machine learning model using LSTM networks to predict stock prices and visualize the results.

## Requirements

- Python 3.x
- NumPy
- Pandas
- Matplotlib
- yfinance
- scikit-learn
- Keras
- TensorFlow
- Streamlit

## Installation

Install the required libraries using pip:

```bash
pip install numpy pandas matplotlib yfinance scikit-learn keras tensorflow streamlit
```

## Data Collection

Fetch historical stock data for Google from January 1, 2012, to December 21, 2023, using the `yfinance` library.

## Data Preprocessing

- Calculate the 100-day and 200-day moving averages.
- Prepare the training and test datasets.
- Scale the data using MinMaxScaler.

## Model Building

- Build and compile the LSTM model with multiple LSTM layers and dropout layers.
- Train the model on the prepared training data.

## Model Evaluation

- Evaluate the model using test data.
- Visualize the predicted and actual stock prices to assess the model's performance.

## Saving the Model

Save the trained model for future use:

```python
model.save('Stock_Predictions_Model.keras')
```

## Running with Streamlit

To run the project using Streamlit, create a file named `app.py` and use the following command:

```bash
streamlit run app.py
```

The Streamlit application will display the stock data, moving averages, and predictions in an interactive web interface.

## Conclusion

This project demonstrates how to use LSTM networks to predict stock prices using historical data and visualize the results with Streamlit. By following this guide, you can apply similar techniques to predict prices for other stocks or financial instruments.
