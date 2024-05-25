# Agile Price Prediction

This repository contains a script to predict electricity prices using various features such as weather data, sunlight data, and UK bank holidays. The script trains a Random Forest Regressor model and evaluates its performance based on Mean Absolute Error (MAE). The prediction results are also plotted and saved as an image.

## Files

- `predictor.py`: The main script that loads data, preprocesses it, trains the model, and makes predictions.
- `agilef2023.csv`: The agile electricity prices dataset.
- `weather.csv`: The weather data dataset.
- `daily sunlight.csv`: The daily sunlight duration dataset.
- `uk_bank_holidays.csv`: The UK bank holidays dataset.

## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- joblib
- matplotlib

You can install the required packages using:


pip install pandas numpy scikit-learn joblib matplotlib

## Usage

Load Data: The script reads data from the provided CSV files.
Preprocess Data: The script preprocesses and merges the datasets, adding relevant features like day_of_week and bank_hol.
Train Model: The script trains a Random Forest Regressor model.
Evaluate Model: The script evaluates the model's performance using Mean Absolute Error (MAE).
Predict Prices: The script predicts electricity prices for the next day and plots the results.

## Changing Input Variables

You can change the input variables such as wind speeds, daylight duration, and sunshine duration percentage directly in the script. These variables are used in the predict_tomorrow function.

To change these variables, locate the following section in predictor.py:

wind_speeds = [7, 7, 7, 7, 7, 8, 9, 9, 10, 11, 12, 12, 12, 12, 13, 13, 13, 12, 12, 11, 10, 9, 9, 9]
daylight_duration = timedelta(hours=15, minutes=49, seconds=53).total_seconds()
sunshine_duration_percent = 0.55	

Wind speeds can be obtained from https://www.timeanddate.com/weather/uk/norwich/hourly
Future improvements, maybe get daylight hours, weather from online source
