import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Load the datasets
def load_data():
    agile_prices = pd.read_csv('sample_data/agilef2023.csv')
    weather_data = pd.read_csv('sample_data/weather.csv')
    sunlight_data = pd.read_csv('sample_data/daily sunlight.csv')
    bank_holidays = pd.read_csv('sample_data/uk_bank_holidays.csv')
    return agile_prices, weather_data, sunlight_data, bank_holidays

# Preprocess and merge the data with day_of_week and bank_hol features
def preprocess_and_merge(agile_prices, weather_data, sunlight_data, bank_holidays):
    agile_prices['timestamp'] = pd.to_datetime(agile_prices['timestamp']).dt.tz_convert(None)
    weather_data['time'] = pd.to_datetime(weather_data['time']).dt.tz_localize('UTC').dt.tz_convert(None)
    sunlight_data['date'] = pd.to_datetime(sunlight_data['date'], format='%d/%m/%Y')
    bank_holidays['timestamp'] = pd.to_datetime(bank_holidays['timestamp'])

    weather_data_resampled = weather_data.set_index('time').resample('30T').interpolate(method='linear').reset_index()
    merged_data = pd.merge(agile_prices, weather_data_resampled, how='left', left_on='timestamp', right_on='time')
    merged_data['date'] = pd.to_datetime(merged_data['timestamp'].dt.date)
    bank_holidays['date'] = pd.to_datetime(bank_holidays['timestamp'].dt.date)
    merged_data = pd.merge(merged_data, sunlight_data, how='left', on='date')
    merged_data = pd.merge(merged_data, bank_holidays[['date', 'bank_hol']], how='left', on='date')

    merged_data['bank_hol'] = merged_data['bank_hol'].apply(lambda x: 1 if x == 'yes' else 0).fillna(0)
    merged_data['day_of_week'] = merged_data['timestamp'].dt.dayofweek
    merged_data['hour_of_day'] = merged_data['timestamp'].dt.hour
    merged_data['hour'] = merged_data['timestamp'].dt.hour

    features = ['hour', 'wind_speed_100m (km/h)', 'daylight_duration (s)', 'sunshine_duration (s)', 'day_of_week', 'hour_of_day', 'bank_hol']
    return merged_data, features

# Preprocess and merge the data without the bank_hol feature
def preprocess_and_merge_without_bank_hol(agile_prices, weather_data, sunlight_data, bank_holidays):
    agile_prices['timestamp'] = pd.to_datetime(agile_prices['timestamp']).dt.tz_convert(None)
    weather_data['time'] = pd.to_datetime(weather_data['time']).dt.tz_localize('UTC').dt.tz_convert(None)
    sunlight_data['date'] = pd.to_datetime(sunlight_data['date'], format='%d/%m/%Y')
    bank_holidays['timestamp'] = pd.to_datetime(bank_holidays['timestamp'])

    weather_data_resampled = weather_data.set_index('time').resample('30T').interpolate(method='linear').reset_index()
    merged_data = pd.merge(agile_prices, weather_data_resampled, how='left', left_on='timestamp', right_on='time')
    merged_data['date'] = pd.to_datetime(merged_data['timestamp'].dt.date)
    bank_holidays['date'] = pd.to_datetime(bank_holidays['timestamp'].dt.date)
    merged_data = pd.merge(merged_data, sunlight_data, how='left', on='date')
    merged_data = pd.merge(merged_data, bank_holidays[['date', 'bank_hol']], how='left', on='date')

    merged_data['day_of_week'] = merged_data['timestamp'].dt.dayofweek
    merged_data['hour_of_day'] = merged_data['timestamp'].dt.hour
    merged_data['hour'] = merged_data['timestamp'].dt.hour

    features = ['hour', 'wind_speed_100m (km/h)', 'daylight_duration (s)', 'sunshine_duration (s)', 'day_of_week', 'hour_of_day']
    return merged_data, features

# Preprocess and merge the original data without day_of_week and bank_hol features
def preprocess_and_merge_original(agile_prices, weather_data, sunlight_data):
    agile_prices['timestamp'] = pd.to_datetime(agile_prices['timestamp']).dt.tz_convert(None)
    weather_data['time'] = pd.to_datetime(weather_data['time']).dt.tz_localize('UTC').dt.tz_convert(None)
    sunlight_data['date'] = pd.to_datetime(sunlight_data['date'], format='%d/%m/%Y')

    weather_data_resampled = weather_data.set_index('time').resample('30T').interpolate(method='linear').reset_index()
    merged_data = pd.merge(agile_prices, weather_data_resampled, how='left', left_on='timestamp', right_on='time')
    merged_data['date'] = pd.to_datetime(merged_data['timestamp'].dt.date)
    merged_data = pd.merge(merged_data, sunlight_data, how='left', on='date')

    merged_data = merged_data.drop(columns=['time'])
    merged_data = merged_data.rename(columns={'date': 'date'})
    merged_data['day_of_week'] = merged_data['timestamp'].dt.dayofweek
    merged_data['hour_of_day'] = merged_data['timestamp'].dt.hour
    merged_data['hour'] = merged_data['timestamp'].dt.hour

    features = ['hour', 'wind_speed_100m (km/h)', 'daylight_duration (s)', 'sunshine_duration (s)', 'day_of_week', 'hour_of_day']
    return merged_data, features

# Train the model
def train_model(X_train, y_train, model_path='model.pkl', scaler_path='scaler.pkl'):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    return model, scaler

# Load the model and scaler
def load_model(model_path='model.pkl', scaler_path='scaler.pkl'):
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    return None, None

# Predict electricity prices for tomorrow
def predict_tomorrow(model, scaler, features, wind_speeds, daylight_duration, sunshine_duration_percent):
    sunshine_duration = 0.55 * daylight_duration
    tomorrow = datetime.now().date() + timedelta(days=0)
    timestamps = [datetime(tomorrow.year, tomorrow.month, tomorrow.day, hour, minute) for hour in range(24) for minute in [0, 30]]

    pred_data = {
        'timestamp': timestamps,
        'hour': [timestamp.hour for timestamp in timestamps],
        'minute': [timestamp.minute for timestamp in timestamps],
        'wind_speed_100m (km/h)': [wind_speeds[i // 2] for i in range(len(timestamps))],
        'daylight_duration (s)': [daylight_duration] * len(timestamps),
        'sunshine_duration (s)': [sunshine_duration] * len(timestamps),
        'day_of_week': [timestamps[0].weekday()] * len(timestamps),
        'hour_of_day': [timestamp.hour for timestamp in timestamps],
        'bank_hol': [0] * len(timestamps)  # Assuming tomorrow is not a bank holiday
    }

    pred_df = pd.DataFrame(pred_data)
    X_pred = pred_df[features]
    X_pred_scaled = scaler.transform(X_pred)
    pred_prices = model.predict(X_pred_scaled)

    pred_df['predicted_price'] = pred_prices
    pred_df['timestamp_bst'] = pred_df['timestamp'] + timedelta(hours=1)

    pred_df[['timestamp_bst', 'predicted_price']].to_csv('predicted_prices.csv', index=False)

    return pred_df

def evaluate_model(model, scaler, X_test, y_test):
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    return mae

def evaluate_old_model():
    agile_prices, weather_data, sunlight_data, _ = load_data()  # Ignore bank_holidays
    merged_data, features = preprocess_and_merge_original(agile_prices, weather_data, sunlight_data)

    X = merged_data[features]
    y = merged_data['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_path = 'model_old.pkl'
    scaler_path = 'scaler_old.pkl'
    model, scaler = load_model(model_path, scaler_path)

    if model is None or scaler is None:
        model, scaler = train_model(X_train, y_train, model_path, scaler_path)

    mae = evaluate_model(model, scaler, X_test, y_test)
    return mae

def evaluate_model_without_bank_hol():
    agile_prices, weather_data, sunlight_data, bank_holidays = load_data()
    merged_data, features = preprocess_and_merge_without_bank_hol(agile_prices, weather_data, sunlight_data, bank_holidays)

    X = merged_data[features]
    y = merged_data['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_path = 'model_without_bank_hol.pkl'
    scaler_path = 'scaler_without_bank_hol.pkl'
    model, scaler = load_model(model_path, scaler_path)

    if model is None or scaler is None:
        model, scaler = train_model(X_train, y_train, model_path, scaler_path)

    mae = evaluate_model(model, scaler, X_test, y_test)
    return mae

def main():
    agile_prices, weather_data, sunlight_data, bank_holidays = load_data()
    merged_data, features = preprocess_and_merge(agile_prices, weather_data, sunlight_data, bank_holidays)

    X = merged_data[features]
    y = merged_data['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_path = 'model.pkl'
    scaler_path = 'scaler.pkl'
    model, scaler = load_model(model_path, scaler_path)

    if model is None or scaler is None:
        model, scaler = train_model(X_train, y_train, model_path, scaler_path)

    wind_speeds = [2, 2, 3, 2, 2, 3, 4, 2, 2, 2, 2, 4, 5, 6, 8, 9, 10, 9, 8, 7, 5, 5, 4, 4]
    daylight_duration = timedelta(hours=16, minutes=10, seconds=53).total_seconds()
    sunshine_duration_percent = 0.40

    pred_df = predict_tomorrow(model, scaler, features, wind_speeds, daylight_duration, sunshine_duration_percent)

    # Plot the predicted prices adjusted for BST with time on the x-axis
    plt.figure(figsize=(14, 7))
    plt.plot(pred_df['timestamp_bst'], pred_df['predicted_price'], marker='o', linestyle='-')
    plt.title('Predicted Electricity Prices for Tomorrow (BST)')
    plt.xlabel('Time')
    plt.ylabel('Predicted Price')
    plt.xticks(ticks=pred_df['timestamp_bst'], labels=pred_df['timestamp_bst'].dt.strftime('%H:%M'), rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('predicted_prices_plot.png')
    plt.show()

    # Evaluate the model
    mae = evaluate_model(model, scaler, X_test, y_test)
    print(f'Mean Absolute Error: {mae}')
    return mae, pred_df

# Evaluate the different models
mae_old = evaluate_old_model()
mae_without_bank_hol = evaluate_model_without_bank_hol()
mae_new, pred_df = main()

print(f'Old Model MAE: {mae_old}')
print(f'Model without bank_hol MAE: {mae_without_bank_hol}')
print(f'New Model MAE: {mae_new}')
