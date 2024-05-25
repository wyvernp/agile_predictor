import pandas as pd
from datetime import datetime, time
import matplotlib.pyplot as plt

energy_shift = True
# Load the data
agile_prices = pd.read_csv('../sample_data/csv_agile_J_South_Eastern_England2024.csv')
usage_data = pd.read_csv('../sample_data/power_usage2024.csv')

# Convert timestamp columns to datetime
agile_prices['timestamp'] = pd.to_datetime(agile_prices['timestamp'])
usage_data['timestamp'] = pd.to_datetime(usage_data['timestamp'])

# Convert Agile prices from pence to pounds
agile_prices['price'] = agile_prices['price'] / 100

# Set the timestamps as the index for easier resampling
agile_prices.set_index('timestamp', inplace=True)
usage_data.set_index('timestamp', inplace=True)

# Define your tariff details
off_peak_hours = (time(1, 30), time(8, 30))
off_peak_rate = 0.133119
on_peak_rate = 0.314475

# Function to determine if a time is in the off-peak period
def is_off_peak(timestamp):
    return off_peak_hours[0] <= timestamp.time() < off_peak_hours[1]

# Calculate daily costs for the current tariff
def calculate_current_tariff_cost(usage):
    cost = 0
    for timestamp, row in usage.iterrows():
        if is_off_peak(timestamp):
            cost += row['kwh'] * off_peak_rate
        else:
            cost += row['kwh'] * on_peak_rate
    return cost

# Resample to daily data
daily_usage = usage_data.resample('D').sum()
daily_usage['current_tariff_cost'] = usage_data.resample('D').apply(calculate_current_tariff_cost)

# Calculate daily costs for the agile tariff
daily_usage['agile_tariff_cost'] = 0
for date, usage_row in daily_usage.iterrows():
    daily_cost = 0
    daily_usage_period = usage_data[date:date + pd.Timedelta(days=1)]
    daily_prices_period = agile_prices[date:date + pd.Timedelta(days=1)]
    
    for timestamp, usage_row in daily_usage_period.iterrows():
        if timestamp in daily_prices_period.index:
            price = daily_prices_period.loc[timestamp, 'price']
            daily_cost += usage_row['kwh'] * price
    daily_usage.at[date, 'agile_tariff_cost'] = daily_cost

# Identify the baseline usage and excess usage from the early morning spike
baseline_usage = usage_data['kwh'].min()
usage_data['excess_usage'] = usage_data['kwh'] - baseline_usage
usage_data.loc[usage_data['excess_usage'] < 0, 'excess_usage'] = 0

# Shift excess usage to the cheapest time of the day
daily_usage['shifted_usage_cost'] = daily_usage['agile_tariff_cost']
for date in daily_usage.index:
    daily_usage_period = usage_data[date:date + pd.Timedelta(days=1)]
    daily_prices_period = agile_prices[date:date + pd.Timedelta(days=1)]
    cheapest_time = daily_prices_period['price'].idxmin()
    if cheapest_time in daily_usage_period.index:
        excess_usage = daily_usage_period['excess_usage'].sum()
        shifted_cost = excess_usage * daily_prices_period.loc[cheapest_time, 'price']
        daily_usage.at[date, 'shifted_usage_cost'] -= excess_usage * daily_usage_period.loc[daily_usage_period['excess_usage'] > 0, 'kwh'].mean()
        daily_usage.at[date, 'shifted_usage_cost'] += shifted_cost

# Save the results to a CSV
output_filename = 'daily_cost_comparison.csv'
daily_usage.to_csv(output_filename)

# Confirm the CSV has been saved
print(f"The daily cost comparison has been saved to {output_filename}.")

# Plotting the daily costs
plt.figure(figsize=(12, 6))
plt.plot(daily_usage.index, daily_usage['current_tariff_cost'], label='Current Tariff Cost', color='blue')
plt.plot(daily_usage.index, daily_usage['agile_tariff_cost'], label='Agile Tariff Cost', color='green')
if energy_shift == True
    plt.plot(daily_usage.index, daily_usage['shifted_usage_cost'], label='Shifted Usage Cost', color='red')
plt.xlabel('Date')
plt.ylabel('Cost (£)')
plt.title('Daily Electricity Cost Comparison')
plt.legend()
plt.grid(True)
plt.show()