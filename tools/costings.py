import pandas as pd
import matplotlib.pyplot as plt
import requests
from datetime import time

# Load the consumption data
consumption_data = pd.read_csv('../sample_data/consumption.csv')

# Strip any leading or trailing spaces from the column names
consumption_data.columns = consumption_data.columns.str.strip()

# Rename the 'Consumption (kWh)' column to 'Consumption'
consumption_data.rename(columns={'Consumption (kWh)': 'Consumption'}, inplace=True)

# Convert the 'Start' column to datetime
consumption_data['Start'] = pd.to_datetime(consumption_data['Start'])

# Set the 'Start' column as the index for easier resampling
consumption_data.set_index('Start', inplace=True)

# Define your tariff details
off_peak_hours = (time(1, 30), time(8, 30))
off_peak_rate = 0.133119
on_peak_rate = 0.314475
product_code = 'AGILE-18-02-21'
tariff_code = 'E-1R-AGILE-18-02-21-J'
api_key = ''  # Replace with your actual API key
energy_shift = True

# Function to determine if a time is in the off-peak period
def is_off_peak(timestamp):
    return off_peak_hours[0] <= timestamp.time() < off_peak_hours[1]

# Function to fetch agile prices for a given date range in chunks
def fetch_agile_prices(start_date, end_date, product_code, tariff_code, api_key, chunk_size=2):
    headers = {
        'Authorization': api_key
    }
    
    date_range = pd.date_range(start_date, end_date, freq=f'{chunk_size}D')
    all_prices = pd.DataFrame()

    for i in range(len(date_range) - 1):
        period_from = date_range[i].strftime('%Y-%m-%dT%H:%M:%SZ')
        period_to = date_range[i + 1].strftime('%Y-%m-%dT%H:%M:%SZ')
        
        url = f'https://api.octopus.energy/v1/products/{product_code}/electricity-tariffs/{tariff_code}/standard-unit-rates/'
        params = {
            'period_from': period_from,
            'period_to': period_to
        }

        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            data = response.json()
            prices = pd.DataFrame(data['results'])
            prices['timestamp'] = pd.to_datetime(prices['valid_from'])
            prices.set_index('timestamp', inplace=True)
            prices.sort_index(inplace=True)
            prices['price'] = prices['value_inc_vat'] / 100  # Convert to pounds
            all_prices = pd.concat([all_prices, prices[['price']]])
        else:
            raise Exception(f"Failed to fetch agile prices for {period_from} to {period_to}")

    return all_prices

# Determine the date range from the consumption data
start_date = consumption_data.index.min().date().strftime('%Y-%m-%dT%H:%M:%SZ')
end_date = consumption_data.index.max().date().strftime('%Y-%m-%dT%H:%M:%SZ')

# Fetch the agile prices for the determined date range
agile_prices = fetch_agile_prices(start_date, end_date, product_code, tariff_code, api_key)

# Calculate daily costs for the current tariff
def calculate_current_tariff_cost(usage):
    cost = 0
    for timestamp, row in usage.iterrows():
        if is_off_peak(timestamp):
            cost += row['Consumption'] * off_peak_rate
        else:
            cost += row['Consumption'] * on_peak_rate
    return cost

# Resample to daily data
daily_usage = consumption_data.resample('D').sum()
daily_usage['current_tariff_cost'] = consumption_data.resample('D').apply(calculate_current_tariff_cost)

# Calculate daily costs for the agile tariff
daily_usage['agile_tariff_cost'] = 0
for date, usage_row in daily_usage.iterrows():
    daily_cost = 0
    daily_usage_period = consumption_data[date:date + pd.Timedelta(days=1)]
    daily_prices_period = agile_prices[date:date + pd.Timedelta(days=1)]
    
    if not daily_prices_period.empty:
        for timestamp, usage_row in daily_usage_period.iterrows():
            if timestamp in daily_prices_period.index:
                price = daily_prices_period.loc[timestamp, 'price']
                daily_cost += usage_row['Consumption'] * price
        daily_usage.at[date, 'agile_tariff_cost'] = daily_cost

# Identify the baseline usage and excess usage from the early morning spike
baseline_usage = consumption_data['Consumption'].min()
consumption_data['excess_usage'] = consumption_data['Consumption'] - baseline_usage
consumption_data.loc[consumption_data['excess_usage'] < 0, 'excess_usage'] = 0

# Shift excess usage to the cheapest time of the day
shift_percentage = 0.5 # 50% of excess usage

daily_usage['shifted_usage_cost'] = daily_usage['agile_tariff_cost']
for date in daily_usage.index:
    daily_usage_period = consumption_data[date:date + pd.Timedelta(days=1)]
    daily_prices_period = agile_prices[date:date + pd.Timedelta(days=1)]
    if not daily_prices_period.empty:
        cheapest_time = daily_prices_period['price'].idxmin()
    
        if cheapest_time in daily_usage_period.index:
            excess_usage = daily_usage_period['excess_usage'].sum()
            
            # Calculate the amount to be shifted based on the specified percentage
            amount_to_shift = excess_usage * shift_percentage
            
            # Calculate the average cost of the excess usage before shifting
            avg_excess_cost = daily_usage_period.loc[daily_usage_period['excess_usage'] > 0, 'Consumption'].mean()
            
            # Calculate the shifted cost at the cheapest time
            shifted_cost = amount_to_shift * daily_prices_period.loc[cheapest_time, 'price']
            
            # Update the shifted usage cost in the daily usage DataFrame
            daily_usage.at[date, 'shifted_usage_cost'] -= amount_to_shift * avg_excess_cost
            daily_usage.at[date, 'shifted_usage_cost'] += shifted_cost

# Save the results to a CSV
output_filename = 'daily_cost_comparison.csv'
daily_usage.to_csv(output_filename)

# Confirm the CSV has been saved
print(f"The daily cost comparison has been saved to {output_filename}.")

# Calculate total costs
total_current_tariff_cost = daily_usage['current_tariff_cost'].sum()
total_agile_tariff_cost = daily_usage['agile_tariff_cost'].sum()
total_shifted_usage_cost = daily_usage['shifted_usage_cost'].sum() if energy_shift else None

# Output the total costs
print(f"Total Current Tariff Cost: £{total_current_tariff_cost:.2f}")
print(f"Total Agile Tariff Cost: £{total_agile_tariff_cost:.2f}")
if energy_shift:
    print(f"Total Shifted Usage Cost: £{total_shifted_usage_cost:.2f}")

# Plotting the daily costs
plt.figure(figsize=(12, 6))
plt.plot(daily_usage.index, daily_usage['current_tariff_cost'], label='Current Tariff Cost', color='blue')
plt.plot(daily_usage.index, daily_usage['agile_tariff_cost'], label='Agile Tariff Cost', color='green')
if energy_shift:
    plt.plot(daily_usage.index, daily_usage['shifted_usage_cost'], label='Shifted Usage Cost', color='red')
plt.xlabel('Date')
plt.ylabel('Cost (£)')
plt.title('Daily Electricity Cost Comparison')
plt.legend()
plt.grid(True)
plt.show()
