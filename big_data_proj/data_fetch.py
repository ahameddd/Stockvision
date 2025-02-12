import requests
import pandas as pd

API_KEY = '2Q9247HS4H8DSL1U'
symbol = 'NVDA' 
url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={API_KEY}&outputsize=full'

response = requests.get(url)
data = response.json()

df = pd.DataFrame(data['Time Series (Daily)']).T
df.columns = ['open', 'high', 'low', 'close', 'volume']
df.index = pd.to_datetime(df.index)


df.to_csv(f'{symbol}_daily_stock_data.csv')
