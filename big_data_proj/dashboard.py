import dash
from dash import dcc, html
import plotly.graph_objs as go
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras.models import load_model


df = pd.read_csv('NVDA_daily_stock_data.csv')


df['SMA_50'] = df['close'].rolling(window=50).mean()
df['SMA_200'] = df['close'].rolling(window=200).mean()


scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[['close']])


look_back = 60
X_test = []
for i in range(look_back, len(scaled_data)):
    X_test.append(scaled_data[i - look_back:i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

model = load_model('lstm_nvda_model.h5')
predicted_prices = model.predict(X_test)


predicted_prices_rescaled = scaler.inverse_transform(predicted_prices)

predicted_prices_full = np.zeros(len(df))  
predicted_prices_full[:] = np.nan 
predicted_prices_full[look_back:] = predicted_prices_rescaled.flatten()


predicted_df = pd.read_csv('predicted_prices.csv')
predicted_df['Date'] = pd.to_datetime(predicted_df['Date'])
predicted_prices_from_csv = predicted_df['Predicted Price'].values
predicted_dates = predicted_df['Date']

app = dash.Dash(__name__)

app.layout = html.Div(children=[
    html.H1('Stock Price Dashboard'),


    dcc.Graph(
        id='stock-graph',
        figure={
            'data': [
                go.Scatter(x=df.index, y=df['close'], mode='lines', name='Actual Prices', line={'color': 'blue'}),
                go.Scatter(x=df.index, y=predicted_prices_full, mode='lines', name='Predicted Prices', line={'color': 'orange'}),
                go.Scatter(x=df.index, y=df['SMA_50'], mode='lines', name='50-Day Moving Average', line={'dash': 'dash', 'color': 'green'}),
                go.Scatter(x=df.index, y=df['SMA_200'], mode='lines', name='200-Day Moving Average', line={'dash': 'dash', 'color': 'purple'}),
            ],
            'layout': go.Layout(
                title='Stock Prices with Predictions and Moving Averages',
                xaxis={'title': 'Time'},
                yaxis={'title': 'Price'},
                hovermode='x unified'
            )
        }
    ),

  
    dcc.Graph(
        id='volume-bar',
        figure={
            'data': [
                go.Bar(x=df.index, y=df['volume'], name='Volume', marker={'color': 'lightblue'})
            ],
            'layout': go.Layout(
                title='Trading Volume',
                xaxis={'title': 'Time'},
                yaxis={'title': 'Volume'},
                barmode='stack'
            )
        }
    ),

 
    dcc.Graph(
        id='predicted-stock-graph',
        figure={
            'data': [
                go.Scatter(x=predicted_dates, y=predicted_prices_from_csv, mode='lines', name='Predicted Prices from CSV', line={'color': 'orange'}),
            ],
            'layout': go.Layout(
                title='Predicted Stock Prices from CSV',
                xaxis={'title': 'Date'},
                yaxis={'title': 'Predicted Price'},
                hovermode='x unified'
            )
        }
    )

])

if __name__ == '__main__':
    app.run_server(debug=True)
    