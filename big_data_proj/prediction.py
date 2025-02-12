import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

model = load_model('lstm_nvda_model.h5')  

data = pd.read_csv('NVDA_daily_stock_data.csv')  

data = data[['date', 'close']]
data['date'] = pd.to_datetime(data['date'])
data.sort_values('date', inplace=True) 
data.set_index('date', inplace=True)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['close'].values.reshape(-1, 1))


time_step = 60  
prediction_step = 7  


last_data = scaled_data[-time_step:].reshape(1, time_step, 1)


predictions = []


for _ in range(prediction_step):
    
    next_price = model.predict(last_data)
    
    
    predictions.append(next_price[0, 0])  
    
    
    last_data = np.append(last_data[:, 1:, :], next_price.reshape(1, 1, 1), axis=1)


predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))


last_date = data.index[-1]  
predicted_dates = pd.date_range(start=last_date, periods=prediction_step + 1, freq='B')[1:]

predictions_df = pd.DataFrame(data=predictions, index=predicted_dates, columns=['Predicted Price'])


predictions_df.to_csv('predicted_prices.csv')  


plt.figure(figsize=(14, 5))
plt.plot(predicted_dates, predictions.flatten(), color='red', label='Predicted Prices for Next Week')
plt.title('Stock Price Prediction Using Existing LSTM Model')
plt.xlabel('Date')
plt.ylabel('Price')
plt.xticks(rotation=45)  
plt.legend()
plt.grid()
plt.tight_layout()  
plt.show()
