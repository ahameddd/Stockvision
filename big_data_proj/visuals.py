import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import load_model
import os


df = pd.read_csv('NVDA_daily_stock_data.csv')

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[['close']])


X_test = []
look_back = 60
for i in range(look_back, len(scaled_data)):
    X_test.append(scaled_data[i-look_back:i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


model_path = 'lstm_model.h5'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"{model_path} does not exist. Please ensure it has been saved correctly.")


model = load_model(model_path)


predicted_prices = model.predict(X_test)
predicted_prices = scaler.inverse_transform(predicted_prices)  

print("Actual Prices:")
print(df['close'][look_back:].values)  

print("\nPredicted Prices:")
print(predicted_prices)

mse = mean_squared_error(df['close'][look_back:].values, predicted_prices.flatten())
mae = mean_absolute_error(df['close'][look_back:].values, predicted_prices.flatten())
rmse = np.sqrt(mse)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")


df['SMA_50'] = df['close'].rolling(window=50).mean()
df['SMA_200'] = df['close'].rolling(window=200).mean()

plt.figure(figsize=(10,6))
plt.plot(df['close'], label='Actual Prices', color='blue')
plt.plot(df.index[look_back:], predicted_prices, label='Predicted Prices', color='orange')
plt.plot(df['SMA_50'], label='50-day Moving Average', color='green', linestyle='--')
plt.plot(df['SMA_200'], label='200-day Moving Average', color='purple', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.title('Stock Price Prediction with Moving Averages')
plt.show()



errors = df['close'][look_back:].values - predicted_prices.flatten()

plt.figure(figsize=(10, 6))
plt.plot(errors, label='Prediction Error', color='red')
plt.title('Prediction Error Over Time')
plt.xlabel('Time')
plt.ylabel('Error')
plt.legend()
plt.show()

plt.figure(figsize=(10,6))
plt.plot(df['close'], label='Actual Prices', color='blue')
plt.plot(df.index[look_back:], predicted_prices, label='Predicted Prices', color='orange')  # Match the indices
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.title('Stock Price Prediction')
plt.show()

