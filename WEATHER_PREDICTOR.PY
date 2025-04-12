#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import datetime

np.random.seed(42)
dates = pd.date_range(start="2020-01-01", end="2024-12-31", freq="D")
n_days = len(dates)

data = {
    "Date": dates,
    "Temperature": 20 + 10 * np.sin(np.arange(n_days) * 2 * np.pi / 365) + np.random.normal(0, 2, n_days),
    "Rainfall": np.maximum(0, 5 * np.sin(np.arange(n_days) * 2 * np.pi / 365) + np.random.normal(0, 3, n_days)),
    "Humidity": 60 + 20 * np.sin(np.arange(n_days) * 2 * np.pi / 365) + np.random.normal(0, 5, n_days),
    "Wind_Speed": np.random.uniform(1, 10, n_days)
}

df = pd.DataFrame(data)

df["DayOfYear"] = df["Date"].dt.dayofyear
df["Month"] = df["Date"].dt.month
df["Year"] = df["Date"].dt.year

inputs = ["DayOfYear", "Month", "Year", "Wind_Speed"]
outputs = ["Temperature", "Rainfall", "Humidity"]

X = df[inputs]
y = df[outputs]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

errors = mean_squared_error(y_test, y_pred, multioutput="raw_values")
accuracy = r2_score(y_test, y_pred, multioutput="raw_values")

print("How Good Are the Predictions?")
for i, output in enumerate(outputs):
    print(f"{output}: Error = {errors[i]:.2f}, Accuracy = {accuracy[i]:.2f}")

def predict_weather(date_str, wind_speed=5.0):
    try:
        input_date = pd.to_datetime(date_str)
        day_of_year = input_date.dayofyear
        month = input_date.month
        year = input_date.year
        input_data = np.array([[day_of_year, month, year, wind_speed]])
        prediction = model.predict(input_data)
        return {
            "Date": date_str,
            "Temperature (°C)": round(prediction[0][0], 1),
            "Rainfall (mm)": round(prediction[0][1], 1),
            "Humidity (%)": round(prediction[0][2], 1)
        }
    except:
        return "Oops! Use date format YYYY-MM-DD, like 2025-04-13."

print("\nWeather Forecast for April 13, 2025:")
print(predict_weather("2025-04-13"))

plt.figure(figsize=(8, 4))
plt.plot(y_test["Temperature"].values[:30], label="Real Temperature", marker="o")
plt.plot(y_pred[:30, 0], label="Predicted Temperature", marker="x")
plt.title("Real vs Predicted Temperature")
plt.xlabel("Days")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.show()

import joblib
joblib.dump(model, "my_weather_predictor.pkl")
print("\nModel saved as 'my_weather_predictor.pkl'. To be reused later!")