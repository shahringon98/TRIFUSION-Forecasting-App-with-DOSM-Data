import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# LSTM Model Definition
class LSTMForecaster(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1, output_size=1):
        super(LSTMForecaster, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Function to create sequences for LSTM
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# ARIMA Forecast Function
@st.cache_data
def arima_forecast(train_data, forecast_steps):
    try:
        model = ARIMA(train_data, order=(5, 1, 0))  # Simplified; tune as needed
        fit = model.fit()
        forecast = fit.forecast(steps=forecast_steps)
        return forecast
    except Exception as e:
        st.error(f"ARIMA failed: {e}")
        return np.zeros(forecast_steps)

# LSTM Forecast Function
@st.cache_data
def lstm_forecast(train_data, forecast_steps, seq_length=4, epochs=50):  # seq_length=4 for quarterly
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(train_data.reshape(-1, 1))

    X, y = create_sequences(scaled_data, seq_length)
    X = torch.from_numpy(X).float().unsqueeze(2)
    y = torch.from_numpy(y).float()

    model = LSTMForecaster()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        outputs = model(X)
        optimizer.zero_grad()
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

    forecasts = []
    inputs = torch.from_numpy(scaled_data[-seq_length:]).float().unsqueeze(0).unsqueeze(2)
    model.eval()
    for _ in range(forecast_steps):
        pred = model(inputs)
        forecasts.append(pred.item())
        inputs = torch.cat((inputs[:, 1:, :], pred.unsqueeze(0).unsqueeze(2)), dim=1)

    return scaler.inverse_transform(np.array(forecasts).reshape(-1, 1)).flatten()

# Dummy LLM Rationale
def dummy_llm_rationale(forecast):
    return "Forecast based on recent trends in economic growth, influenced by factors like export demand and policy rates. Confidence: Medium."

# Meta-Controller
def meta_controller(arima_pred, lstm_pred, val_data):
    val_steps = len(val_data)
    arima_val_pred = arima_forecast(val_data[:-val_steps], val_steps)
    lstm_val_pred = lstm_forecast(val_data[:-val_steps], val_steps)

    arima_rmse = mean_squared_error(val_data[-val_steps:], arima_val_pred, squared=False)
    lstm_rmse = mean_squared_error(val_data[-val_steps:], lstm_val_pred, squared=False)

    total_rmse = arima_rmse + lstm_rmse
    w_arima = (lstm_rmse / total_rmse) if total_rmse > 0 else 0.5
    w_lstm = (arima_rmse / total_rmse) if total_rmse > 0 else 0.5

    ensemble_pred = w_arima * arima_pred + w_lstm * lstm_pred
    return ensemble_pred, {"ARIMA Weight": w_arima, "LSTM Weight": w_lstm}

# Streamlit App
st.title("TRIFUSION Forecasting App with DOSM GDP Data")
st.write("Loads quarterly real GDP (constant 2015 prices, RM million) from DOSM for forecasting.")

@st.cache_data
def load_dosm_data():
    url = 'https://storage.dosm.gov.my/gdp/gdp_qtr_real.parquet'
    df = pd.read_parquet(url)
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['series'] == 'abs'].copy()  # Absolute values
    df['quarter'] = df['date'].dt.year.astype(str) + '-Q' + df['date'].dt.quarter.astype(str)
    df = df[['quarter', 'gdp']].set_index('quarter')
    return df['gdp'].values.astype(float), df.index

data, dates = load_dosm_data()
st.subheader("Loaded DOSM Data Preview")
st.dataframe(pd.DataFrame({"Quarter": dates, "Real GDP (RM Million)": data}))

forecast_steps = st.slider("Forecast Quarters", min_value=1, max_value=8, value=4)

# Split: 80% train, 20% test
split = int(0.8 * len(data))
train_data = data[:split]
test_data = data[split:]

# Forecasts
arima_pred = arima_forecast(train_data, forecast_steps)
lstm_pred = lstm_forecast(train_data, forecast_steps)

# Meta-controller using pseudo-validation from full data
ensemble_pred, weights = meta_controller(arima_pred, lstm_pred, data)

# Dummy LLM
rationale = dummy_llm_rationale(ensemble_pred)

# Display Results
st.subheader("Forecast Results (Next Quarters)")
results = pd.DataFrame({
    "ARIMA": arima_pred,
    "LSTM": lstm_pred,
    "Ensemble": ensemble_pred
}, index=[f"Future Q{i+1}" for i in range(forecast_steps)])
st.dataframe(results)

st.subheader("Weights from Meta-Controller")
st.json(weights)

st.subheader("Dummy LLM Rationale")
st.write(rationale)

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(dates, data, label="Historical GDP")
future_quarters = [f"{dates[-1][:4]}-Q{int(dates[-1][-1]) + i + 1}" if int(dates[-1][-1]) + i < 4 else f"{int(dates[-1][:4]) + 1}-Q{ (int(dates[-1][-1]) + i) % 4 + 1}" for i in range(forecast_steps)]
ax.plot(future_quarters, ensemble_pred, label="Ensemble Forecast", color="green")
ax.set_xlabel("Quarter")
ax.set_ylabel("Real GDP (RM Million)")
ax.legend()
ax.tick_params(axis='x', rotation=45)
st.pyplot(fig)

# Metrics on test data
if len(test_data) >= forecast_steps:
    test_rmse = mean_squared_error(test_data[:forecast_steps], ensemble_pred, squared=False)
    st.write(f"Ensemble RMSE on Test Data: {test_rmse:.2f}")
