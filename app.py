import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import pmdarima as pm
import plotly.graph_objects as go

# LSTM Model Definition
class LSTMForecaster(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1, output_size=1):
        super(LSTMForecaster, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Function to create sequences for LSTM
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i:i+seq_length])
        ys.append(data[i+seq_length])
    return np.array(xs), np.array(ys)

# ARIMA Forecast Function (auto_arima)
@st.cache_data
def arima_forecast(train_data, forecast_steps):
    try:
        model = pm.auto_arima(train_data, seasonal=False, stepwise=True, suppress_warnings=True)
        forecast = model.predict(n_periods=forecast_steps)
        return forecast
    except Exception as e:
        st.error(f"ARIMA failed: {e}")
        return np.zeros(forecast_steps)

# LSTM Forecast Function (with GPU support)
@st.cache_data
def lstm_forecast(train_data, forecast_steps, seq_length, hidden_size, num_layers, epochs, lr):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(train_data.reshape(-1, 1))

    X, y = create_sequences(scaled_data, seq_length)
    X = torch.from_numpy(X).float().unsqueeze(2).to(device)
    y = torch.from_numpy(y).float().to(device)

    model = LSTMForecaster(input_size=1, hidden_size=hidden_size, num_layers=num_layers, output_size=1).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        outputs = model(X)
        optimizer.zero_grad()
        loss = criterion(outputs.view(-1), y.view(-1))
        loss.backward()
        optimizer.step()

    # Prediction loop
    forecasts = []
    inputs = torch.from_numpy(scaled_data[-seq_length:]).float().unsqueeze(0).unsqueeze(2).to(device)
    model.eval()
    for _ in range(forecast_steps):
        pred = model(inputs)
        forecasts.append(pred.item())
        pred_reshaped = pred.unsqueeze(0).unsqueeze(2)
        inputs = torch.cat((inputs[:, 1:, :], pred_reshaped), dim=1)

    return scaler.inverse_transform(np.array(forecasts).reshape(-1, 1)).flatten()

# Meta-Controller with validation split
def meta_controller(arima_pred, lstm_pred, train_data, val_data, seq_length,
                    hidden_size, num_layers, epochs, lr):
    # ARIMA validation
    arima_val_pred = arima_forecast(train_data, len(val_data))
    # LSTM validation
    lstm_val_pred = lstm_forecast(train_data, len(val_data), seq_length, hidden_size, num_layers, epochs, lr)

    arima_rmse = mean_squared_error(val_data, arima_val_pred, squared=False)
    lstm_rmse = mean_squared_error(val_data, lstm_val_pred, squared=False)
    total_rmse = arima_rmse + lstm_rmse
    w_arima = lstm_rmse / total_rmse if total_rmse > 0 else 0.5
    w_lstm = arima_rmse / total_rmse if total_rmse > 0 else 0.5

    ensemble_pred = w_arima * arima_pred + w_lstm * lstm_pred
    return ensemble_pred, {"ARIMA Weight": w_arima, "LSTM Weight": w_lstm,
                           "ARIMA RMSE (val)": arima_rmse, "LSTM RMSE (val)": lstm_rmse}

# Dummy LLM Rationale
def dummy_llm_rationale(forecast):
    return "Forecast based on recent trends in economic growth, influenced by factors like export demand and policy rates. Confidence: Medium."

# Load DOSM Data
@st.cache_data
def load_dosm_data():
    url = 'https://storage.dosm.gov.my/gdp/gdp_qtr_real.parquet'
    df = pd.read_parquet(url)
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['series'] == 'abs'].copy()  # Absolute values
    df['quarter'] = df['date'].dt.year.astype(str) + '-Q' + df['date'].dt.quarter.astype(str)
    df = df[['quarter', 'gdp']].set_index('quarter')
    return df['gdp'].values.astype(float), df.index

# Streamlit UI
st.title("TRIFUSION Forecasting App with DOSM GDP Data")
st.write("Loads quarterly real GDP (constant 2015 prices, RM million) from DOSM for forecasting.")

data, dates = load_dosm_data()
st.subheader("Loaded DOSM Data Preview")
st.dataframe(pd.DataFrame({"Quarter": dates, "Real GDP (RM Million)": data}))

# Sidebar for LSTM Hyperparameters
st.sidebar.title("LSTM Hyperparameters")
seq_length = st.sidebar.slider("Sequence Length", min_value=2, max_value=8, value=4)
hidden_size = st.sidebar.slider("Hidden Size", min_value=4, max_value=128, value=50)
num_layers = st.sidebar.slider("Num Layers", min_value=1, max_value=3, value=1)
epochs = st.sidebar.slider("Epochs", min_value=10, max_value=200, value=50)
learning_rate = st.sidebar.number_input("Learning Rate", min_value=1e-5, max_value=1e-1, value=0.001, format="%.5f")

forecast_steps = st.slider("Forecast Quarters", min_value=1, max_value=8, value=4)

# Split: 70% train, 15% val, 15% test
n = len(data)
train_split = int(0.7 * n)
val_split = int(0.85 * n)
train_data = data[:train_split]
val_data = data[train_split:val_split]
test_data = data[val_split:]

# Forecasts
arima_pred = arima_forecast(train_data, forecast_steps)
lstm_pred = lstm_forecast(train_data, forecast_steps, seq_length, hidden_size, num_layers, epochs, learning_rate)

# Meta-controller with validation split
ensemble_pred, weights = meta_controller(
    arima_pred, lstm_pred, train_data, val_data, seq_length, hidden_size, num_layers, epochs, learning_rate
)

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

st.subheader("Weights and RMSE from Meta-Controller")
st.json(weights)

st.subheader("Dummy LLM Rationale")
st.write(rationale)

# Interactive Plotly chart
future_quarters = []
q_year, q_num = int(dates[-1][:4]), int(dates[-1][-1])
for i in range(forecast_steps):
    next_q = q_num + i + 1
    if next_q <= 4:
        future_quarters.append(f"{q_year}-Q{next_q}")
    else:
        years_add = (next_q - 1) // 4
        xq = ((next_q - 1) % 4) + 1
        future_quarters.append(f"{q_year + years_add}-Q{xq}")

fig = go.Figure()
# Historical
fig.add_trace(go.Scatter(x=list(dates), y=data, mode='lines+markers', name='Historical GDP'))
# Ensemble forecast
fig.add_trace(go.Scatter(x=future_quarters, y=ensemble_pred, mode='lines+markers', name='Ensemble Forecast', line=dict(color='green')))
fig.update_layout(xaxis_title='Quarter', yaxis_title='Real GDP (RM Million)', width=950, height=450)
st.plotly_chart(fig)

# Metrics on test data
if len(test_data) >= forecast_steps:
    test_rmse = mean_squared_error(test_data[:forecast_steps], ensemble_pred, squared=False)
    st.write(f"Ensemble RMSE on Test Data: {test_rmse:.2f}")
