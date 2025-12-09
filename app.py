import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.vector_ar.var_model import VAR
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Multivariate LSTM Model (input multiple series, output GDP)
class MultivariateLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=50, num_layers=1, output_size=1):
        super(MultivariateLSTM, self).__init__()
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

# Create sequences for multivariate data
def create_multivariate_sequences(data, seq_length, target_col=0):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length, target_col]  # Target is GDP
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# VAR Forecast Function
@st.cache_data
def var_forecast(train_df, forecast_steps):
    try:
        model = VAR(train_df)
        fit = model.fit(maxlags=4)  # Adjust lags as needed
        forecast = fit.forecast(train_df.values[-fit.k_ar:], steps=forecast_steps)
        return forecast[:, 0]  # Return GDP forecasts
    except Exception as e:
        st.error(f"VAR failed: {e}")
        return np.zeros(forecast_steps)

# Multivariate LSTM Forecast Function
@st.cache_data
def lstm_forecast(train_data, forecast_steps, seq_length=4, epochs=50):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(train_data)  # train_data is (samples, 2)

    X, y = create_multivariate_sequences(scaled_data, seq_length)
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).float().unsqueeze(1)

    model = MultivariateLSTM(input_size=2)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        outputs = model(X)
        optimizer.zero_grad()
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

    # Forecast (autoregressive: use last known M2, predict GDP sequentially)
    forecasts = []
    inputs = torch.from_numpy(scaled_data[-seq_length:]).float().unsqueeze(0)
    model.eval()
    for _ in range(forecast_steps):
        pred = model(inputs)
        forecasts.append(pred.item())
        # Shift: append predicted GDP, keep last M2 (assume M2 exog or repeat last)
        new_input = torch.cat((inputs[0, -1, 1].unsqueeze(0), pred[0]), dim=0).unsqueeze(0)  # [M2_last, GDP_pred]
        inputs = torch.cat((inputs[:, 1:, :], new_input.unsqueeze(0)), dim=1)

    return scaler.inverse_transform(np.concatenate((np.zeros((forecast_steps, 1)), np.array(forecasts).reshape(-1, 1)), axis=1))[:, 1]  # Inverse GDP

# Dummy LLM Rationale
def dummy_llm_rationale(forecast):
    return "Forecast influenced by GDP trends and M2 money supply growth, reflecting liquidity impacts on economic activity. Confidence: Medium."

# Meta-Controller
def meta_controller(var_pred, lstm_pred, val_data):
    val_steps = min(4, len(val_data) // 2)  # Small val for demo
    val_df = pd.DataFrame(val_data, columns=['gdp', 'm2'])
    var_val_pred = var_forecast(val_df.iloc[:-val_steps], val_steps)
    lstm_val_pred = lstm_forecast(val_data[:-val_steps], val_steps)

    var_rmse = mean_squared_error(val_data[-val_steps:, 0], var_val_pred, squared=False)
    lstm_rmse = mean_squared_error(val_data[-val_steps:, 0], lstm_val_pred, squared=False)

    total_rmse = var_rmse + lstm_rmse
    w_var = (lstm_rmse / total_rmse) if total_rmse > 0 else 0.5
    w_lstm = (var_rmse / total_rmse) if total_rmse > 0 else 0.5

    ensemble_pred = w_var * var_pred + w_lstm * lstm_pred
    return ensemble_pred, {"VAR Weight": w_var, "LSTM Weight": w_lstm}

# Streamlit App
st.title("Multivariate TRIFUSION Forecasting App with DOSM GDP & BNM M2 Data")
st.write("Loads quarterly real GDP (DOSM) and resamples monthly M2 (BNM) to quarterly for multivariate forecasting.")

@st.cache_data
def load_data():
    # Load GDP (quarterly)
    gdp_url = 'https://storage.dosm.gov.my/gdp/gdp_qtr_real.parquet'
    gdp_df = pd.read_parquet(gdp_url)
    gdp_df['date'] = pd.to_datetime(gdp_df['date'])
    gdp_df = gdp_df[gdp_df['series'] == 'abs'].copy()
    gdp_df['quarter'] = gdp_df['date'].dt.to_period('Q').astype(str)
    gdp_df = gdp_df.set_index('date')[['value', 'quarter']]  # Corrected to 'value'

    # Load M2 (monthly, resample to quarterly end)
    m2_url = 'https://storage.data.gov.my/finsector/money_aggregates.parquet'
    m2_full_df = pd.read_parquet(m2_url)
    print("Unique measures in M2 data:", m2_full_df['measure'].unique())  # Check console
    m2_df = m2_full_df[m2_full_df['measure'].str.contains('broad_money_m2', case=False)]  # Updated filter
    m2_df['date'] = pd.to_datetime(m2_df['date'])
    m2_df = m2_df.set_index('date')['supply'].resample('Q').last()  # End of quarter
    m2_df = pd.DataFrame({'m2': m2_df})
    m2_df['quarter'] = m2_df.index.to_period('Q').astype(str)

    # Merge on quarter
    merged_df = gdp_df.merge(m2_df, on='quarter', how='inner')
    data = merged_df[['value', 'm2']].values.astype(float)  # Corrected to 'value' for GDP
    quarters = merged_df['quarter'].values

    return data, quarters

data, quarters = load_data()
st.subheader("Loaded Multivariate Data Preview")
st.dataframe(pd.DataFrame({"Quarter": quarters, "Real GDP (RM Million)": data[:, 0], "M2 (RM Million)": data[:, 1]}))

forecast_steps = st.slider("Forecast Quarters", min_value=1, max_value=8, value=4)

# Split: 80% train, 20% test
split = int(0.8 * len(data))
train_data = data[:split]
test_data = data[split:]

# Forecasts (GDP)
var_pred = var_forecast(pd.DataFrame(train_data, columns=['gdp', 'm2']), forecast_steps)
lstm_pred = lstm_forecast(train_data, forecast_steps)

# Meta-controller using pseudo-validation on full data (focus on GDP)
ensemble_pred, weights = meta_controller(var_pred, lstm_pred, data)

# Dummy LLM
rationale = dummy_llm_rationale(ensemble_pred)

# Display Results
st.subheader("GDP Forecast Results (Next Quarters)")
results = pd.DataFrame({
    "VAR": var_pred,
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
ax.plot(quarters, data[:, 0], label="Historical GDP")
future_quarters = [f"{quarters[-1][:4]}-Q{int(quarters[-1][-1]) + i + 1}" if int(quarters[-1][-1]) + i < 4 else f"{int(quarters[-1][:4]) + 1}-Q{(int(quarters[-1][-1]) + i) % 4 + 1}" for i in range(forecast_steps)]
ax.plot(future_quarters, ensemble_pred, label="Ensemble GDP Forecast", color="green")
ax.plot(quarters, data[:, 1] / 10, label="Historical M2 (scaled /10)", color="orange", alpha=0.5)  # Scaled for visibility
ax.set_xlabel("Quarter")
ax.set_ylabel("Value (RM Million)")
ax.legend()
ax.tick_params(axis='x', rotation=45)
st.pyplot(fig)

# Metrics on test data (GDP)
if len(test_data) >= forecast_steps:
    test_rmse = mean_squared_error(test_data[:forecast_steps, 0], ensemble_pred, squared=False)
    st.write(f"Ensemble RMSE on Test GDP: {test_rmse:.2f}")
