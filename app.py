# app.py - Minimal, Stable Version
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import warnings
warnings.filterwarnings('ignore')

# --- FIX #1: Memory Management ---
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True) if tf.config.experimental.list_physical_devices('GPU') else None

# --- FIX #2: Simpler State Management ---
if 'data' not in st.session_state: st.session_state.data = None
if 'model' not in st.session_state: st.session_state.model = None

st.title("📈 LSTM Forecaster - Minimal Version")

# --- DATA LOADING (with fallback) ---
st.sidebar.header("1. Load Data")
@st.cache_data
def load_data():
    try:
        url = "https://raw.githubusercontent.com/plotly/datasets/master/timeseries.csv"
        df = pd.read_csv(url, index_col=0, parse_dates=True)
        return df
    except:
        # Fallback: generate synthetic data
        st.warning("⚠️ Could not load dataset, using synthetic data")
        dates = pd.date_range('2020-01-01', periods=500)
        data = np.cumsum(np.random.randn(500)) + 100
        return pd.DataFrame(data, index=dates, columns=['value'])

if st.sidebar.button("Load Sample Data"):
    st.session_state.data = load_data()
    st.success("✅ Data loaded!")

# Only show the rest if data exists
if st.session_state.data is not None:
    df = st.session_state.data
    target_col = st.sidebar.selectbox("Select column", df.columns.tolist())
    series = df[target_col].dropna()
    
    # --- PREPROCESSING ---
    st.sidebar.header("2. Preprocess")
    lookback = st.sidebar.slider("Lookback", 5, 50, 10)
    epochs = st.sidebar.slider("Epochs", 5, 50, 20)
    
    if st.sidebar.button("Prepare Data"):
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()
        
        X, y = [], []
        for i in range(len(scaled) - lookback):
            X.append(scaled[i:i+lookback])
            y.append(scaled[i+lookback])
        
        X = np.array(X).reshape(-1, lookback, 1)
        y = np.array(y)
        
        # Store in session
        st.session_state.X = X
        st.session_state.y = y
        st.session_state.scaler = scaler
        st.session_state.lookback = lookback
        st.success("✅ Data prepared!")
    
    # --- TRAIN MODEL ---
    if hasattr(st.session_state, 'X'):
        if st.sidebar.button("Train LSTM"):
            with st.spinner("Training..."):
                model = Sequential([
                    LSTM(50, activation='relu', input_shape=(st.session_state.lookback, 1)),
                    Dropout(0.2),
                    Dense(1)
                ])
                model.compile(optimizer='adam', loss='mse')
                model.fit(st.session_state.X, st.session_state.y, epochs=epochs, verbose=0)
                st.session_state.model = model
                st.success("✅ Model trained!")
        
        # --- PREDICT & VISUALIZE ---
        if st.session_state.model is not None:
            model = st.session_state.model
            predictions = model.predict(st.session_state.X, verbose=0)
            
            # Invert scaling
            pred_actual = st.session_state.scaler.inverse_transform(predictions)
            y_actual = st.session_state.scaler.inverse_transform(st.session_state.y.reshape(-1, 1))
            
            # Plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=series.index[st.session_state.lookback:], y=y_actual.flatten(), name="Actual", line=dict(color="blue")))
            fig.add_trace(go.Scatter(x=series.index[st.session_state.lookback:], y=pred_actual.flatten(), name="Predicted", line=dict(color="red", dash='dot')))
            fig.update_layout(title="Forecast", height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Metrics
            rmse = np.sqrt(mean_squared_error(y_actual, pred_actual))
            mae = mean_absolute_error(y_actual, pred_actual)
            st.metric("RMSE", f"{rmse:.2f}")
            st.metric("MAE", f"{mae:.2f}")

# Debug info
with st.expander("Debugging Info"):
    st.text("Session State Keys:")
    st.write(list(st.session_state.keys()))
    st.text("TensorFlow Version:")
    st.write(tf.__version__)
