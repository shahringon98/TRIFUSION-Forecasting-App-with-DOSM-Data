import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.stattools import adfuller
import io
import base64
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="LSTM Time Series Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Session state for persistence
if 'data' not in st.session_state:
    st.session_state.data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'search_results' not in st.session_state:
    st.session_state.search_results = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None

# Title
st.title("📈 LSTM Time Series Forecasting")
st.markdown("Built with Streamlit + TensorFlow")

# Sidebar
st.sidebar.header("Configuration")

# 1) DATA LOADING
st.header("1️⃣ Data Loading")
data_source = st.sidebar.radio("Data Source", ["Air Passengers (built-in)", "Upload CSV"])

@st.cache_data
def load_sample_data():
    url = "https://raw.githubusercontent.com/AileenNielsen/TimeSeriesAnalysisWithPython/master/data/AirPassengers.csv"
    df = pd.read_csv(url)
    df['Month'] = pd.to_datetime(df['Month'])
    df.set_index('Month', inplace=True)
    return df

@st.cache_data
def load_uploaded_file(file):
    df = pd.read_csv(file, index_col=0, parse_dates=True)
    return df

if data_source == "Air Passengers (built-in)":
    with st.spinner("Loading sample dataset..."):
        st.session_state.data = load_sample_data()
    st.success("✅ Loaded Air Passengers dataset")
else:
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    if uploaded_file:
        try:
            st.session_state.data = load_uploaded_file(uploaded_file)
            st.success("✅ File uploaded successfully")
        except Exception as e:
            st.error(f"Error loading file: {e}")

if st.session_state.data is not None:
    df = st.session_state.data
    st.write(f"Dataset shape: {df.shape}")
    st.dataframe(df.head())

    # Select column
    target_col = st.selectbox("Select target column", df.columns.tolist())
    series = df[target_col].dropna()

    # Plot raw data
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.index, y=series.values, mode='lines', name=target_col))
    fig.update_layout(title="Raw Time Series", height=400)
    st.plotly_chart(fig, use_container_width=True)

# 2) DATA PREPROCESSING
st.header("2️⃣ Data Preprocessing")

if st.session_state.data is not None:
    with st.expander("Transformation & Stationarity", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            # Scaling
            scaler_type = st.radio("Scaling Method", ["MinMaxScaler", "StandardScaler"])
            apply_diff = st.checkbox("Apply Differencing", value=False)
            
        with col2:
            # Train-test split
            test_size = st.slider("Test Size (%)", 10, 40, 20) / 100
            lookback = st.slider("Lookback Window (timesteps)", 7, 365, 30)
            horizon = st.slider("Forecast Horizon", 1, 30, 1)

        if st.button("Prepare Data"):
            with st.spinner("Processing data..."):
                # Apply differencing if selected
                processed_series = series.diff().dropna() if apply_diff else series
                
                # Scale
                scaler = MinMaxScaler() if scaler_type == "MinMaxScaler" else StandardScaler()
                scaled_data = scaler.fit_transform(processed_series.values.reshape(-1, 1)).flatten()
                
                # Create sequences
                def create_sequences(data, lookback, horizon):
                    X, y = [], []
                    for i in range(len(data) - lookback - horizon + 1):
                        X.append(data[i:i+lookback])
                        y.append(data[i+lookback:i+lookback+horizon])
                    return np.array(X), np.array(y)
                
                X, y = create_sequences(scaled_data, lookback, horizon)
                
                # Split
                split_idx = int(len(X) * (1 - test_size))
                st.session_state.X_train = X[:split_idx]
                y_train = y[:split_idx]
                st.session_state.X_test = X[split_idx:]
                st.session_state.y_test = y[split_idx:]
                st.session_state.scaler = scaler
                st.session_state.lookback = lookback
                st.session_state.horizon = horizon
                
                # Reshape for LSTM: (samples, timesteps, features)
                st.session_state.X_train = st.session_state.X_train.reshape(-1, lookback, 1)
                st.session_state.X_test = st.session_state.X_test.reshape(-1, lookback, 1)
                
                st.success(f"✅ Data prepared! Train: {st.session_state.X_train.shape}, Test: {st.session_state.X_test.shape}")
                
                # Stationarity test
                adf_result = adfuller(processed_series)
                st.info(f"**ADF Test**: p-value = {adf_result[1]:.4f} ({'Stationary' if adf_result[1] < 0.05 else 'Non-stationary'})")

# 3) GRID SEARCH
st.header("3️⃣ Hyperparameter Grid Search")

if st.session_state.X_train is not None:
    with st.form("grid_search_form"):
        st.markdown("Define hyperparameter ranges:")
        col1, col2 = st.columns(2)
        
        with col1:
            layers_range = st.slider("LSTM Layers", 1, 4, (1, 3))
            units_range = st.slider("Units per Layer", 32, 256, (64, 128), step=32)
            dropout_range = st.slider("Dropout Rate", 0.0, 0.5, (0.1, 0.3), step=0.1)
        
        with col2:
            epochs = st.slider("Epochs", 10, 100, 50)
            batch_size = st.selectbox("Batch Size", [16, 32, 64, 128])
            cv_folds = st.slider("Cross-Validation Folds", 3, 5, 3)
        
        submit_search = st.form_submit_button("🔍 Run Grid Search", use_container_width=True)

    if submit_search:
        with st.status("Running grid search...") as status:
            # Generate parameter grid
            param_grid = []
            for l in range(layers_range[0], layers_range[1] + 1):
                for u in range(units_range[0], units_range[1] + 1, 32):
                    for d in np.arange(dropout_range[0], dropout_range[1] + 0.1, 0.1):
                        param_grid.append({
                            "layers": int(l),
                            "units": int(u),
                            "dropout": round(d, 1)
                        })
            
            # Cross-validation
            tscv = TimeSeriesSplit(n_splits=cv_folds)
            results = []
            
            progress_bar = st.progress(0)
            
            for i, params in enumerate(param_grid[:10]):  # Limit for demo
                status.update(f"Testing model {i+1}/{len(param_grid)}: {params}")
                
                val_losses = []
                for train_idx, val_idx in tscv.split(st.session_state.X_train):
                    X_tr, X_val = st.session_state.X_train[train_idx], st.session_state.X_train[val_idx]
                    y_tr, y_val = y_train[train_idx], y_train[val_idx]
                    
                    model = Sequential([
                        LSTM(params["units"], return_sequences=True, input_shape=(lookback, 1)),
                        Dropout(params["dropout"]),
                        LSTM(params["units"]),
                        Dropout(params["dropout"]),
                        Dense(horizon)
                    ])
                    model.compile(optimizer='adam', loss='mse')
                    
                    history = model.fit(
                        X_tr, y_tr,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(X_val, y_val),
                        verbose=0,
                        callbacks=[
                            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
                        ]
                    )
                    val_losses.append(min(history.history['val_loss']))
                
                results.append({
                    **params,
                    "avg_val_loss": np.mean(val_losses),
                    "std_val_loss": np.std(val_losses)
                })
                progress_bar.progress((i + 1) / len(param_grid))
            
            # Sort by validation loss
            st.session_state.search_results = sorted(results, key=lambda x: x['avg_val_loss'])[:10]
            status.update("✅ Grid search complete!")
        
        # Display results
        st.subheader("🏆 Top 10 Models")
        results_df = pd.DataFrame(st.session_state.search_results)
        st.dataframe(results_df.style.highlight_min(subset=['avg_val_loss']), use_container_width=True)

# 4) FINAL MODEL
st.header("4️⃣ Final Model Training & Prediction")

if st.session_state.search_results is not None:
    # Auto-select best parameters
    best_params = st.session_state.search_results[0]
    st.info(f"**Best Parameters**: Layers={best_params['layers']}, Units={best_params['units']}, Dropout={best_params['dropout']}")
    
    if st.button("🚀 Train Final Model", use_container_width=True, type="primary"):
        with st.spinner("Training final model..."):
            # Build model
            model = Sequential()
            for i in range(best_params['layers']):
                return_seq = i < best_params['layers'] - 1
                model.add(LSTM(
                    best_params['units'],
                    return_sequences=return_seq,
                    input_shape=(lookback, 1) if i == 0 else None
                ))
                model.add(Dropout(best_params['dropout']))
            model.add(Dense(horizon))
            
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            
            # Progress bar for epochs
            progress_bar = st.progress(0)
            epoch_text = st.empty()
            
            class ProgressCallback(tf.keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    progress_bar.progress((epoch + 1) / epochs)
                    epoch_text.text(f"Epoch {epoch + 1}/{epochs} - Loss: {logs['loss']:.4f}")
            
            # Train
            history = model.fit(
                st.session_state.X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.2,
                callbacks=[
                    ProgressCallback(),
                    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
                ],
                verbose=0
            )
            
            st.session_state.model = model
            st.session_state.history = history
            
            # Predictions
            train_pred = model.predict(st.session_state.X_train, verbose=0)
            test_pred = model.predict(st.session_state.X_test, verbose=0)
            
            # Inverse transform
            def inverse_transform_preds(preds):
                # Reshape to 2D for scaler
                preds_2d = preds.reshape(-1, 1)
                inv = st.session_state.scaler.inverse_transform(preds_2d)
                return inv.flatten()
            
            st.session_state.train_pred = inverse_transform_preds(train_pred)
            st.session_state.test_pred = inverse_transform_preds(test_pred)
            st.session_state.y_train_actual = st.session_state.scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
            st.session_state.y_test_actual = st.session_state.scaler.inverse_transform(st.session_state.y_test.reshape(-1, 1)).flatten()
            
            st.success("✅ Model trained and predictions generated!")

# VISUALIZATION
if st.session_state.model is not None:
    st.header("📊 Results Visualization")
    
    # Plot results
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=("Time Series Prediction", "Training Loss", "Residuals"),
        vertical_spacing=0.08,
        row_heights=[0.5, 0.25, 0.25]
    )
    
    # Plot predictions
    train_idx = df.index[:len(st.session_state.train_pred)]
    test_idx = df.index[-len(st.session_state.test_pred):]
    
    fig.add_trace(go.Scatter(x=train_idx, y=st.session_state.y_train_actual, name="Train Actual", line=dict(color="blue")), row=1, col=1)
    fig.add_trace(go.Scatter(x=train_idx, y=st.session_state.train_pred, name="Train Predicted", line=dict(color="orange", dash='dot')), row=1, col=1)
    fig.add_trace(go.Scatter(x=test_idx, y=st.session_state.y_test_actual, name="Test Actual", line=dict(color="green")), row=1, col=1)
    fig.add_trace(go.Scatter(x=test_idx, y=st.session_state.test_pred, name="Test Predicted", line=dict(color="red", dash='dot')), row=1, col=1)
    
    # Plot loss
    fig.add_trace(go.Scatter(x=list(range(len(st.session_state.history.history['loss']))), y=st.session_state.history.history['loss'], name="Train Loss", line=dict(color="blue")), row=2, col=1)
    fig.add_trace(go.Scatter(x=list(range(len(st.session_state.history.history['val_loss']))), y=st.session_state.history.history['val_loss'], name="Val Loss", line=dict(color="red")), row=2, col=1)
    
    # Plot residuals
    train_resid = st.session_state.y_train_actual - st.session_state.train_pred
    test_resid = st.session_state.y_test_actual - st.session_state.test_pred
    fig.add_trace(go.Scatter(x=train_idx, y=train_resid, mode='markers', name="Train Residuals", marker=dict(color="blue", size=3)), row=3, col=1)
    fig.add_trace(go.Scatter(x=test_idx, y=test_resid, mode='markers', name="Test Residuals", marker=dict(color="red", size=3)), row=3, col=1)
    
    fig.update_layout(height=900, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)
    
    # Metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    train_rmse = np.sqrt(mean_squared_error(st.session_state.y_train_actual, st.session_state.train_pred))
    test_rmse = np.sqrt(mean_squared_error(st.session_state.y_test_actual, st.session_state.test_pred))
    train_mae = mean_absolute_error(st.session_state.y_train_actual, st.session_state.train_pred)
    test_mae = mean_absolute_error(st.session_state.y_test_actual, st.session_state.test_pred)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Train RMSE", f"{train_rmse:.2f}")
    col2.metric("Test RMSE", f"{test_rmse:.2f}")
    col3.metric("Train MAE", f"{train_mae:.2f}")
    col4.metric("Test MAE", f"{test_mae:.2f}")
    
    # Download model
    model_bytes = io.BytesIO()
    st.session_state.model.save(model_bytes)
    model_bytes.seek(0)
    b64 = base64.b64encode(model_bytes.read()).decode()
    href = f'<a href="data:file/h5;base64,{b64}" download="lstm_model.h5">📥 Download Trained Model</a>'
    st.markdown(href, unsafe_allow_html=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.info("Built with ❤️ using Streamlit")
