import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# --- KILL SWITCH: If any step fails, show error and STOP ---
def safe_run(func, error_msg):
    try:
        return func()
    except Exception as e:
        st.error(f"🔴 {error_msg}")
        st.code(str(e))  # Show the exact error
        st.stop()  # Stop the app here

st.set_page_config(page_title="LSTM App", layout="wide")
st.title("🔧 LSTM Time Series - DEFENSIVE MODE")

# --- SECTION 1: DATA LOADING ---
st.header("1️⃣ Load Data")
st.info("This step loads data. If it fails, your CSV or network is the problem.")

@st.cache_data(ttl=3600)
def load_sample_data():
    # Try multiple sources in case one fails
    sources = [
        "https://raw.githubusercontent.com/plotly/datasets/master/timeseries.csv",
        "https://raw.githubusercontent.com/AileenNielsen/TimeSeriesAnalysisWithPython/master/data/AirPassengers.csv"
    ]
    for url in sources:
        try:
            df = pd.read_csv(url, index_col=0, parse_dates=True)
            if not df.empty:
                return df
        except:
            continue
    # Ultimate fallback: create fake data
    st.warning("⚠️ All URLs failed. Using synthetic data.")
    dates = pd.date_range('2020-01-01', periods=200)
    return pd.DataFrame({'value': np.cumsum(np.random.randn(200)) + 100}, index=dates)

if st.button("📥 LOAD DATA (Click me first)"):
    st.session_state.data = safe_run(load_sample_data, "Data loading failed")
    st.success("✅ Data loaded successfully!")

# --- SECTION 2: PREPROCESSING ---
if 'data' in st.session_state and st.session_state.data is not None:
    st.header("2️⃣ Preprocess Data")
    st.info("If this fails, your data format is wrong.")
    
    df = st.session_state.data
    target_col = st.selectbox("Select target column", df.columns.tolist())
    series = df[target_col].dropna()
    
    # Parameters
    lookback = st.slider("Lookback window", 5, 50, 10)
    epochs = st.slider("Epochs", 5, 30, 10)  # Reduced for speed
    
    if st.button("🧹 PREPARE DATA"):
        def prep_data():
            # Scale
            scaler = safe_run(lambda: __import__('sklearn.preprocessing', fromlist=['MinMaxScaler']).MinMaxScaler(), 
                              "Failed to import sklearn")
            scaled = safe_run(lambda: scaler.fit_transform(series.values.reshape(-1, 1)).flatten(),
                              "Scaling failed")
            
            # Create sequences
            def create_seq(data, lb):
                X, y = [], []
                for i in range(len(data) - lb):
                    X.append(data[i:i+lb])
                    y.append(data[i+lb])
                return np.array(X), np.array(y)
            
            X, y = safe_run(lambda: create_seq(scaled, lookback), "Sequence creation failed")
            
            # Store everything
            st.session_state.scaler = scaler
            st.session_state.X = X.reshape(-1, lookback, 1)
            st.session_state.y = y
            st.session_state.lookback = lookback
            st.session_state.epochs = epochs
            return "Success"
        
        safe_run(prep_data, "Data preparation failed")
        st.success("✅ Data prepared!")

# --- SECTION 3: TENSORFLOW (Most likely failure point) ---
if 'X' in st.session_state:
    st.header("3️⃣ Model Training")
    st.warning("🐢 TensorFlow is SLOW. This may take 1-2 minutes.")
    
    # Memory management
    try:
        import tensorflow as tf
        tf.config.experimental.set_memory_growth(
            tf.config.experimental.list_physical_devices('GPU')[0], True
        ) if tf.config.experimental.list_physical_devices('GPU') else None
    except:
        pass
    
    if st.button("🚀 TRAIN MODEL (Be patient)"):
        with st.spinner("Training in progress... DO NOT REFRESH"):
            def train_model():
                # Lazy import to fail fast if TensorFlow is missing
                tf = __import__('tensorflow')
                from tensorflow.keras.models import Sequential
                from tensorflow.keras.layers import LSTM, Dense, Dropout
                
                model = Sequential([
                    LSTM(32, activation='relu', input_shape=(lookback, 1)),
                    Dropout(0.2),
                    Dense(1)
                ])
                model.compile(optimizer='adam', loss='mse')
                
                # Train with progress
                progress = st.progress(0)
                for i in range(epochs):
                    model.fit(st.session_state.X, st.session_state.y, 
                             epochs=1, batch_size=16, verbose=0)
                    progress.progress((i + 1) / epochs)
                
                st.session_state.model = model
                return "Training complete"
            
            safe_run(train_model, "TENSORFLOW FAILED. See error below.")
            st.success("✅ Model trained!")

# --- SECTION 4: RESULTS ---
if 'model' in st.session_state:
    st.header("4️⃣ Results")
    
    def get_results():
        model = st.session_state.model
        predictions = model.predict(st.session_state.X, verbose=0)
        
        # Inverse scale
        pred_actual = st.session_state.scaler.inverse_transform(predictions).flatten()
        y_actual = st.session_state.scaler.inverse_transform(st.session_state.y.reshape(-1, 1)).flatten()
        
        # Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=series.index[lookback:], y=y_actual, name="Actual", line=dict(color="blue")))
        fig.add_trace(go.Scatter(x=series.index[lookback:], y=pred_actual, name="Predicted", line=dict(color="red", dash='dot')))
        st.plotly_chart(fig, use_container_width=True)
        
        # Metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        rmse = np.sqrt(mean_squared_error(y_actual, pred_actual))
        mae = mean_absolute_error(y_actual, pred_actual)
        st.metric("RMSE", f"{rmse:.2f}")
        st.metric("MAE", f"{mae:.2f}")
    
    safe_run(get_results, "Result generation failed")

# --- DEBUGGING INFO ---
with st.expander("🐛 Click here if you still see 'Error running app'"):
    st.info("Copy the error message below and paste it in your reply to me.")
    st.code(st.session_state)
