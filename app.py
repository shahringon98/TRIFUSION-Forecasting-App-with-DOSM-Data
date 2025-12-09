import streamlit as st
import numpy as np
import pandas as pd
import json
from typing import List, Dict, Optional, Tuple, Any
import warnings
from dataclasses import dataclass
from datetime import datetime
import base64

warnings.filterwarnings('ignore')

# ================= FORECASTCONFIG ============================
@dataclass
class ForecastConfig:
    statistical_order: Tuple[int, int, int] = (1, 1, 1)
    lookback: int = 36
    epochs: int = 80
    llm_model: str = "gpt-3.5-turbo"
    api_key: Optional[str] = None
    max_tokens: int = 300
    temperature: float = 0.2
    use_rag: bool = True
    rag_top_k: int = 10
    rag_hybrid_weight: float = 0.5
    window_size: int = 50
    alpha: float = 2.5
    guardrail_threshold: float = 0.4
    uncertainty_weighting: bool = True
    state: Optional[str] = None

# ===================== SYNTHETIC DATA ===============================

def generate_synthetic_cpi(state: str, start_date: str) -> pd.DataFrame:
    dates = pd.date_range(start=start_date, end=pd.Timestamp.now(), freq='M')
    base_cpi = 100
    values = []
    for i, date in enumerate(dates):
        trend = 1 + (date.year - 2015) * 0.015 + i * 0.0003
        seasonal = 1 + 0.02 * np.sin(2 * np.pi * date.month / 12)
        breaks = 1.0
        if pd.Timestamp('2018-09-01') <= date <= pd.Timestamp('2018-12-01'):
            breaks *= 1.02
        if pd.Timestamp('2020-03-01') <= date <= pd.Timestamp('2021-06-01'):
            breaks *= 1.04
        if pd.Timestamp('2022-06-01') <= date <= pd.Timestamp('2022-09-01'):
            breaks *= 1.03
        noise = np.random.normal(0, 0.25)
        cpi = base_cpi * trend * seasonal * breaks + noise
        values.append(max(95, min(135, cpi)))
    return pd.DataFrame({'date': dates, 'state': state, 'index': values})

def generate_synthetic_exogenous(start_date: str) -> pd.DataFrame:
    dates = pd.date_range(start=start_date, end=pd.Timestamp.now(), freq='M')
    data = []
    for date in dates:
        oil_trend = 60 + (date.year - 2020) * 2
        oil_cycle = 15 * np.sin(2 * np.pi * date.year / 3)
        oil_price = max(40, min(120, oil_trend + oil_cycle + np.random.normal(0, 5)))
        usd_myr_trend = 4.2 + 0.1 * np.sin(2 * np.pi * date.year / 5)
        usd_myr = max(3.8, min(4.8, usd_myr_trend + np.random.normal(0, 0.08)))
        policy_shock = 1.0 if date in [pd.Timestamp('2018-09-01'), pd.Timestamp('2022-06-01')] else 0.0
        if pd.Timestamp('2020-03-01') <= date <= pd.Timestamp('2021-03-01'):
            covid_impact = 1.5
        elif pd.Timestamp('2021-04-01') <= date <= pd.Timestamp('2021-12-01'):
            covid_impact = 1.2
        else:
            covid_impact = 0.0
        data.append({
            'date': date,
            'oil_price': oil_price,
            'usd_myr': usd_myr,
            'policy_shock': policy_shock,
            'covid_impact': covid_impact
        })
    return pd.DataFrame(data)

# ===================== SIMPLE FORECASTER ===============================

class StatisticalForecaster:
    def __init__(self, config: ForecastConfig):
        self.config = config
        self.model_fit = None
        self.history = None

    def fit(self, y: np.ndarray):
        self.history = y.copy()
        st.success(f"✅ Statistical model fitted (synthetic demo)")
        return self

    def predict(self, steps: int = 1) -> np.ndarray:
        if self.history is None or len(self.history) == 0:
            return np.zeros(steps)
        trend = (self.history[-1] - self.history[-min(12, len(self.history))]) / min(12, len(self.history))
        last_value = self.history[-1]
        return last_value + np.arange(1, steps + 1) * trend * 0.5

# ====================== APP ================================

class TRIFUSIONApp:
    def __init__(self):
        self.config = None
        self.state_options = ["Malaysia", "Selangor", "Johor", "Kedah", "Sabah", "Sarawak", "Penang"]

    def run(self):
        st.set_page_config(page_title="TRIFUSION Synthetic CPI Forecast", page_icon="📈", layout="wide")
        st.markdown("""
        <div style="text-align:center;">
            <h1>📈 TRIFUSION Synthetic CPI Forecast</h1>
            <p>Simulated hybrid time series forecasting (no external data required)</p>
        </div>
        """, unsafe_allow_html=True)
        self._render_sidebar()
        if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
            self._run_analysis()

    def _render_sidebar(self):
        with st.sidebar:
            st.header("⚙️ Configuration")
            lookback = st.slider("Lookback Window", 12, 60, 36)
            epochs = st.slider("Training Epochs", 30, 200, 80)
            state = st.selectbox("Select State", self.state_options, index=0)
            start_date = st.date_input("Start Date", value=pd.to_datetime("2015-01-01"))
            self.config = ForecastConfig(
                lookback=lookback,
                epochs=epochs,
                state=state
            )

    def _run_analysis(self):
        # --- Data loading ---
        with st.spinner("🔄 Generating synthetic CPI data..."):
            cpi_data = generate_synthetic_cpi(self.config.state, str(self.config.lookback) + "-01-01")
            st.write("CPI data sample:", cpi_data.head())

        with st.spinner("🔄 Generating synthetic exogenous data..."):
            exog_data = generate_synthetic_exogenous("2015-01-01")
            st.write("Exogenous data sample:", exog_data.head())

        cpi_data['date'] = pd.to_datetime(cpi_data['date'])
        exog_data['date'] = pd.to_datetime(exog_data['date'])

        full_data = pd.merge(cpi_data, exog_data, on='date', how='outer')
        full_data = full_data.sort_values('date').reset_index(drop=True)
        st.write("Merged data preview:", full_data.head())

        numeric_cols = ['oil_price', 'usd_myr', 'policy_shock', 'covid_impact']
        for col in numeric_cols:
            if col in full_data.columns:
                full_data[col] = full_data[col].fillna(method='ffill').fillna(0)

        full_data = full_data.dropna(subset=['index', 'date'])
        st.write("Rows after cleaning:", len(full_data))
        st.write("Clean data preview:", full_data.head())

        if full_data.empty:
            st.error("❌ No valid data available after merging. Synthetic fallback failed.")
            return

        y = full_data['index'].values
        train_size = len(y) - 24
        y_train, y_test = y[:train_size], y[train_size:]

        st.markdown("### CPI Training Data")
        st.line_chart(y_train)
        st.markdown("### CPI Testing Data")
        st.line_chart(y_test)

        # --- Forecaster ---
        forecaster = StatisticalForecaster(self.config)
        forecaster.fit(y_train)
        forecast = forecaster.predict(len(y_test))

        st.markdown("### Forecast Results")
        st.line_chart({'Actual': y_test, 'Forecast': forecast})

        # --- Basic Export ---
        result_df = pd.DataFrame({
            'date': full_data['date'].iloc[train_size:],
            'actual_cpi': y_test,
            'forecast': forecast
        })
        csv = result_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Forecast Results as CSV",
            data=csv,
            file_name=f"trifusion_synthetic_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

def main():
    app = TRIFUSIONApp()
    app.run()

if __name__ == "__main__":
    main()
