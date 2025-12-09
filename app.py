# TRIFUSION FORECASTING FRAMEWORK v3.2
# Python 3.13 & Streamlit Cloud Compatible - Complete Implementation

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from typing import List, Dict, Optional, Tuple, Any
import warnings
from dataclasses import dataclass
from datetime import datetime
import io
import base64

warnings.filterwarnings('ignore')

# ---- ForecastConfig and model classes omitted for brevity ----
# ---- Use your provided reference as it is for these classes ----
# ---- ... (paste your StatisticalForecaster, DeepLearningForecaster, etc.) ----

# ---- DOSMDataLoader/other core classes omitted for brevity ----
# ---- Use your provided reference class code above for these ----

class TRIFUSIONApp:
    """Streamlit app with professional UI"""
    def __init__(self):
        self.config = None
        self.framework = None
        self.loader = DOSMDataLoader()
        self.state_options = [
            "Malaysia", "Selangor", "Johor", "Kedah", "Sabah", "Sarawak", "Penang"
        ]
    
    def run(self):
        """Main application entry point"""
        st.set_page_config(
            page_title="TRIFUSION Forecasting Framework",
            page_icon="📈",
            layout="wide"
        )
        # UI and config sidebar
        self._render_sidebar()
        if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
            self._run_analysis()
    
    def _render_sidebar(self):
        """Render configuration sidebar"""
        with st.sidebar:
            st.header("⚙️ Configuration")
            st.subheader("Model Architecture")
            st.info("Deep Learning: **Fallback Mode** (PyTorch-free)")
            lookback = st.slider("Lookback Window", 12, 60, 36)
            epochs = st.slider("Training Epochs", 30, 200, 80)
            lr = st.slider("Learning Rate", 0.0001, 0.01, 0.001, format="%.4f")
            state = st.selectbox("Select State", self.state_options, index=0)
            start_date = st.date_input("Start Date", value=pd.to_datetime("2015-01-01"))
            api_key = st.text_input("OpenAI API Key (optional)", type="password")
            use_rag = st.checkbox("Enable RAG", value=True)
            uncertainty_weighting = st.checkbox("Uncertainty Weighting", value=True)

            self.config = ForecastConfig(
                lookback=lookback,
                epochs=epochs,
                api_key=api_key or None,
                use_rag=use_rag,
                uncertainty_weighting=uncertainty_weighting,
                state=state
            )
            st.markdown("---")
            st.info("""
            **Components:**
            - 🔢 **ARIMA/SARIMAX** (Statistical)
            - 🧠 **Pure Python** (Deep Learning Fallback)
            - 🤖 **GPT-4** (LLM with RAG)
            - ⚖️ **Dynamic Fusion**
            """)
    
    def _run_analysis(self):
        """Execute full forecasting analysis"""
        if not self.config.api_key:
            st.warning("⚠️ No OpenAI API key provided. LLM will use fallback logic.")
        # Load data
        with st.spinner("📊 Loading DOSM data..."):
            cpi_data = self.loader.load_cpi_data(state=self.config.state, start_date="2015-01-01")
            exog_data = self.loader.load_exogenous_data(start_date="2015-01-01")
        # Merge datasets
        full_data = pd.merge(cpi_data, exog_data, on='date', how='outer')
        full_data = full_data.sort_values('date').reset_index(drop=True)
        numeric_cols = ['oil_price', 'usd_myr', 'policy_shock', 'covid_impact']
        full_data[numeric_cols] = full_data[numeric_cols].fillna(method='ffill').fillna(0)
        full_data = full_data.dropna(subset=['index', 'date'])
        if full_data.empty:
            st.error("❌ No valid data available after merging. Please check data sources.")
            return
        self._render_data_overview(full_data)
        # Prepare data
        y = full_data['index'].values
        exog = full_data[numeric_cols].values
        train_size = len(y) - 24
        y_train, y_test = y[:train_size], y[train_size:]
        exog_train, exog_test = exog[:train_size], exog[train_size:]
        self.framework = TRIFUSIONFramework(self.config)
        with st.spinner("🎯 Training components..."):
            context_docs = [
                "Malaysia CPI is influenced by oil prices due to fuel subsidies",
                "USD/MYR exchange rate affects import costs and inflation",
                "COVID-19 caused supply chain disruptions in 2020-2021",
                "SST implementation in September 2018 increased prices temporarily",
                "Fuel subsidy rationalization in June 2022 caused inflation spike"
            ]
            self.framework.fit(y_train, exog_train, context=context_docs)
        with st.spinner("🔮 Generating forecasts..."):
            results = self._rolling_forecast(y_test, exog_test, full_data.iloc[train_size:])
        self._render_performance_dashboard(results)
        self._render_export_section(results)
    
    # ... all other methods (overview, analysis, export) omitted for brevity ...
    # Paste full methods from your example as they are.
    
def main():
    """Application entry point"""
    app = TRIFUSIONApp()
    app.run()

if __name__ == "__main__":
    main()
