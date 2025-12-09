import streamlit as st
import numpy as np
import pandas as pd
import json
import requests
from typing import List, Dict, Optional, Tuple, Any
import warnings
from dataclasses import dataclass
from datetime import datetime
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

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

    def validate(self):
        assert self.lookback > 0, "Lookback must be positive"
        assert 0 <= self.temperature <= 2, "Temperature must be in [0, 2]"

# ===================== ANSI MODEL CLASSES ===============================
class StatisticalForecaster:
    def __init__(self, config: ForecastConfig):
        self.config = config
        self.model_fit = None
        self.history = None

    def fit(self, y: np.ndarray, exog: Optional[np.ndarray] = None):
        try:
            from statsmodels.tsa.arima.model import ARIMA
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            self.history = y.copy()
            if exog is not None:
                self.model_fit = SARIMAX(y, exog=exog, order=self.config.statistical_order).fit(disp=False)
            else:
                self.model_fit = ARIMA(y, order=self.config.statistical_order).fit(disp=False)
            st.success(f"✅ Statistical model fitted (AIC: {self.model_fit.aic:.2f})")
        except Exception as e:
            st.error(f"Statistical model failed: {str(e)}")
            self.model_fit = None
        return self

    def predict(self, steps: int = 1, exog_future: Optional[np.ndarray] = None) -> np.ndarray:
        if self.model_fit is None:
            return np.full(steps, self.history[-1] if self.history is not None else 0)
        try:
            forecast = self.model_fit.get_forecast(steps=steps, exog=exog_future)
            return forecast.predicted_mean
        except:
            return np.full(steps, self.history[-1])

    def get_confidence_interval(self) -> Optional[np.ndarray]:
        if hasattr(self, 'model_fit') and self.model_fit is not None:
            try:
                return self.model_fit.get_forecast(steps=1).conf_int()
            except:
                pass
        return None

# LSTM Helper
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

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

class DeepLearningForecaster:
    def __init__(self, config: ForecastConfig):
        self.config = config
        self.model = None
        self.scaler = None
        self.history = None
        self.seq_length = max(4, self.config.lookback // 3)  # Minimum 4 for quarterly-like

    def fit(self, y: np.ndarray, exog: Optional[np.ndarray] = None):
        self.history = y.copy()
        if len(y) < self.seq_length + 1:
            st.warning("Data too short for DL; using fallback")
            return self
        self.scaler = MinMaxScaler()
        y_scaled = self.scaler.fit_transform(y.reshape(-1, 1))
        X, ys = create_sequences(y_scaled, self.seq_length)
        if len(X) == 0:
            return self
        X = torch.from_numpy(X).float().unsqueeze(2)  # (samples, seq, 1)
        ys = torch.from_numpy(ys).float().unsqueeze(1)
        self.model = LSTMForecaster(input_size=1)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        for epoch in range(self.config.epochs):
            outputs = self.model(X)
            loss = criterion(outputs, ys)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        st.success("✅ Deep learning model fitted")
        return self

    def predict(self, steps: int = 1, uncertainty_quantification: bool = False) -> np.ndarray:
        if self.model is None or self.scaler is None:
            # Fallback if not fitted properly
            if len(self.history) == 0:
                return np.full(steps, 0)
            last_value = self.history[-1]
            trend = 0
            if len(self.history) >= 10:
                trend = np.mean(self.history[-5:]) - np.mean(self.history[-10:-5])
            forecasts = last_value + np.arange(1, steps + 1) * trend * 0.5
            if uncertainty_quantification:
                noise = np.random.normal(0, 0.01 * np.std(self.history) + 1e-8, steps)
                forecasts += noise
            return np.maximum(forecasts, last_value * 0.9)
        # LSTM predict
        y_scaled = self.scaler.transform(self.history.reshape(-1, 1))
        inputs = torch.from_numpy(y_scaled[-self.seq_length:]).float().unsqueeze(0).unsqueeze(2)
        forecasts = []
        self.model.eval()
        for _ in range(steps):
            pred = self.model(inputs)
            forecasts.append(pred.item())
            inputs = torch.cat((inputs[:, 1:, :], pred.unsqueeze(0).unsqueeze(2)), dim=1)
        return self.scaler.inverse_transform(np.array(forecasts).reshape(-1, 1)).flatten()

class RAGPipeline:
    def __init__(self, config: ForecastConfig):
        self.config = config
        self.corpus = []
        self.vectorizer = None
        self.tfidf_matrix = None
        self.cache = {}

    def build_corpus(self, corpus: List[str]):
        if not corpus:
            st.warning("Empty corpus provided")
            return
        self.corpus = corpus
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        st.success(f"✅ RAG corpus loaded ({len(corpus)} documents)")

    def retrieve(self, query: str, top_k: Optional[int] = None) -> Tuple[List[str], List[float], List[bool]]:
        if self.vectorizer is None:
            return [], [], []
        cache_key = f"query:{query}:k:{top_k}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        top_k = top_k or self.config.rag_top_k
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = np.argsort(scores)[-top_k:][::-1]
        retrieved_docs = [self.corpus[i] for i in top_indices if i < len(self.corpus)]
        retrieved_scores = [float(scores[i]) for i in top_indices if i < len(scores)]
        is_reliable = [s > 0.3 for s in retrieved_scores]
        self.cache[cache_key] = (retrieved_docs, retrieved_scores, is_reliable)
        return retrieved_docs, retrieved_scores, is_reliable

class LLMForecaster:
    def __init__(self, config: ForecastConfig, rag_pipeline: Optional[RAGPipeline] = None):
        self.config = config
        self.rag = rag_pipeline
        self.history = None

    def predict(self, y: np.ndarray, steps: int = 1, context: Optional[str] = None) -> Tuple[np.ndarray, str, float]:
        self.history = y
        retrieved_docs, scores, reliability = [], [], []
        if self.config.use_rag and self.rag and context:
            retrieved_docs, scores, reliability = self.rag.retrieve(context)
        prompt = f"""You are an expert economic forecaster.
       
TASK: Forecast the next {steps} values for the time series: {y[-10:].tolist()}
OUTPUT FORMAT (JSON ONLY):
{{"forecast": [number1, number2, ...], "reasoning": "Your analysis here", "confidence": 0.0-1.0}}
CONTEXT: {context or "No additional context provided"}
RELEVANT KNOWLEDGE: {' | '.join(retrieved_docs[:3]) if retrieved_docs else "None"}
RELIABILITY SCORES: {reliability[:3] if reliability else "N/A"}
GUIDELINES:
- Consider trend, seasonality, and recent anomalies
- Factor in economic context if provided
- Be conservative with confidence for volatile periods"""
        return self._call_llm(prompt, steps)

    def _call_llm(self, prompt: str, steps: int) -> Tuple[np.ndarray, str, float]:
        try:
            import openai
            openai.api_key = self.config.api_key
            response = openai.ChatCompletion.create(
                model=self.config.llm_model,
                messages=[
                    {"role": "system", "content": "You are a forecasting assistant. Output only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            content = response.choices[0].message.content
            content = content.strip().strip('`').replace('```json', '').replace('```', '')
            data = json.loads(content)
            forecast = np.array(data["forecast"][:steps], dtype=float)
            reasoning = data.get("reasoning", "No reasoning provided")
            confidence = float(data.get("confidence", 0.5))
            return forecast, reasoning, confidence
        except Exception as e:
            st.warning(f"LLM unavailable: {str(e)[:100]}")
            return self._fallback_forecast(steps)

    def _fallback_forecast(self, steps: int) -> Tuple[np.ndarray, str, float]:
        if len(self.history) >= 10:
            trend = np.mean(self.history[-5:]) - np.mean(self.history[-10:-5])
        elif len(self.history) >= 5:
            trend = np.mean(self.history[-3:]) - np.mean(self.history[:-3])
        else:
            trend = 0
        last_value = self.history[-1] if len(self.history) > 0 else 100
        forecast = last_value + np.arange(1, steps + 1) * trend * 0.5
        forecast += np.random.normal(0, abs(trend) * 0.1 + 0.01, steps)
        return forecast, "Trend-based fallback (LLM unavailable)", 0.3

class MetaController:
    def __init__(self, config: ForecastConfig):
        self.config = config
        self.loss_history = {'statistical': [], 'deep_learning': [], 'llm': []}
        self.weights_history = []

    def compute_loss(self, y_true: float, y_pred: float, uncertainty: float, model_type: str):
        base_loss = (y_true - y_pred) ** 2
        adjusted_loss = base_loss + self.config.alpha * uncertainty
        self.loss_history[model_type].append(adjusted_loss)
        if len(self.loss_history[model_type]) > self.config.window_size:
            self.loss_history[model_type].pop(0)

    def update_weights(self, uncertainties: Dict[str, float]) -> np.ndarray:
        avg_losses = [np.mean(self.loss_history[t]) if self.loss_history[t] else 0 for t in ['statistical', 'deep_learning', 'llm']]
        for i, model_type in enumerate(['statistical', 'deep_learning', 'llm']):
            if uncertainties and model_type in uncertainties:
                avg_losses[i] += uncertainties[model_type] * 2
        if avg_losses[2] > self.config.guardrail_threshold:
            avg_losses[2] = np.inf
        exp_terms = np.exp(-self.config.alpha * np.array(avg_losses))
        exp_terms = np.where(np.isinf(avg_losses), 0, exp_terms)
        sum_exp = np.sum(exp_terms)
        weights = exp_terms / sum_exp if sum_exp > 0 else np.array([1/3, 1/3, 1/3])
        self.weights_history.append(weights.copy())
        return weights

class TRIFUSIONFramework:
    def __init__(self, config: ForecastConfig):
        self.config = config
        self.statistical = StatisticalForecaster(config)
        self.deep_learning = DeepLearningForecaster(config)
        self.rag = RAGPipeline(config)
        self.llm = LLMForecaster(config, self.rag)
        self.meta_controller = MetaController(config)
        self.history = None
        self.exog_history = None

    def fit(self, y: np.ndarray, exog: Optional[np.ndarray] = None, context: Optional[List[str]] = None):
        self.history = y.copy()
        self.exog_history = exog.copy() if exog is not None else None
        with st.spinner("🔧 Training Statistical Model..."):
            self.statistical.fit(y, exog)
        with st.spinner("🧠 Training Deep Learning Model..."):
            self.deep_learning.fit(y, exog)
        if context and self.config.use_rag:
            with st.spinner("📚 Building RAG Corpus..."):
                self.rag.build_corpus(context)
        st.success("🎯 All models trained!")
        return self

    def predict(self, steps: int = 1, exog_future: Optional[np.ndarray] = None, context: Optional[str] = None) -> Dict[str, Any]:
        pred_stat = self.statistical.predict(steps, exog_future)
        pred_deep = self.deep_learning.predict(steps)
        pred_llm, reasoning, confidence = self.llm.predict(self.history, steps, context)
        min_len = min(len(pred_stat), len(pred_deep), len(pred_llm))
        pred_stat, pred_deep, pred_llm = pred_stat[:min_len], pred_deep[:min_len], pred_llm[:min_len]
        uncertainties = {'statistical': 0.1, 'deep_learning': 0.1, 'llm': 1.0 - confidence}
        weights = self.meta_controller.update_weights(uncertainties)
        hybrid = weights[0] * pred_stat + weights[1] * pred_deep + weights[2] * pred_llm
        return {
            'forecast': hybrid,
            'components': {
                'statistical': pred_stat,
                'deep_learning': pred_deep,
                'llm': pred_llm
            },
            'weights': {
                'statistical': weights[0],
                'deep_learning': weights[1],
                'llm': weights[2]
            },
            'uncertainties': uncertainties,
            'explanation': reasoning,
            'confidence_interval': self.statistical.get_confidence_interval(),
            'overall_confidence': 1.0 - np.dot(weights, list(uncertainties.values()))
        }

    def update_with_new_data(self, y_new: float, exog_new: Optional[np.ndarray] = None):
        if self.history is not None:
            self.history = np.append(self.history, y_new)
            if self.exog_history is not None and exog_new is not None:
                self.exog_history = np.vstack([self.exog_history, exog_new])

# ======= TOP-LEVEL CACHED FUNCTIONS FOR DOSM DATA LOADING ====
@st.cache_data(ttl=3600)
def load_cpi_data_cached(state: str, start_date: str) -> pd.DataFrame:
    api_url = "https://api.data.gov.my/data-catalogue"
    state_map = {"Penang": "pulau pinang"}
    api_state = state_map.get(state, state).lower()
    id_value = "cpi_headline" if state == "Malaysia" else "cpi_state"
    params = {
        "id": id_value,
        "limit": 0,
        "division": "overall"
    }
    if state != "Malaysia":
        params["state"] = api_state
    try:
        response = requests.get(api_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        if not data:
            raise ValueError("Empty response from API")
        df = pd.DataFrame(data)
        if 'date' not in df.columns:
            raise KeyError("'date' column not found in API response")
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])  # Drop rows with invalid dates
        df = df[df['date'] >= pd.Timestamp(start_date)].sort_values('date').reset_index(drop=True)
        if df.empty:
            raise ValueError("No data after filtering")
        if state == "Malaysia":
            df['state'] = "Malaysia"
        st.success(f"✅ Loaded {len(df)} CPI records for {state}")
        return df
    except requests.exceptions.RequestException as e:
        st.error(f"❌ API request failed: {str(e)}")
    except ValueError as e:
        st.error(f"❌ Data processing error: {str(e)}")
    except KeyError as e:
        st.error(f"❌ Missing expected column: {str(e)}")
    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")
    return generate_synthetic_cpi(state, start_date)

@st.cache_data
def load_exogenous_data_cached(start_date: str) -> pd.DataFrame:
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
    return pd.DataFrame({'date': dates, 'state': state, 'cpi': values})  # Use 'cpi' for consistency

# =========== Streamlit App Class (uses top-level cached functions) ==========
class TRIFUSIONApp:
    def __init__(self):
        self.config = None
        self.framework = None
        self.state_options = ["Malaysia", "Selangor", "Johor", "Kedah", "Sabah", "Sarawak", "Penang"]

    def run(self):
        st.set_page_config(page_title="TRIFUSION Forecasting Framework", page_icon="📈", layout="wide")
        st.markdown("""
        <div class="main-header">
            <h1>📈 TRIFUSION Forecasting Framework</h1>
            <p>Advanced Hybrid Time Series Forecasting with LLM Integration</p>
            <p style="color: #eee; font-size: 0.9em;">Powered by DOSM Malaysia Open Data</p>
        </div>
        """, unsafe_allow_html=True)
        self._render_sidebar()
        if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
            self._run_analysis()

    def _render_sidebar(self):
        with st.sidebar:
            st.header("⚙️ Configuration")
            st.info("Deep Learning: **LSTM Enabled**")
            lookback = st.slider("Lookback Window", 12, 60, 36)
            epochs = st.slider("Training Epochs", 30, 200, 80)
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
            - 🧠 **LSTM** (Deep Learning)
            - 🤖 **GPT-4 Fallback** (LLM with RAG)
            - ⚖️ **Dynamic Fusion**
            """)

    def _run_analysis(self):
        if not self.config.api_key:
            st.warning("⚠️ No OpenAI API key provided. LLM will use fallback logic.")
        with st.spinner("📊 Loading DOSM data..."):
            cpi_data = load_cpi_data_cached(state=self.config.state, start_date="2015-01-01")
            exog_data = load_exogenous_data_cached(start_date="2015-01-01")
        full_data = pd.merge(cpi_data, exog_data, on='date', how='outer')
        full_data = full_data.sort_values('date').reset_index(drop=True)
        numeric_cols = ['oil_price', 'usd_myr', 'policy_shock', 'covid_impact']
        full_data[numeric_cols] = full_data[numeric_cols].ffill().fillna(0)
        full_data = full_data.dropna(subset=['cpi', 'date'])  # Use 'cpi'
        if full_data.empty:
            st.error("❌ No valid data available after merging. Please check data sources.")
            return
        y = full_data['cpi'].values
        exog = full_data[numeric_cols].values
        train_size = len(y) - 24 if len(y) > 24 else int(0.8 * len(y))
        y_train, y_test = y[:train_size], y[train_size:]
        exog_train, exog_test = exog[:train_size], exog[train_size:]
        self.framework = TRIFUSIONFramework(self.config)
        context_docs = [
            "Malaysia CPI is influenced by oil prices due to fuel subsidies",
            "USD/MYR exchange rate affects import costs and inflation",
            "COVID-19 caused supply chain disruptions in 2020-2021",
            "SST implementation in September 2018 increased prices temporarily",
            "Fuel subsidy rationalization in June 2022 caused inflation spike"
        ]  # Make configurable if needed
        with st.spinner("🎯 Training components..."):
            self.framework.fit(y_train, exog_train, context=context_docs)
        forecast_steps = 12
        with st.spinner("🔮 Generating Forecast..."):
            result = self.framework.predict(steps=forecast_steps, exog_future=exog_test[:forecast_steps] if len(exog_test) >= forecast_steps else None, context=f"Forecast CPI for {self.config.state}")
        st.subheader("Forecast Results")
        future_dates = pd.date_range(start=full_data['date'].iloc[-1] + pd.DateOffset(months=1), periods=forecast_steps, freq='M')
        forecast_df = pd.DataFrame({
            "Date": future_dates,
            "Hybrid Forecast": result['forecast'],
            "Statistical": result['components']['statistical'],
            "Deep Learning": result['components']['deep_learning'],
            "LLM": result['components']['llm']
        })
        st.dataframe(forecast_df)
        st.subheader("Model Weights")
        st.json(result['weights'])
        st.subheader("Explanation")
        st.write(result['explanation'])
        st.subheader("Forecast Visualization")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(full_data['date'], y, label="Historical CPI")
        ax.plot(forecast_df['Date'], forecast_df['Hybrid Forecast'], label="Hybrid Forecast", color='green')
        ax.plot(forecast_df['Date'], forecast_df['Statistical'], label="Statistical", alpha=0.5)
        ax.plot(forecast_df['Date'], forecast_df['Deep Learning'], label="Deep Learning", alpha=0.5)
        ax.plot(forecast_df['Date'], forecast_df['LLM'], label="LLM", alpha=0.5)
        ax.set_xlabel("Date")
        ax.set_ylabel("CPI Value")
        ax.legend()
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig)
        # Export
        csv = forecast_df.to_csv(index=False)
        st.download_button("📥 Download Forecast CSV", csv, "trifusion_forecast.csv", "text/csv")
        st.success("🎉 Analysis complete!")

def main():
    app = TRIFUSIONApp()
    app.run()

if __name__ == "__main__":
    main()
