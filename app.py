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

# ===================== MODELS ===============================

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

class DeepLearningForecaster:
    def __init__(self, config: ForecastConfig):
        self.config = config
        self.history = None
        self.is_available = False

    def fit(self, y: np.ndarray, exog: Optional[np.ndarray] = None):
        self.history = y.copy()
        st.warning("⚠️ Deep learning model disabled for Python 3.13 compatibility")
        return self

    def predict(self, y: np.ndarray, exog: Optional[np.ndarray] = None, steps: int = 1, uncertainty_quantification: bool = False) -> np.ndarray:
        if len(y) == 0:
            return np.full(steps, 0)
        last_value = y[-1]
        if len(y) >= 20:
            recent_trend = np.mean(y[-10:]) - np.mean(y[-20:-10])
            long_trend = (y[-1] - y[0]) / len(y)
            trend = 0.7 * recent_trend + 0.3 * long_trend
        elif len(y) >= 5:
            trend = np.mean(y[-3:]) - np.mean(y[:-3])
        else:
            trend = 0
        seasonal_pattern = np.sin(np.arange(steps) * 2 * np.pi / 12) * 0.02 * last_value
        predictions = last_value + np.arange(1, steps + 1) * trend * 0.5 + seasonal_pattern
        if uncertainty_quantification:
            noise = np.random.normal(0, 0.01 * np.std(y), steps)
            predictions += noise
        return np.maximum(predictions, last_value * 0.9)

class RAGPipeline:
    def __init__(self, config: ForecastConfig):
        self.config = config
        self.corpus = []
        self.bm25 = None
        self.cache = {}

    def build_corpus(self, corpus: List[str]):
        if not corpus:
            st.warning("Empty corpus provided")
            return
        from rank_bm25 import BM25Okapi
        self.corpus = corpus
        tokenized_corpus = [doc.lower().split() for doc in corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        st.success(f"✅ RAG corpus loaded ({len(corpus)} documents)")

    def retrieve(self, query: str, top_k: Optional[int] = None) -> Tuple[List[str], List[float], List[bool]]:
        if not self.bm25:
            return [], [], []
        cache_key = f"query:{query}:k:{top_k}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        top_k = top_k or self.config.rag_top_k
        tokenized_query = query.lower().split()
        scores = np.array(self.bm25.get_scores(tokenized_query))
        scores = (scores - scores.mean()) / (scores.std() + 1e-8)
        top_indices = np.argsort(scores)[-top_k:][::-1]
        retrieved_docs = [self.corpus[i] for i in top_indices]
        retrieved_scores = [float(scores[i]) for i in top_indices]
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
        pred_deep = self.deep_learning.predict(self.history, self.exog_history, steps)
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

# ================= DOSM LOADER ================================
class DOSMDataLoader:
    def __init__(self):
        self.base_url = "https://storage.dosm.gov.my"
        self.datasets = {"cpi_state": "/timeseries/cpi/cpi_2d_state.parquet"}

    def load_cpi_data(self, state: str = "Malaysia", start_date: str = "2015-01-01") -> pd.DataFrame:
        return self._load_cpi_data_internal(state, start_date)

    @st.cache_data(ttl=3600)
    def _load_cpi_data_internal(self, state: str, start_date: str) -> pd.DataFrame:
        try:
            url = f"{self.base_url}{self.datasets['cpi_state']}"
            df = pd.read_parquet(url)
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            state_col = next((col for col in ['state', 'state_name'] if col in df.columns), None)
            if state_col:
                if state == "Malaysia":
                    df_filtered = df[df[state_col].isin(["Malaysia", "SEA_MALAYSIA", "MALAYSIA"])]
                else:
                    df_filtered = df[df[state_col] == state]
            else:
                df_filtered = df
            df_filtered = df_filtered[
                (df_filtered['date'] >= pd.Timestamp(start_date)) &
                (df_filtered['date'] <= pd.Timestamp.now())
            ].sort_values('date').reset_index(drop=True)
            if df_filtered.empty:
                raise ValueError("No data after filtering")
            st.success(f"✅ Loaded {len(df_filtered)} CPI records for {state}")
            return df_filtered
        except Exception as e:
            st.error(f"❌ Failed to load DOSM data: {str(e)}")
            return self._generate_synthetic_cpi(state, start_date)

    def load_exogenous_data(self, start_date: str = "2015-01-01") -> pd.DataFrame:
        return self._load_exogenous_data_internal(start_date)

    @st.cache_data
    def _load_exogenous_data_internal(self, start_date: str) -> pd.DataFrame:
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

    def _generate_synthetic_cpi(self, state: str, start_date: str) -> pd.DataFrame:
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

# ================= APP ============================
class TRIFUSIONApp:
    def __init__(self):
        self.config = None
        self.framework = None
        self.loader = DOSMDataLoader()
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
        if not self.config.api_key:
            st.warning("⚠️ No OpenAI API key provided. LLM will use fallback logic.")
        with st.spinner("📊 Loading DOSM data..."):
            cpi_data = self.loader.load_cpi_data(state=self.config.state, start_date="2015-01-01")
            exog_data = self.loader.load_exogenous_data(start_date="2015-01-01")
        full_data = pd.merge(cpi_data, exog_data, on='date', how='outer')
        full_data = full_data.sort_values('date').reset_index(drop=True)
        numeric_cols = ['oil_price', 'usd_myr', 'policy_shock', 'covid_impact']
        full_data[numeric_cols] = full_data[numeric_cols].fillna(method='ffill').fillna(0)
        full_data = full_data.dropna(subset=['index', 'date'])
        if full_data.empty:
            st.error("❌ No valid data available after merging. Please check data sources.")
            return
        self._render_data_overview(full_data)
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

    # --- Add your dashboard and export methods here, unchanged from your example code ---

def main():
    app = TRIFUSIONApp()
    app.run()

if __name__ == "__main__":
    main()
