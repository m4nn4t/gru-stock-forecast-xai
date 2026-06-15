# Multi-Stock GRU Forecast Platform

A deep learning platform that predicts stock prices and directional movement (UP/DOWN) for 10 major tech stocks over a 7-day horizon. Built with Bidirectional GRU neural networks, explainable AI (SHAP + LIME), FinBERT sentiment analysis, and GenAI-powered natural language explanations.

---

## Stocks Covered

AAPL · MSFT · GOOGL · AMZN · META · NVDA · TSLA · AMD · NFLX · INTC

---

## Architecture Overview

```
Raw Price Data (Tiingo API)
        ↓
Feature Engineering (13 technical indicators)
        ↓
Per-Stock MinMaxScaler (train-only fit)
        ↓
Sliding Window Sequences (60-day lookback)
        ↓
    ┌───────────────────────────────┐
    │  Regressor (BiGRU)            │  → 7-day price trajectory
    │  Classifier (GRU + Sigmoid)   │  → P(UP) + confidence
    └───────────────────────────────┘
        ↓
SHAP + LIME (feature importance)
        ↓
FinBERT (news sentiment)
        ↓
Groq LLaMA (natural language explanation)
```

---

## Technologies Used

### 1. Data Sources

#### Tiingo API
- **What it does:** Provides split-adjusted historical daily OHLCV stock data (`adjClose`, `adjHigh`, `adjLow`, `adjVolume`).
- **Why adjClose:** AAPL had a 4:1 stock split in August 2020. Using raw `close` creates a massive price discontinuity that corrupts the scaler and confuses the model. `adjClose` accounts for all splits and gives a clean continuous price series.
- **Pros:** Clean adjusted data, reliable 7+ year history, simple REST API.
- **Cons:** Free tier has limited endpoints; no real-time data.

#### Yahoo Finance (yfinance)
- **What it does:** Downloads SPY (S&P 500 ETF) and VIX (fear index) daily data for market context.
- **Why:** Individual stock models perform better when they can see what the broader market is doing. A stock dropping 2% on a day the whole market drops 3% is very different from dropping 2% on a green market day.
- **Pros:** Free, no API key needed, wide coverage.
- **Cons:** Occasional missing data on weekends/holidays; column names can vary by version (MultiIndex issue handled in code).

#### Finnhub API
- **What it does:** Provides recent news headlines per ticker for sentiment analysis.
- **Why used at inference only:** The free tier only has ~3.5% coverage over our 7-year training period. Feeding a near-zero constant feature into the model would actively hurt training. Instead, it is fetched live at inference time to provide qualitative sentiment context.
- **Pros:** Free tier, easy REST API, good recent coverage.
- **Cons:** Very sparse historical data; not suitable as a training feature on the free plan.

---

### 2. Feature Engineering (13 Stationary Features)

All features are **stationary** — they measure changes, ratios, or oscillators rather than raw price levels. This is critical: raw prices are non-stationary (they trend upward over years) and confuse the scaler and model on out-of-sample data.

| Feature | Formula / Description | Why Important |
|---|---|---|
| `RETURN_1D` | `(close_t - close_{t-1}) / close_{t-1}` | Captures daily momentum |
| `RETURN_5D` | `(close_t - close_{t-5}) / close_{t-5}` | Weekly momentum signal |
| `MOMENTUM_20` | `(close_t - close_{t-20}) / close_{t-20}` | Monthly trend direction |
| `EMA_CROSS` | `(EMA10 - EMA30) / close` | Trend crossover signal, normalised |
| `RSI` | 14-day Relative Strength Index | Overbought/oversold oscillator (0–100) |
| `MACD_HIST_PCT` | `(MACD - Signal) / close` | Momentum shift, normalised by price |
| `BB_POS` | `(close - BB_lower) / (BB_upper - BB_lower)` | Position within Bollinger Band (0=bottom, 1=top) |
| `BB_WIDTH` | `(BB_upper - BB_lower) / close` | Volatility measure |
| `ATR_PCT` | `ATR(14) / close` | True range volatility, normalised |
| `VOL_RATIO` | `volume / SMA(volume, 20)` | Volume spike detection |
| `SPY_RETURN` | Daily % return of S&P 500 ETF | Market-wide context |
| `VIX_NORM` | `VIX / 100` | Fear/uncertainty index, normalised |
| `DOW` | `day_of_week / 4` | Day-of-week seasonal effect (0=Mon, 1=Fri) |

**Why stationary features matter:** Previous versions of this project used raw Close prices as features. On test data (dates after training), the prices were outside the scaler's range (prices had risen since training), making all features either 0 or 1. This caused systematic directional inversion and accuracy below 50%.

---

### 3. Target Variable

**7-day cumulative returns:**
```
target[i] = (close[t+i] - close[t]) / close[t] × 100    for i = 1 to 7
```

**Why returns, not prices:**
- Predicting absolute prices causes the model to learn "tomorrow ≈ today" (random walk prior), which leads to lagged predictions and directional inversion.
- Returns are stationary and directly encode the direction: positive = UP, negative = DOWN.
- Returns are comparable across stocks of different price levels (NVDA $132 vs META $634).

**Standardisation:** Targets are z-scored per stock using training-set mean and std before feeding to the model. This prevents the model from collapsing to predicting the mean return (which is near zero and trivially minimises MSE).

---

### 4. Data Leakage Prevention

**The bug:** Earlier notebooks fit `MinMaxScaler` on the entire dataset (train + test) before splitting. This leaks future price statistics into the scaler, artificially inflating performance on the test set.

**The fix:** For each stock, the scaler is fit exclusively on training rows, then applied (transform only) to test rows. This is enforced separately per stock so no cross-stock leakage occurs either.

---

### 5. Model Architecture

#### Regressor — Price Trajectory (BiGRU)
```
Input: (60 days × 13 features)
→ GRU(128, return_sequences=True) + Dropout(0.20)
→ GRU(64, return_sequences=True)  + Dropout(0.15)
→ GRU(32)
→ Dense(32, relu)
→ Dense(7, linear)     ← 7 return values, one per forecast day
```
- **Loss:** Mean Squared Error on standardised returns
- **Optimiser:** Adam (lr=1e-3) with ReduceLROnPlateau
- **Early stopping:** Patience=15 on val_loss

#### Classifier — Direction + Confidence (GRU + Sigmoid)
```
Input: (60 days × 13 features)
→ GRU(128, return_sequences=True) + Dropout(0.20)
→ GRU(64) + Dropout(0.15)
→ Dense(32, relu)
→ Dense(1, sigmoid)    ← P(UP) for Day 1
```
- **Loss:** Binary cross-entropy
- **Output:** Probability of UP movement (0.5 = uncertain, 1.0 = strongly UP)
- **Confidence mapping:** `confidence = 50 + |P(UP) - 0.5| × 90`  → range 50%–95%

**Why two separate models:**
A single regression model trained with MSE always collapses to predicting the mean return (~0%) because that minimises MSE in a noisy environment. Separating regression (for price magnitude) from classification (for direction) allows each model to optimise for its own objective independently.

**Pros of GRU over LSTM:**
- Fewer parameters → faster training
- Comparable performance on financial time series
- Less prone to vanishing gradient on short sequences

**Cons:**
- Still struggles with long-range dependencies beyond 60 days
- Sensitive to hyperparameter choices
- No attention mechanism (Transformer would handle long-range better)

---

### 6. Evaluation Metrics

| Metric | What it Measures | Expected Range |
|---|---|---|
| RMSE | Average price prediction error in dollars | $1–$12 depending on stock price |
| MAE | Average absolute price error | Slightly lower than RMSE |
| MAPE | Error as % of actual price | 1–3% (good for stocks) |
| R² | How well predictions track actual price trend | 0.93–0.99 (our results) |
| Directional Accuracy | % of days where UP/DOWN prediction is correct | 50–60% realistic range |
| Precision | Of all predicted UP days, how many were actually UP | Varies by stock |
| Recall | Of all actual UP days, how many did model catch | High (model biased toward UP) |
| F1 Score | Harmonic mean of Precision and Recall | 40–72% in our results |

**Context for directional accuracy:**
Academic literature consistently shows 52–57% as achievable with technical indicators alone. Our results (50.8%–56.2%) are in line with this. Random baseline is exactly 50%.

---

### 7. Explainable AI (XAI)

#### SHAP (SHapley Additive exPlanations)
- **What it does:** Assigns each feature a contribution score for a specific prediction. Based on cooperative game theory — how much does each feature "contribute" to moving the prediction away from the baseline?
- **Implementation:** `shap.GradientExplainer` — works natively with TensorFlow/Keras by computing gradients. Both bar plots (global importance) and beeswarm plots (per-sample signed importance) are generated for all 10 stocks.
- **Key finding (bar chart):** `VIX_NORM` (fear index) is the most globally important feature on average. When market fear rises, the model predicts downward pressure across all tech stocks.
- **Key finding (beeswarm — per stock):** Feature importance is NOT uniform across stocks. Three distinct patterns emerge:
  - **Large-cap, institutionally traded stocks (MSFT, AMZN, NVDA, NFLX):** `VIX_NORM` dominates — macro fear drives their short-term price moves more than individual technical signals.
  - **High-volatility stocks (TSLA, AMD):** `BB_POS` and `RSI` dominate — where the stock sits within its own Bollinger Band and momentum oscillators matter more than broad market sentiment.
  - **Stocks with high intrinsic volatility (META, INTC):** `ATR_PCT` dominates — the stock's own daily range drives predictions more than external signals.
- **Colour pattern:** Across all beeswarm plots, red dots (high VIX) consistently fall on the negative SHAP side — high market fear pushes all stock predictions downward regardless of stock type. Economically correct.
- **Why this matters:** The stock-specific variation in dominant features confirms that the per-stock model architecture (rather than one shared model) was the correct design choice. A single shared model would average out these differences and lose predictive signal.
- **Pros:** Theoretically grounded, consistent with model internals, fast with GradientExplainer.
- **Cons:** Approximate for deep learning; exact Shapley values are computationally intractable for neural nets.

#### LIME (Local Interpretable Model-agnostic Explanations)
- **What it does:** Explains individual predictions by perturbing the input and fitting a simple linear model around that neighbourhood.
- **Implementation:** `lime.lime_tabular` on flattened input (60 days × 13 features = 780 features). A wrapper function reshapes flat input back to 3D for the GRU. Applied to the most recent 60-day window for all 10 stocks.
- **Key findings (per stock):**
  - **AAPL, MSFT:** DOW (day-of-week) is the top local driver — both large-cap stocks have strong weekly seasonality patterns the model has learned.
  - **GOOGL:** D1 and D2 features (the *oldest* days in the 60-day window, ~2 months ago) dominate — the model uses longer-range context for GOOGL than any other stock, suggesting stronger long-range momentum dependencies.
  - **META:** D53_MOMENTUM_20 is a large negative driver — medium-term momentum from ~7 weeks ago pulls predictions down. Past momentum cycles matter more than recent signals.
  - **TSLA, AMD:** Recent BB_POS and RSI (D59, D60) dominate and are positive — consistent with SHAP beeswarm finding. The model asks "where is the price right now within its Bollinger Band?"
  - **NVDA, AMZN:** VOL_RATIO drives predictions locally — volume spikes are the strongest signal.
  - **NFLX:** VOL_RATIO from earlier days (D10) dominates — NFLX price moves are preceded by volume signals several days earlier.
  - **INTC:** RETURN_5D is a large negative driver — recent 5-day losses predict further decline.
- **LIME vs SHAP consistency:** TSLA/AMD results align perfectly (BB_POS + RSI in both). GOOGL's long-range dependency is a new finding SHAP did not reveal. MSFT/AMZN/NVDA show VIX globally (SHAP) but volume/BB locally (LIME) — both valid at different scopes.
- **Pros:** Model-agnostic, easy to interpret, good for explaining single predictions to non-technical clients.
- **Cons:** Can be unstable (different runs give slightly different explanations); slow for large input spaces (780 features).

**SHAP vs LIME:**
- SHAP: global feature importance across all predictions, grounded in cooperative game theory, consistent across runs.
- LIME: local (per-prediction) importance for the most recent window, intuitive, more client-friendly, reveals recency and stock-specific patterns SHAP averages out.
- Both are used together for a complete XAI picture — SHAP tells you what the model generally relies on; LIME tells you what drove today's specific prediction.

---

### 8. NLP Sentiment Analysis — FinBERT

- **What it is:** BERT pre-trained on financial news and fine-tuned for sentiment classification (Positive / Neutral / Negative).
- **Model:** `ProsusAI/finbert` via HuggingFace Transformers.
- **Implementation:** Pipeline classifies up to 20 recent headlines per stock. Sentiment score = `P(positive) - P(negative)`.
- **Why not used as a training feature:** Finnhub free tier only covers ~3.5% of our 7-year training period. 96.5% of training rows would have `sentiment = 0`, making it a near-useless feature that adds noise.
- **How it is used:** Fetched live at inference time. Shown alongside predictions as qualitative context. When sentiment aligns with model direction (e.g., TSLA: model predicts UP, sentiment is POSITIVE), confidence is boosted by +5%.
- **Key finding:** TSLA had positive sentiment (score +0.198) aligning with the model's bullish prediction — confidence boosted to 60%.
- **Pros:** Domain-specific (much better than general BERT for financial text), free model, strong NLP showcase.
- **Cons:** Sentiment alone has low predictive power for short-term price; sparse API coverage on free tier.

---

### 9. Generative AI — Groq + LLaMA

- **What it does:** Takes the model's numerical output (price, direction, confidence, metrics, sentiment) and generates a professional 3–4 sentence client-facing explanation in natural language.
- **Model:** `llama-3.1-8b-instant` via Groq API.
- **Why Groq over Gemini/OpenAI:** Groq's free tier provides 14,400 requests/day with no credit card required. Gemini free tier was exhausted quickly; OpenAI requires billing setup.
- **Also used for:** A Q&A chatbot that answers natural language questions about the predictions ("Which stock looks most bullish?", "Are any stocks predicted to go DOWN?").
- **Pros:** Converts technical output into client-readable text; makes the platform accessible to non-technical users; no hallucination risk on factual questions (model output is passed directly in the prompt).
- **Cons:** LLM responses can be slightly verbose; requires internet connection; subject to API rate limits.

---

### 10. Why SMOTE Was Not Applied

SMOTE (Synthetic Minority Oversampling Technique) is used to address class imbalance in classification problems. It was not needed here for two reasons:

1. **Primary task is regression:** We predict return values (continuous), not binary classes. SMOTE does not apply to regression targets.
2. **Label balance is ~50/50:** The UP/DOWN split across the dataset is approximately 50% each (confirmed in diagnostics), which is balanced by nature — stock markets go up and down roughly equally at the daily level.

---

## Results Summary

| Ticker | R² | MAPE | Dir Acc | Sentiment |
|---|---|---|---|---|
| AAPL | 0.983 | 1.09% | 53.1% | Neutral |
| MSFT | 0.934 | 0.97% | 53.7% | Neutral |
| GOOGL | 0.958 | 1.30% | 56.2% ✅ | Neutral |
| AMZN | 0.967 | 1.32% | 50.8% ⚠️ | Neutral |
| META | 0.967 | 1.48% | 53.9% | Neutral |
| NVDA | 0.951 | 2.58% | 54.8% | Neutral (+0.147) |
| TSLA | 0.979 | 2.94% | 55.3% ✅ | Positive (+0.198) |
| AMD | 0.977 | 2.21% | 52.5% ⚠️ | Neutral |
| NFLX | 0.990 | 1.36% | 53.9% | Neutral |
| INTC | 0.984 | 2.34% | 52.8% | Neutral |

**Best performer:** GOOGL — 56.2% directional accuracy, R²=0.958
**Most bullish signal:** TSLA — model UP + positive sentiment aligned
**Weakest:** AMZN (50.8%) and INTC (classifier biased to UP, F1=0%)

---

## Limitations

1. **Directional accuracy is modest (50–56%)** — this is realistic and consistent with academic literature on technical-indicator-based forecasting. Stock markets are efficient and largely unpredictable from price history alone.
2. **Classifier bias toward UP** — trained on 2018–2025 data which was predominantly a bull market. The classifier rarely predicts DOWN because most training days were UP.
3. **INTC classifier failure** — F1 score of 0% means the classifier never predicted DOWN for INTC. This is a known limitation and should be flagged when presenting results.
4. **No fundamental data** — earnings, P/E ratio, revenue growth, insider trading are not included. These would meaningfully improve predictions.
5. **Sentiment coverage** — Finnhub free tier only provides recent headlines, not historical. Sentiment cannot be backtested.
6. **No macro features** — interest rates, inflation, GDP data are not included.

---

## Future Improvements

- Add earnings calendar features (pre/post earnings volatility spike)
- Use Transformer architecture (better long-range dependency)
- Incorporate fundamental data (P/E, revenue growth)
- Train on longer history (10+ years)
- Add attention mechanism to identify which days matter most
- Use ensemble of models (GRU + XGBoost + Transformer) for more robust predictions
- Address classifier UP-bias with class weighting or threshold tuning

---

## Tech Stack

| Component | Library/Tool |
|---|---|
| Data — Stocks | Tiingo REST API |
| Data — Market | yfinance (SPY, VIX) |
| Data — News | Finnhub API |
| Deep Learning | TensorFlow / Keras |
| Feature Scaling | scikit-learn MinMaxScaler |
| XAI | SHAP, LIME |
| NLP Sentiment | HuggingFace Transformers (FinBERT) |
| GenAI Explanations | Groq API (LLaMA 3.1 8B) |
| Visualisation | Matplotlib |
| Data Processing | NumPy, Pandas |
| Model Persistence | joblib, Keras .keras format |

---

## Disclaimer

All predictions are generated by a machine learning model trained on historical data. They do not constitute financial advice. Past model performance does not guarantee future accuracy. Always consult a qualified financial advisor before making investment decisions.
