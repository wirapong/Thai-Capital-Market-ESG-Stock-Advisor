# 🇹🇭 Thai Capital Market ESG Stock Advisor

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Google Gemini](https://img.shields.io/badge/Google%20Gemini-2.5%20Flash-4285F4?style=for-the-badge&logo=google&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**An AI-powered, research-backed investment analytics platform for ESG-rated securities on the Stock Exchange of Thailand (SET)**

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-Streamlit%20App-FF4B4B?style=for-the-badge)](https://thai-capital-market-esg-stock-advisor.streamlit.app/)

</div>

---

## 📌 Overview

**Thai Capital Market ESG Stock Advisor** is an end-to-end quantitative investment analysis system that integrates **Generative AI**, **Deep Learning**, and **Panel Econometrics** to support data-driven investment decisions in the Thai capital market. The system is built upon two peer-reviewed research publications and designed to bridge the gap between academic ESG research and practical investment analytics.

The platform covers the full analytical workflow — from real-time market data ingestion and technical analysis to structural impact modelling and deep learning price forecasting — all through an interactive, browser-based interface.

---

## 🎯 Research Foundation

This system is developed directly from the following published research:

> **[1]** Detthamrong, U., Klangbunrueang, R., **Chansanam, W.**, & Dasri, R. (2025). Deep Learning for Sustainable Finance: Robust ESG Index Forecasting in an Emerging Market Context. *Sustainability*, *18*(1), 110. https://doi.org/10.3390/su18010110

> **[2]** Detthamrong, U., Klangbunrueang, R., **Chansanam, W.**, & Dasri, R. (2026). The Impact of ESG Performance on Financial Performance: Evidence from Listed Companies in Thailand. *Forecasting*, *8*(1), 14. https://doi.org/10.3390/forecast8010014

| Research Paper | Core Contribution to This System |
|---|---|
| Detthamrong et al. (2025) | GRU model architecture, Free-float weighting, Time-based data split strategy |
| Detthamrong et al. (2026) | Fixed Effects Panel Regression, Granger Causality Test, ESG → ROA/ROCE framework |

> 💡 **Funding:** This project is supported by research grants from the Office of the Permanent Secretary, Ministry of Higher Education, Science, Research and Innovation (OPS MHESI), and the Capital Market Development Fund (CMDF), Thailand.

---

## ✨ Key Features

### 🤖 Tab 1 — AI-Powered ESG & Market Analysis
- **Large Language Model Integration** via Google Gemini 2.5 Flash (LangChain framework)
- Structured system prompt with **Zero Hallucination** enforcement and embedded knowledge base
- Real-time **News Sentiment Analysis** with Positive / Neutral / Negative scoring
- **Technical Analysis** module covering RSI (14-day), MACD (12/26/9), and Bollinger Bands (20-day, ±2σ)
- Streaming response output for low-latency user experience

### 📈 Tab 2 — Financial Valuation Models
- **Comparable Company Analysis (Comps):** Peer group identification by SET Sub-Sector with live market multiples (EV/EBITDA, P/E Forward, P/BV, Dividend Yield)
- **Interactive DCF Model:** Real-time Discounted Cash Flow valuation with adjustable WACC, short-term growth rate, and terminal growth rate via Streamlit sliders
- **ESG-adjusted WACC:** Guidance to reduce WACC by 0.5%–1.0% for AAA-rated ESG companies, reflecting lower risk premium

### 🔬 Tab 3 — Advanced ESG Quantitative Models
- **Fixed Effects Panel Regression** — Tests ESG impact on ROA and ROCE across listed companies
- **Granger Causality Test** — Examines whether ESG scores temporally precede financial performance
- **GRU Deep Learning Model** — Forecasts SET ESG index and individual stock prices with free-float-adjusted weighting

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Streamlit Web Interface                    │
│         Tab 1: AI Analysis  │  Tab 2: Valuation  │  Tab 3: Quant │
└──────────────────┬───────────────────┬─────────────────────┘
                   │                   │
      ┌────────────▼──────┐   ┌────────▼──────────────────┐
      │  Business Logic   │   │     Quantitative Engine    │
      │                   │   │                            │
      │ • LangChain/LLM   │   │ • Panel Regression (OLS)  │
      │ • Technical Calc  │   │ • Granger Causality Test  │
      │ • DCF Engine      │   │ • GRU Neural Network      │
      │ • Peer Mapping    │   │ • MinMaxScaler Pipeline   │
      └────────────┬──────┘   └────────┬──────────────────┘
                   │                   │
      ┌────────────▼───────────────────▼──────────────────┐
      │                    Data Layer                       │
      │  Yahoo Finance API │ Google News RSS │ CSV Datasets │
      └─────────────────────────────────────────────────────┘
```

---

## 🗂️ Project Structure

```
thai-capital-market-esg-stock-advisor/
│
├── app.py                          # Main application entry point
├── requirements.txt                # Python dependencies
├── .streamlit/
│   └── secrets.toml                # API keys (not committed)
│
├── data/
│   ├── esg_database.csv            # ESG ratings & CG scores for SET companies
│   ├── sector_mapping.csv          # SET 8-sector / 28-sub-sector classification
│   ├── Thailand_ESG__data_30102025.csv   # Panel dataset (ESG + ROA/ROCE by firm-year)
│   └── Thai_SETESG_Data_2014_2024.csv    # Time-series dataset (2014–2024, for GRU)
│
└── README.md
```

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.10 or higher
- A valid **Google API Key** with access to Gemini models

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/thai-capital-market-esg-stock-advisor.git
cd thai-capital-market-esg-stock-advisor
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API Key

Create `.streamlit/secrets.toml`:

```toml
GOOGLE_API_KEY = "your_google_api_key_here"
```

Or set as an environment variable:

```bash
export GOOGLE_API_KEY="your_google_api_key_here"
```

### 4. Run the Application

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

---

## 📦 Dependencies

| Package | Version | Purpose |
|---|---|---|
| `streamlit` | ≥1.30 | Web UI framework |
| `yfinance` | ≥0.2 | Yahoo Finance market data API |
| `pandas` | ≥2.0 | Data manipulation |
| `numpy` | ≥1.24 | Numerical computing |
| `plotly` | ≥5.0 | Interactive visualizations |
| `statsmodels` | ≥0.14 | Panel regression, Granger causality |
| `scikit-learn` | ≥1.3 | MinMaxScaler, data preprocessing |
| `tensorflow` | ≥2.13 | GRU deep learning model |
| `langchain-google-genai` | ≥0.1 | Gemini LLM integration via LangChain |
| `requests` | ≥2.31 | HTTP client with retry mechanism |

---

## 🔬 Methodology

### Data Pipeline

```
Raw Data Sources
     │
     ├── Yahoo Finance API  →  Price history (6mo, 1D interval)
     │                         Financial statements (IS, BS, CF)
     │                         Market multiples (PE, PBV, EV/EBITDA)
     │
     ├── Google News RSS    →  Recent news headlines (Thai language)
     │
     └── CSV Datasets       →  ESG ratings, firm-year financials,
                               SET ESG index (2014–2024)
     │
     ▼
Data Preprocessing
     ├── Missing Values:    Forward-fill (LOCF) → Zero-fill (first row)
     ├── Weighting:         Free-float-adjusted market capitalization
     ├── Peer Grouping:     Sub-Sector matching via sector_mapping.csv
     ├── Scaling:           MinMaxScaler → [0, 1] for GRU input
     └── Splitting:         Time-based 80/20 split (no shuffling)
```

### Model 1 — Fixed Effects Panel Regression

Tests whether ESG scores have a statistically significant impact on firm financial performance:

$$Y_{it} = \alpha + \beta_1 \text{ESG}_{it} + \mu_i + \varepsilon_{it}$$

Where $Y_{it}$ ∈ {ROA, ROCE}, $\mu_i$ = firm fixed effects, significance threshold: **p < 0.05**

### Model 2 — Granger Causality Test

Tests temporal precedence: *does ESG in year t predict financial performance in year t+1?*

- Annual cross-sectional means aggregated to market-level time series
- Maximum lag: **1 year**
- Test statistic: **SSR F-test**, significance threshold: **p < 0.05**

### Model 3 — GRU Neural Network

Forecasts SET ESG index or individual stock price:

```
Input Sequence (60 timesteps) → GRU(50) → Dropout(0.2)
                               → GRU(50) → Dropout(0.2)
                               → Dense(1) → Predicted Price
```

- **Optimizer:** Adam
- **Loss Function:** Mean Squared Error (MSE)
- **Epochs:** 5 | **Batch Size:** 32
- **Validation:** Held-out test set (last 20% of time series)

### Model 4 — Interactive DCF Valuation

$$\text{UFCF}_t = \text{EBIT}_t \times (1 - \tau) + \text{D\&A}_t - \text{CapEx}_t - \Delta\text{NWC}_t$$

$$\text{Terminal Value} = \frac{\text{UFCF}_5 \times (1+g)}{\text{WACC} - g}$$

$$\text{Equity Value} = \text{PV(FCF)} + \text{PV(TV)} + \text{Cash} - \text{Debt}$$

---

## 🖥️ Usage

1. **Navigate to the app:** [https://thai-capital-market-esg-stock-advisor.streamlit.app/](https://thai-capital-market-esg-stock-advisor.streamlit.app/)
2. **Enter a SET ticker symbol** (e.g., `PTT`, `KBANK`, `AOT`, `BDMS`) in the search box
3. **Browse the three analysis tabs:**

| Tab | What You Get |
|---|---|
| 🤖 AI ESG Analysis | Full AI-generated research report with sentiment chart |
| 📈 DCF & Comps | Peer comparison table + interactive DCF fair value |
| 🔬 Advanced Quant | Run Panel Regression, Granger Test, or GRU forecast |

---

## 📊 Datasets

| File | Description | Use |
|---|---|---|
| `esg_database.csv` | ESG ratings (AAA–BBB) and CG scores for ~600 SET companies | ESG badge display in AI analysis |
| `sector_mapping.csv` | SET 8-sector / 28-sub-sector classification | Peer group identification for Comps |
| `Thailand_ESG__data_30102025.csv` | Firm-year panel: ESG score, ROA, ROCE, firm_id, year | Panel Regression & Granger Causality |
| `Thai_SETESG_Data_2014_2024.csv` | SETESG daily/monthly index + constituent prices (2014–2024) | GRU price forecasting |

> ⚠️ **Note:** The panel and time-series datasets are used exclusively for research-grade modelling. Market data is fetched live from Yahoo Finance at runtime.

---

## 🤖 AI System Prompt Design

The LLM component operates under a structured **Knowledge Base prompt** that enforces:

- **Zero Hallucination policy** — responses are strictly grounded in provided data
- **Mandatory model recommendations** aligned with Detthamrong et al. (2025, 2026)
- **Structured output format** covering: company overview, news sentiment score, technical analysis, ESG integration, and price forecast
- **Mandatory citations** to the research papers in every response

The AI analyst role is configured as a specialist in *Data Science, Econometrics, and Quantitative Analysis for the Thai capital market*.

---

## 📐 SET Industry Classification

The system implements the official SET industry classification structure:

| # | Sector (กลุ่มอุตสาหกรรม) | Sub-Sectors (หมวดธุรกิจ) |
|---|---|---|
| 1 | Agro & Food Industry | Agribusiness, Food & Beverage |
| 2 | Consumer Products | Fashion, Home & Office Products, Personal Products & Pharmaceuticals |
| 3 | Financials | Banking, Finance & Securities, Insurance |
| 4 | Industrials | Automotive, Industrial Materials & Machine, Packaging, Paper & Printing Materials, Petrochemicals & Chemicals, Steel & Metal Products |
| 5 | Property & Construction | Construction Materials, Construction Services, Property Development, Property Fund & REITs |
| 6 | Resources | Energy & Utilities, Mining |
| 7 | Services | Commerce, Health Care Services, Media & Publishing, Professional Services, Tourisms & Leisure, Transportation & Logistics |
| 8 | Technology | Electronic Components, Information & Communication Technology |

---

## 📄 Citation

If you use this system or the underlying datasets in your research, please cite:

```bibtex
@article{detthamrong2025deep,
  title   = {Deep Learning for Sustainable Finance: Robust ESG Index
             Forecasting in an Emerging Market Context},
  author  = {Detthamrong, Udomsak and Klangbunrueang, Rujiras and
             Chansanam, Wirapong and Dasri, Rossukon},
  journal = {Sustainability},
  volume  = {18},
  number  = {1},
  pages   = {110},
  year    = {2025},
  doi     = {10.3390/su18010110}
}

@article{detthamrong2026impact,
  title   = {The Impact of ESG Performance on Financial Performance:
             Evidence from Listed Companies in Thailand},
  author  = {Detthamrong, Udomsak and Klangbunrueang, Rujiras and
             Chansanam, Wirapong and Dasri, Rossukon},
  journal = {Forecasting},
  volume  = {8},
  number  = {1},
  pages   = {14},
  year    = {2026},
  doi     = {10.3390/forecast8010014}
}
```

---

## 👥 Authors & Affiliations

**Wirapong Chansanam, Ph.D.**
Associate Professor, Faculty of Humanities and Social Sciences
Khon Kaen University (KKU), Thailand
Visiting Scholar, Nanyang Technological University (NTU), Singapore
✉️ wirach@kku.ac.th | 🔗 [Scopus Profile](https://www.scopus.com/authid/detail.uri?authorId=56623107000)

**Udomsak Detthamrong, Ph.D.** — Khon Kaen University

**Rujiras Klangbunrueang** — Khon Kaen University

**Rossukon Dasri** — Khon Kaen University

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## ⚠️ Disclaimer

> This application is developed for **academic research and educational purposes only**. The AI-generated analyses, model outputs, and valuation estimates do **not** constitute financial advice, investment recommendations, or solicitation to buy or sell any securities. Users should conduct independent due diligence and consult a licensed financial advisor before making any investment decisions. Past performance of models does not guarantee future results.

---

<div align="center">

**Digital Humanities Research Group | Khon Kaen University**

*Supported by OPS MHESI & Capital Market Development Fund (CMDF)*

⭐ If this project is useful for your research, please consider starring the repository.

</div>
