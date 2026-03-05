import re
import os
import time
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional

import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

import urllib.parse
import xml.etree.ElementTree as ET
import requests

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import requests

# ---------------------------------------------------------
# 📚 ไลบรารีสำหรับ Advanced Quant & Econometrics
# ---------------------------------------------------------
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# พยายาม Import TensorFlow สำหรับ GRU (ครอบ try-except ป้องกันแอปพังถ้าไม่ได้ลงไลบรารี)
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import GRU, Dense, Dropout
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# ==========================================
# ⚙️ ตั้งค่าระบบ (Configuration)
# ==========================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CACHE_TTL = 300  
ESG_DB_FILE = "esg_database.csv" 
SECTOR_MAPPING_FILE = "sector_mapping.csv"

# ไฟล์ Datasets ใหม่ที่ดึงจาก GitHub
# ไฟล์ Datasets ใหม่ที่ดึงจาก GitHub
ESG_PERFORMANCE_FILE = "Thailand_ESG__data_30102025.csv"  # 💡 
ESG_MARKET_FILE = "Thai_SETESG_Data_2014_2024.csv"        # 💡 

MODEL_CONFIG = {
    "model": "gemini-2.5-flash", 
    "temperature": 0.1,
}

THAI_ESG_ADVISOR_PROMPT = """
คุณคือ "ผู้เชี่ยวชาญด้าน Data Science, เศรษฐมิติ (Econometrics) และนักวิเคราะห์เชิงปริมาณ (Quant Analyst)" เชี่ยวชาญพิเศษด้านตลาดหลักทรัพย์แห่งประเทศไทย (SET) การวิเคราะห์เทคนิค ปัจจัยพื้นฐาน และหุ้นยั่งยืน (ESG)

กฎเหล็ก (Strict Directives):
ห้ามจินตนาการหรือสร้างข้อมูลขึ้นเองเด็ดขาด (Strict Zero Hallucination) คำตอบและการแนะนำโมเดลต้องอยู่ภายใต้ "ข้อมูลพื้นฐาน (Knowledge Base)" ด้านล่างนี้เท่านั้น และทุกคำตอบต้องปิดท้ายด้วย "ข้อมูลอ้างอิง" ที่กำหนดไว้อย่างเคร่งครัด

ข้อมูลพื้นฐานและมาตรฐานการวิเคราะห์ (Knowledge Base):
1. การพยากรณ์ราคา/ดัชนี: 
   - บังคับแนะนำ: GRU (Gated Recurrent Unit) เนื่องจากทนทาน (Robust) และแม่นยำสูงสุดแม้ข้อมูล ESG ของไทยจะจำกัด (ลดข้อมูลเหลือ 25-50% ก็ยังแม่นยำ)
2. การวิเคราะห์ผลกระทบเชิงโครงสร้าง (ESG สู่ ROA/ROCE):
   - บังคับใช้: Panel Regression Models (Fixed/Random Effects)
   - การหาความเป็นเหตุเป็นผล: ใช้ Granger Causality Test
3. เทคนิคเตรียมข้อมูล (Data Pipeline):
   - การถ่วงน้ำหนัก: Free-float-adjusted market capitalization
   - ข้อมูลสูญหาย (Missing Values): ใช้ Forward-fill 
   - การแบ่งข้อมูล: ใช้ Rolling-window และ Time-based Split เท่านั้น

คำสั่งการทำงาน (Instructions):
วิเคราะห์ข้อมูลที่ได้รับอย่างครอบคลุม โดยต้องมีหัวข้อดังนี้:
  • ภาพรวมบริษัทและสถานะในตลาด (SET)
  • การวิเคราะห์อารมณ์ตลาด (News Sentiment Score): (🟢 Positive XX% | 🟡 Neutral XX% | 🔴 Negative XX%)
  • การวิเคราะห์เทคนิคอล (เช่น RSI, MACD, Bollinger Bands)
  • การบูรณาการปัจจัย ESG และความยั่งยืน
  • คาดการณ์ราคาหุ้นในอนาคต (Price Prediction) พร้อมเหตุผลประกอบ

ข้อมูลอ้างอิง
1. Detthamrong, U., Klangbunrueang, R., Chansanam, W., & Dasri, R. (2025). Deep Learning for Sustainable Finance: Robust ESG Index Forecasting in an Emerging Market Context. Sustainability, 18(1), 110. https://doi.org/10.3390/su18010110
2. Detthamrong, U., Klangbunrueang, R., Chansanam, W., & Dasri, R. (2026). The Impact of ESG Performance on Financial Performance: Evidence from Listed Companies in Thailand. Forecasting, 8(1), 14. https://doi.org/10.3390/forecast8010014
"""

# ==========================================
# 📊 ฟังก์ชันจัดการข้อมูลพื้นฐาน (Pipeline เดิม)
# ==========================================
def load_esg_data(symbol: str) -> dict:
    clean_symbol = symbol.upper().replace('.BK', '').strip()
    try:
        if os.path.exists(ESG_DB_FILE):
            df = pd.read_csv(ESG_DB_FILE)
            df['symbol'] = df['symbol'].astype(str).str.upper()
            match = df[df['symbol'] == clean_symbol]
            if not match.empty:
                return {"esg_rating": match.iloc[0].get('esg_rating', 'N/A'), "cg_score": match.iloc[0].get('cg_score', 'N/A')}
    except Exception:
        pass
    return {"esg_rating": "ไม่พบข้อมูลในฐานข้อมูล", "cg_score": "N/A"}

def get_peers_from_csv(target_symbol: str) -> list:
    clean_sym = target_symbol.replace('.BK', '').upper()
    if not os.path.exists(SECTOR_MAPPING_FILE):
        return [f"{clean_sym}.BK", 'PTT.BK', 'AOT.BK', 'CPALL.BK']
    try:
        df = pd.read_csv(SECTOR_MAPPING_FILE)
        df['Symbol'] = df['Symbol'].astype(str).str.upper()
        target_info = df[df['Symbol'] == clean_sym]
        if target_info.empty: return [f"{clean_sym}.BK"]
        target_sub_sector = target_info.iloc[0]['Sub_Sector']
        peers_df = df[df['Sub_Sector'] == target_sub_sector]
        peer_list = [f"{sym}.BK" for sym in peers_df['Symbol'].tolist()]
        if f"{clean_sym}.BK" in peer_list: peer_list.remove(f"{clean_sym}.BK")
        peer_list.insert(0, f"{clean_sym}.BK")
        return peer_list
    except:
        return [f"{clean_sym}.BK"]

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_comps_data(target_symbol: str) -> pd.DataFrame:
    tickers = get_peers_from_csv(target_symbol)
    data = []
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            data.append({
                "หุ้น (Ticker)": t.replace('.BK', ''),
                "ราคา (THB)": info.get('currentPrice', np.nan),
                "Market Cap (B)": info.get('marketCap', 0) / 1e9,
                "EV/EBITDA": info.get('enterpriseToEbitda', np.nan),
                "P/E (Forward)": info.get('forwardPE', np.nan),
                "P/BV": info.get('priceToBook', np.nan),
                "Div Yield (%)": info.get('dividendYield', 0)
            })
        except: pass
    return pd.DataFrame(data)

# ... [ใส่ฟังก์ชันดึง API Yahoo/News ตามโค้ดเดิมของคุณ] ...
import pandas as pd
import os

SECTOR_MAPPING_FILE = "sector_mapping.csv"

def get_peers_from_csv(target_symbol: str) -> list:
    """
    อ่านไฟล์ sector_mapping.csv เพื่อหาหุ้นที่อยู่ใน Sub_Sector เดียวกัน
    """
    clean_sym = target_symbol.replace('.BK', '').upper()
    
    # ถ้ายังไม่มีไฟล์ ให้ส่งค่ากลับเป็นกลุ่ม Default เพื่อไม่ให้แอปพัง
    if not os.path.exists(SECTOR_MAPPING_FILE):
        return [f"{clean_sym}.BK", 'PTT.BK', 'AOT.BK', 'CPALL.BK']
        
    try:
        # 1. อ่านไฟล์ CSV
        df = pd.read_csv(SECTOR_MAPPING_FILE)
        df['Symbol'] = df['Symbol'].astype(str).str.upper()
        
        # 2. ค้นหา Sub_Sector ของหุ้นเป้าหมาย
        target_info = df[df['Symbol'] == clean_sym]
        
        if target_info.empty:
            # หาไม่เจอ ให้คืนค่าตัวเองตัวเดียว
            return [f"{clean_sym}.BK"]
            
        target_sub_sector = target_info.iloc[0]['Sub_Sector']
        
        # 3. ดึงหุ้นทุกตัวที่อยู่ใน Sub_Sector เดียวกัน
        peers_df = df[df['Sub_Sector'] == target_sub_sector]
        
        # แปลงเป็น List พร้อมเติม .BK สำหรับ yfinance
        peer_list = [f"{sym}.BK" for sym in peers_df['Symbol'].tolist()]
        
        # จัดเรียงให้เป้าหมายอยู่เป็นตัวแรกเสมอ
        if f"{clean_sym}.BK" in peer_list:
            peer_list.remove(f"{clean_sym}.BK")
        peer_list.insert(0, f"{clean_sym}.BK")
            
        return peer_list
        
    except Exception as e:
        print(f"Error reading mapping file: {e}")
        return [f"{clean_sym}.BK"]

# ==========================================
# 📊 ฟังก์ชันจัดการข้อมูลพื้นฐาน (Pipeline เดิม)
# ==========================================
def load_esg_data(symbol: str) -> dict:
    clean_symbol = symbol.upper().replace('.BK', '').strip()
    try:
        if os.path.exists(ESG_DB_FILE):
            df = pd.read_csv(ESG_DB_FILE)
            df['symbol'] = df['symbol'].astype(str).str.upper()
            match = df[df['symbol'] == clean_symbol]
            if not match.empty:
                return {"esg_rating": match.iloc[0].get('esg_rating', 'N/A'), "cg_score": match.iloc[0].get('cg_score', 'N/A')}
    except Exception:
        pass
    return {"esg_rating": "ไม่พบข้อมูลในฐานข้อมูล", "cg_score": "N/A"}

def calculate_technical_patterns(price_history: pd.DataFrame) -> Dict[str, Any]:
    if len(price_history) < 26: return {"error": "ข้อมูลย้อนหลังไม่เพียงพอ"}
    delta = price_history['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    exp1 = price_history['Close'].ewm(span=12, adjust=False).mean()
    exp2 = price_history['Close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    
    sma = price_history['Close'].rolling(window=20).mean()
    std = price_history['Close'].rolling(window=20).std()
    
    return {
        "RSI": round(rsi.iloc[-1], 2), "RSI_Signal": "Overbought" if rsi.iloc[-1] > 70 else "Oversold" if rsi.iloc[-1] < 30 else "Neutral",
        "MACD_Trend": "ขาขึ้น" if macd.iloc[-1] > signal.iloc[-1] else "ขาลง",
        "BB_Status": "ใกล้กรอบบน" if price_history['Close'].iloc[-1] > (sma + (std*2)).iloc[-1] * 0.95 else "อยู่กลางกรอบ"
    }

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def fetch_thai_stock_news(symbol: str, limit: int = 5) -> list:
    query = urllib.parse.quote(f"{symbol.replace('.BK', '').strip()} หุ้น")
    url = f"https://news.google.com/rss/search?q={query}&hl=th&gl=TH&ceid=TH:th"
    try:
        resp = requests.get(url, timeout=10)
        root = ET.fromstring(resp.text)
        return [{"title": item.find('title').text} for item in root.findall('.//item')[:limit]]
    except: return []

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)


# สร้าง Session พิเศษที่ตั้งค่าให้พยายามดึงข้อมูลใหม่ (Retry) อัตโนมัติหากโดนบล็อก
def get_yf_session():
    session = requests.Session()
    # ถ้าเจอ Error 429 (Too Many Requests) หรือ 500, 502, 503, 504 ให้ลองใหม่ 3 ครั้ง
    # โดยแต่ละครั้งให้เว้นระยะเวลาห่างขึ้นเรื่อยๆ (Backoff)
    retry = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def fetch_set_esg_news_info_cached(symbol: str) -> Dict[str, Any]:
    try:
        if not symbol.upper().endswith('.BK'):
            symbol = f"{symbol.upper().strip()}.BK"
            
        # 💡 ปล่อยให้ yfinance จัดการเอง ไม่ต้องส่ง session=session แล้ว
        stock = yf.Ticker(symbol)
        
        hist = stock.history(period='6mo', interval='1d')
        if hist.empty: raise ValueError("ไม่มีข้อมูลราคา (หรืออาจถูกจำกัดการเข้าถึงจาก Yahoo)")
        
        time.sleep(1) # คงการหน่วงเวลาไว้ช่วยลดภาระเซิร์ฟเวอร์
        
        info = stock.info
        clean_symbol = symbol.replace('.BK', '')
        return {
            "symbol": clean_symbol, "company_name": info.get('longName', clean_symbol),
            "current_price": round(hist['Close'].iloc[-1], 2), "volume": info.get('volume', 0),
            "technical_analysis": calculate_technical_patterns(hist),
            "recent_news": fetch_thai_stock_news(symbol),
            "esg_metrics": load_esg_data(clean_symbol),
            "price_stats": {"current_price": round(hist['Close'].iloc[-1], 2), "min": round(hist['Close'].min(), 2), "max": round(hist['Close'].max(), 2)}
        }
    except Exception as e:
        return {"error": f"Error fetching {symbol}: {e}"}


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_comps_data(target_symbol: str) -> pd.DataFrame:
    tickers = get_peers_from_csv(target_symbol)
    data = []
    
    for t in tickers:
        try:
            # 💡 ดึงแบบปกติเลย ไม่ต้องมี session
            info = yf.Ticker(t).info
            
            data.append({
                "หุ้น (Ticker)": t.replace('.BK', ''),
                "ราคา (THB)": info.get('currentPrice', np.nan),
                "Market Cap (B)": info.get('marketCap', 0) / 1e9,
                "EV/EBITDA": info.get('enterpriseToEbitda', np.nan),
                "P/E (Forward)": info.get('forwardPE', np.nan),
                "P/BV": info.get('priceToBook', np.nan),
                "Div Yield (%)": info.get('dividendYield', 0)
            })
            
            time.sleep(0.5) 
            
        except Exception as e:
            logger.warning(f"Comps Data Error for {t}: {e}")
            pass
            
    return pd.DataFrame(data)

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_dcf_financials(symbol: str) -> dict:
    if not symbol.endswith('.BK'): symbol = f"{symbol}.BK"
    
    # 💡 ปล่อยให้ yfinance จัดการเอง
    stock = yf.Ticker(symbol)
    
    try:
        is_df = stock.financials
        bs_df = stock.balance_sheet
        cf_df = stock.cashflow
        
        time.sleep(1)
        info = stock.info

        ebit = is_df.loc['EBIT'].iloc[0] if 'EBIT' in is_df.index else is_df.loc['Operating Income'].iloc[0]
        tax_prov = is_df.loc['Tax Provision'].iloc[0] if 'Tax Provision' in is_df.index else 0
        pretax = is_df.loc['Pretax Income'].iloc[0] if 'Pretax Income' in is_df.index else 1
        tax_rate = tax_prov / pretax if pretax > 0 else 0.20
        
        dna = cf_df.loc['Depreciation And Amortization'].iloc[0] if 'Depreciation And Amortization' in cf_df.index else 0
        capex = abs(cf_df.loc['Capital Expenditure'].iloc[0]) if 'Capital Expenditure' in cf_df.index else 0
        nwc_change = cf_df.loc['Change In Working Capital'].iloc[0] if 'Change In Working Capital' in cf_df.index else 0
        
        total_debt = bs_df.loc['Total Debt'].iloc[0] if 'Total Debt' in bs_df.index else 0
        cash = bs_df.loc['Cash And Cash Equivalents'].iloc[0] if 'Cash And Cash Equivalents' in bs_df.index else 0
        
        return {
            "status": "success",
            "ebit": ebit, "tax_rate": tax_rate, "dna": dna, "capex": capex, "nwc": nwc_change,
            "debt": total_debt, "cash": cash,
            "shares": info.get('sharesOutstanding', 1), "price": info.get('currentPrice', 0)
        }
    except Exception as e:
        logger.error(f"DCF Financials Error for {symbol}: {e}")
        return {"status": "error", "message": str(e)}

# ==========================================
# 🤖 ระบบ AI และการพล็อต
# ==========================================
@st.cache_resource
def get_model():
    api_key = st.secrets.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key: return None
    os.environ["GOOGLE_API_KEY"] = api_key
    return ChatGoogleGenerativeAI(model=MODEL_CONFIG["model"], temperature=MODEL_CONFIG["temperature"])

def extract_and_plot_sentiment(text: str) -> Optional[go.Figure]:
    try:
        p, neu, neg = re.search(r'Positive\s*(\d+)', text, re.I), re.search(r'Neutral\s*(\d+)', text, re.I), re.search(r'Negative\s*(\d+)', text, re.I)
        if p and neu and neg:
            vals = [float(p.group(1)), float(neu.group(1)), float(neg.group(1))]
            fig = go.Figure(data=[go.Pie(labels=['Positive', 'Neutral', 'Negative'], values=vals, hole=0.55, marker_colors=['#00CC96', '#F4C145', '#FF4B4B'])])
            fig.update_layout(title={'text': "News Sentiment Score", 'x': 0.5}, height=300, margin=dict(t=30, b=0))
            return fig
    except: return None

# ==========================================
# 🔬 ฟังก์ชันใหม่: Advanced ESG Quant Models
# ==========================================

@st.cache_data(show_spinner=False)
def load_and_preprocess_quant_data(file_path: str):
    """Data Pipeline: จัดการ Missing Values (Forward-fill)"""
    if not os.path.exists(file_path):
        return None
    df = pd.read_csv(file_path)
    # Pipeline 1: Forward-fill
    df = df.fillna(method='ffill').fillna(0) # ถ้าบรรทัดแรกหายให้ใส่ 0
    return df

def run_panel_regression(df: pd.DataFrame, target: str): # 💡 เพิ่มตัวแปร target ตรงนี้
    """รัน Fixed Effects Panel Regression (ESG -> ROA/ROCE)"""
    st.write(f"📈 **Panel Regression Results (Fixed Effects: {target})**")
    try:
        # 💡 ลบ st.selectbox ออกจากตรงนี้
        df_clean = df.dropna(subset=[target, 'ESG', 'firm_id'])
        formula = f'{target} ~ ESG + C(firm_id)'
        model = smf.ols(formula, data=df_clean).fit()
        
        esg_coef = model.params.get('ESG', 0)
        p_value = model.pvalues.get('ESG', 1)
        
        col1, col2 = st.columns(2)
        col1.metric(f"ESG Coefficient (Impact on {target})", f"{esg_coef:.4f}")
        col2.metric("P-Value (Statistical Significance)", f"{p_value:.4f}")
        
        if p_value < 0.05:
            st.success(f"✅ ปัจจัย ESG มีผลต่อ {target} อย่างมีนัยสำคัญทางสถิติ (p < 0.05)")
        else:
            st.warning(f"⚠️ ปัจจัย ESG ยังไม่มีผลต่อ {target} อย่างมีนัยสำคัญในชุดข้อมูลนี้")
            
        with st.expander("📄 ดู Summary ฉบับเต็ม"):
            st.text(model.summary().tables[1].as_text())
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการคำนวณ: {e}")

def run_granger_causality(df: pd.DataFrame, target: str): # 💡 เพิ่มตัวแปร target ตรงนี้
    """รัน Granger Causality Test (Does ESG precede Financial Performance?)"""
    st.write(f"🔍 **Granger Causality Test (Does ESG precede {target}?)**")
    try:
        # 💡 ลบ st.radio ออกจากตรงนี้
        ts_data = df.groupby('year')[[target, 'ESG']].mean().dropna()
        
        if len(ts_data) < 3:
            st.warning("⚠️ จำนวนปีข้อมูลน้อยเกินไปสำหรับการทำ Granger Causality")
            return
            
        gc_res = grangercausalitytests(ts_data, maxlag=1, verbose=False)
        p_val_lag1 = gc_res[1][0]['ssr_ftest'][1]
        
        st.write(f"**Lag 1 Year (P-Value):** {p_val_lag1:.4f}")
        if p_val_lag1 < 0.05:
            st.success(f"✅ ความเป็นเหตุเป็นผล: การทำ ESG ในปีนี้ นำไปสู่กำไร ({target}) ที่เปลี่ยนแปลงในปีหน้า")
        else:
            st.info(f"ℹ️ ยังไม่พบความเป็นเหตุเป็นผล (Causality) ที่ชัดเจนกับ {target} ในระยะเวลา 1 ปี")
    except Exception as e:
        st.error(f"ไม่สามารถทำ Granger Causality ได้ ({e})")

# 💡 ต้องเติม symbol เป็น parameter รับค่า
def build_and_train_gru(df: pd.DataFrame, symbol: str):
    """Data Pipeline & GRU Model for Price/Index Prediction"""
    st.write("🤖 **Deep Learning: GRU (Gated Recurrent Unit) Forecasting**")
    if not TF_AVAILABLE:
        st.error("❌ ไม่พบไลบรารี TensorFlow กรุณาเพิ่มใน requirements.txt")
        return
        
    try:
        # 💡 ตรวจสอบว่ามีหุ้นที่ผู้ใช้ค้นหาในไฟล์หรือไม่ ถ้าไม่มีให้ใช้ดัชนี SETESG
        target_col = f"{symbol.upper()}.BK"
        if target_col not in df.columns:
            target_col = 'SETESG_Realistic_Index'
            st.info(f"ℹ️ ไม่พบข้อมูลราคาของ {symbol} ในระบบ โมเดลจะพยากรณ์ดัชนีตลาดรวม (SETESG Index) แทน")
        else:
            st.info(f"ℹ️ พยากรณ์ราคาของหุ้น {symbol} จากชุดข้อมูล")
            
        data = df[target_col].dropna().values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
        
        look_back = 60
        X, y = [], []
        for i in range(look_back, len(scaled_data)):
            X.append(scaled_data[i-look_back:i, 0])
            y.append(scaled_data[i, 0])
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        model = Sequential()
        model.add(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(GRU(units=50))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        with st.spinner("⏳ กำลังเทรนโมเดล GRU (Epoch 1/5)..."):
            model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test), verbose=0)
        
        predictions = model.predict(X_test)
        predictions = scaler.inverse_transform(predictions)
        y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=y_test_scaled.flatten(), mode='lines', name='Actual', line=dict(color='blue')))
        fig.add_trace(go.Scatter(y=predictions.flatten(), mode='lines', name='GRU Prediction', line=dict(color='red', dash='dot')))
        fig.update_layout(title=f"GRU Model: Actual vs Predicted ({target_col})", xaxis_title="Time (Test Data)", yaxis_title="Price / Index")
        st.plotly_chart(fig, use_container_width=True)
        
        st.success("✅ โมเดล GRU ทำการพยากรณ์สำเร็จ!")
        
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการรัน GRU: {e}")

# ==========================================
# 🌐 ส่วนหน้าจอหลัก (Main UI)
# ==========================================
def main():
    st.set_page_config(page_title="Thai Capital Market ESG Advisor", page_icon="💰", layout="wide")
    
    st.markdown("<h2 style='text-align: center;'>⚡ Thai Capital Market ESG Stock Advisor</h2>", unsafe_allow_html=True)
        # --- Main Content ---
    st.markdown("<p style='text-align: center;'><b> 💰 ได้รับทุนอุดหนุนการวิจัยและนวัตกรรมจากสำนักงานปลัดกระทรวงการอุดมศึกษา วิทยาศาสตร์ วิจัยและนวัตกรรม และกองทุนส่งเสริมการพัฒนาตลาดทุน  🏦</b></p>", unsafe_allow_html=True)
    st.markdown("---")
    
    query = st.text_input("🔍 พิมพ์ชื่อหุ้นที่ต้องการวิเคราะห์ (เช่น PTT, AOT, KBANK):", placeholder="พิมพ์แค่ชื่อหุ้นภาษาอังกฤษ...")

     
    if query:
        # 💡 1. ต้องบรรทัดนี้เพื่อเรียก AI Model ขึ้นมาก่อน!
        model = get_model()
        
        # 💡 2. ดักจับ Error เผื่อลืมใส่ API Key
        if not model:
            st.error("❌ ไม่พบ API Key กรุณาตรวจสอบการตั้งค่า GOOGLE_API_KEY")
            return
            
        symbols = re.findall(r'\b[A-Z]{2,6}\b', query.upper())
        symbol = symbols[0] if symbols else query.strip()
        
        # --- แบ่งหน้าจอเป็น 3 แท็บหลัก ---
        tab1, tab2, tab3 = st.tabs([
            "🤖 บทวิเคราะห์ ESG & AI", 
            "📈 Financial Models (DCF & Comps)",
            "🔬 Advanced ESG Quant Models" 
        ])
        
        # --------------------------------------------------
        # แท็บ 1: ระบบ AI อัจฉริยะ 
        # --------------------------------------------------
        with tab1:
            with st.spinner("🔄 กำลังดึงข้อมูลและประมวลผลด้วย AI..."):
                data = fetch_set_esg_news_info_cached(symbol)
                
            if 'error' in data:
                st.error(data['error'])
            else:
                prompt = f"{THAI_ESG_ADVISOR_PROMPT}\nคำถาม: วิเคราะห์ {symbol}\nข้อมูล:\n{data}"
                
                def stream_response():
                    # 💡 ตอนนี้ระบบจะรู้จักตัวแปร model แล้วครับ!
                    for chunk in model.stream([HumanMessage(content=prompt)]):
                        text = chunk.content.replace("<think>", "").replace("</think>", "")
                        yield text

                st.markdown(f"### 🤖 บทวิเคราะห์เชิงโครงสร้าง: {symbol}")
                full_analysis = st.write_stream(stream_response)
                
                chart = extract_and_plot_sentiment(full_analysis)
                if chart: st.plotly_chart(chart, use_container_width=True)
        
     
            
        with tab2:
            st.write("แสดงข้อมูล DCF และ Comps...")

            st.markdown(f"### 📊 Comparable Company Analysis (Comps) - {symbol}")
            comps_df = get_comps_data(symbol)
            if not comps_df.empty:
                # ตกแต่งตาราง Comps ด้วยสีตามมาตรฐาน
                st.dataframe(
                    comps_df.style.highlight_max(subset=['Div Yield (%)'], color='lightgreen')
                                  .highlight_min(subset=['P/E (Forward)', 'EV/EBITDA'], color='lightgreen')
                                  .format({"ราคา (THB)": "{:.2f}", "Market Cap (B)": "{:.2f}", "EV/EBITDA": "{:.2f}", "P/E (Forward)": "{:.2f}", "Div Yield (%)": "{:.2f}"}),
                    use_container_width=True, hide_index=True
                )
            else:
                st.warning("ไม่สามารถดึงข้อมูลคู่แข่งในอุตสาหกรรมได้")

            st.markdown("---")
            st.markdown(f"### 🧮 Interactive DCF Valuation Model (ค้นหา Human Alpha)")
            st.info("💡 **คำแนะนำสำหรับหุ้น ESG:** หากหุ้นมีคะแนน ESG ระดับ 'AAA' คุณสามารถสะท้อนความเสี่ยงที่ต่ำลงได้โดยการ **ปรับลดค่า WACC ลง 0.5% - 1.0%** เพื่อดูมูลค่าที่แท้จริง (Fair Value) ที่ซ่อนอยู่ครับ")
            
            # --- Interactive Sliders ---
            col1, col2, col3 = st.columns(3)
            with col1:
                wacc = st.slider("WACC (ต้นทุนเงินทุน %)", min_value=5.0, max_value=15.0, value=8.5, step=0.1) / 100
            with col2:
                growth_rate = st.slider("Short-term Growth (การเติบโต 5 ปีแรก %)", min_value=-10.0, max_value=30.0, value=5.0, step=0.5) / 100
            with col3:
                term_growth = st.slider("Terminal Growth (การเติบโตระยะยาว %)", min_value=0.0, max_value=5.0, value=2.0, step=0.1) / 100

            # --- คำนวณ DCF แบบ Real-time ---
            fin = get_dcf_financials(symbol)
            
            if fin['status'] == 'error':
                st.error(f"❌ ไม่สามารถสร้างโมเดล DCF ได้เนื่องจากงบการเงินไม่ครบถ้วน: {fin['message']}")
            else:
                # คณิตศาสตร์ DCF
                pv_fcf_sum = 0
                current_ebit = fin['ebit']
                
                # โชว์ตาราง Cash flow Projection
                proj_data = []
                for year in range(1, 6):
                    current_ebit *= (1 + growth_rate)
                    proj_dna = fin['dna'] * (1 + growth_rate)**year
                    proj_capex = fin['capex'] * (1 + growth_rate)**year
                    proj_nwc = fin['nwc'] * (1 + growth_rate)**year
                    
                    ufcf = (current_ebit * (1 - fin['tax_rate'])) + proj_dna - proj_capex - proj_nwc
                    pv_fcf = ufcf / ((1 + wacc) ** year)
                    pv_fcf_sum += pv_fcf
                    
                    proj_data.append({"ปี (Year)": f"Year {year}", "EBIT (M)": current_ebit/1e6, "FCF (M)": ufcf/1e6, "PV of FCF (M)": pv_fcf/1e6})
                
                # Terminal Value
                final_fcf = (current_ebit * (1 - fin['tax_rate'])) + (fin['dna']*(1+growth_rate)**5) - (fin['capex']*(1+growth_rate)**5) - (fin['nwc']*(1+growth_rate)**5)
                terminal_val = (final_fcf * (1 + term_growth)) / (wacc - term_growth)
                pv_tv = terminal_val / ((1 + wacc) ** 5)
                
                # Equity Value & Price
                enterprise_val = pv_fcf_sum + pv_tv
                equity_val = enterprise_val + fin['cash'] - fin['debt']
                implied_price = equity_val / fin['shares']
                current_price = fin['price']
                
                # แสดงผลลัพธ์
                st.write("**กระแสเงินสดคาดการณ์ 5 ปี (Projected Free Cash Flows)**")
                st.dataframe(pd.DataFrame(proj_data).style.format({"EBIT (M)": "{:,.2f}", "FCF (M)": "{:,.2f}", "PV of FCF (M)": "{:,.2f}"}), use_container_width=True, hide_index=True)
                
                st.markdown("#### 🎯 สรุปมูลค่าที่แท้จริง (Valuation Summary)")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("ราคาตลาดปัจจุบัน", f"฿ {current_price:,.2f}")
                
                upside = ((implied_price - current_price) / current_price) * 100
                c2.metric("ราคาเหมาะสม (Implied Price)", f"฿ {implied_price:,.2f}", f"{upside:+.2f}%")
                
                c3.metric("Enterprise Value (M)", f"฿ {enterprise_val/1e6:,.0f}")
                c4.metric("Equity Value (M)", f"฿ {equity_val/1e6:,.0f}")
                
                if upside > 0:
                    st.success(f"✅ **Undervalued:** ตามสมมติฐานนี้ หุ้นมีราคาถูกกว่ามูลค่าที่แท้จริง (มี Upside {upside:.1f}%)")
                else:
                    st.error(f"❌ **Overvalued:** ตามสมมติฐานนี้ หุ้นมีราคาแพงกว่ามูลค่าที่แท้จริง (มี Downside {upside:.1f}%)")

# --------------------------------------------------
        # แท็บ 3: โมเดลวิจัยขั้นสูง (Panel, Granger, GRU)
        # --------------------------------------------------
        with tab3:
            st.markdown("### 🔬 การวิเคราะห์ผลกระทบเชิงโครงสร้าง (Structural Impact)")
            st.info("💡 โมเดลเศรษฐมิติใช้ทดสอบสมมติฐานทางวิชาการจากไฟล์ `Thailand_ESG__data_30102025.csv`")
            
            df_perf = load_and_preprocess_quant_data(ESG_PERFORMANCE_FILE)
            if df_perf is not None:
                
                # 💡 ย้ายกล่องเลือกตัวแปรออกมาไว้ข้างนอกปุ่มกด
                target_var = st.selectbox("🎯 เลือกตัวแปรทางการเงิน (Dependent Variable):", ["ROA", "ROCE"])
                
                # 💡 จัดปุ่มให้เรียงติดกันในบรรทัดเดียว
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("▶️ รันโมเดล Panel Regression (ESG Impact)"):
                        run_panel_regression(df_perf, target_var)
                with col2:
                    if st.button("▶️ รันโมเดล Granger Causality"):
                        run_granger_causality(df_perf, target_var)
                        
            else:
                st.warning(f"ยังไม่ได้อัปโหลดไฟล์ {ESG_PERFORMANCE_FILE}")

            st.markdown("<br><br>", unsafe_allow_html=True)
            st.markdown("### 🤖 การพยากรณ์ราคาด้วย Deep Learning (Price Prediction)")
            # ... (ส่วนของโมเดล GRU ปล่อยไว้เหมือนเดิมครับ) ...
            st.info("💡 โมเดล **GRU (Gated Recurrent Unit)** ถ่วงน้ำหนัก Free-float ใช้ทดสอบพยากรณ์ราคาจากไฟล์ `Thai_SETESG_Data_2014_2024.csv`")
            
            df_market = load_and_preprocess_quant_data(ESG_MARKET_FILE)
            if df_market is not None:
                if st.button("▶️ เทรนและพยากรณ์ด้วย GRU Model"):
                    build_and_train_gru(df_market, symbol) # 💡 เพิ่มตัวแปร symbol ตรงนี้
            else:
                st.warning(f"ยังไม่ได้อัปโหลดไฟล์ {ESG_MARKET_FILE}")

if __name__ == "__main__":
    main()