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

# ==========================================
# ⚙️ ตั้งค่าระบบ (Configuration)
# ==========================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CACHE_TTL = 300  
ESG_DB_FILE = "esg_database.csv" 

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
def fetch_set_esg_news_info_cached(symbol: str) -> Dict[str, Any]:
    try:
        if not symbol.upper().endswith('.BK'): symbol = f"{symbol.upper().strip()}.BK"
        stock = yf.Ticker(symbol)
        hist = stock.history(period='6mo', interval='1d')
        if hist.empty: raise ValueError("No data")
        
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

# ==========================================
# 📈 ฟังก์ชันใหม่: Financial Models (Comps & DCF)
# ==========================================

# 1. ฐานข้อมูลคู่แข่ง (Peer Mapping)
THAI_PEERS = {
    # ⚡ กลุ่มพลังงานและปิโตรเคมี (Energy & Petrochemicals)
    'PTT': ['PTTEP.BK', 'TOP.BK', 'BCP.BK', 'IRPC.BK', 'BSRC.BK'],
    'PTTEP': ['PTT.BK', 'TOP.BK'], # ต้นน้ำ
    'TOP': ['BCP.BK', 'IRPC.BK', 'SPRC.BK', 'BSRC.BK'], # โรงกลั่น
    
    # 🔌 กลุ่มโรงไฟฟ้า (Power Generation)
    'GULF': ['BGRIM.BK', 'GPSC.BK', 'EA.BK', 'EGCO.BK', 'RATCH.BK'],
    'BGRIM': ['GPSC.BK', 'GULF.BK', 'EGCO.BK'],
    
    # 🏦 กลุ่มธนาคารพาณิชย์ (Banking)
    'KBANK': ['SCB.BK', 'BBL.BK', 'KTB.BK', 'TTB.BK', 'TISCO.BK'],
    'SCB': ['KBANK.BK', 'BBL.BK', 'KTB.BK'],
    
    # 🛒 กลุ่มค้าปลีก (Commerce)
    'CPALL': ['CRC.BK', 'CPAXT.BK', 'BJC.BK'],
    'CRC': ['CPALL.BK', 'HMPRO.BK', 'GLOBAL.BK', 'DOHOME.BK'],
    'HMPRO': ['GLOBAL.BK', 'DOHOME.BK', 'ILM.BK'],
    
    # 📱 กลุ่มสื่อสาร (ICT)
    'ADVANC': ['TRUE.BK', 'INTUCH.BK', 'JAS.BK'],
    'TRUE': ['ADVANC.BK', 'INTUCH.BK'],
    
    # ✈️ กลุ่มท่องเที่ยวและขนส่ง (Transportation & Tourism)
    'AOT': ['BEM.BK', 'BTS.BK', 'AAV.BK', 'BA.BK'],
    'MINT': ['CENTEL.BK', 'ERW.BK', 'SHR.BK'],
    
    # 🏥 กลุ่มการแพทย์ (Healthcare)
    'BDMS': ['BH.BK', 'BCH.BK', 'CHG.BK', 'PR9.BK'],
    'BH': ['BDMS.BK', 'BCH.BK', 'CHG.BK'],
    
    # 💻 กลุ่มชิ้นส่วนอิเล็กทรอนิกส์ (Electronic Components)
    'DELTA': ['KCE.BK', 'HANA.BK', 'CCET.BK', 'SVI.BK'],
    
    # 🏗️ กลุ่มวัสดุก่อสร้างและอสังหาฯ (Construction & Property)
    'SCC': ['SCCC.BK', 'TASCO.BK', 'EPG.BK'],
    'CPN': ['WHA.BK', 'AMATA.BK', 'SPALI.BK', 'AP.BK', 'SIRI.BK']
}

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_comps_data(target_symbol: str) -> pd.DataFrame:
    """ดึงข้อมูลทำตาราง Comps"""
    clean_sym = target_symbol.replace('.BK', '').upper()
    peers = THAI_PEERS.get(clean_sym, ['PTT.BK', 'AOT.BK', 'CPALL.BK']) # Default peers
    tickers = [f"{clean_sym}.BK"] + peers
    
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
                "Div Yield (%)": (info.get('dividendYield', 0) or 0) * 100
            })
        except: pass
    return pd.DataFrame(data)

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_dcf_financials(symbol: str) -> dict:
    """ดึงงบการเงินมาเก็บไว้ใน Cache เพื่อให้เวลาเลื่อน Slider แอปจะได้ไม่กระตุกและไม่โดนแบน API"""
    if not symbol.endswith('.BK'): symbol = f"{symbol}.BK"
    stock = yf.Ticker(symbol)
    
    try:
        is_df = stock.financials
        bs_df = stock.balance_sheet
        cf_df = stock.cashflow
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
# 🌐 ส่วนหน้าจอหลัก (Main UI)
# ==========================================
def main():
    st.set_page_config(page_title="Thai Capital Market ESG Advisor", page_icon="💰", layout="wide")
    
    with st.sidebar:
        st.markdown("### 📊 ฐานข้อมูล ESG (Local)")
        if os.path.exists(ESG_DB_FILE):
            df = pd.read_csv(ESG_DB_FILE)
            st.success(f"✅ โหลดฐานข้อมูลสำเร็จ ({len(df)} บริษัท)")
            st.dataframe(df.head(10), use_container_width=True, hide_index=True)
        else:
            st.warning("⚠️ ไม่พบไฟล์ esg_database.csv")

    st.markdown("<h2 style='text-align: center;'>⚡ Thai Capital Market ESG Stock Advisor</h2>", unsafe_allow_html=True)
        # --- Main Content ---
    st.markdown("<p style='text-align: center;'><b> 💰 ได้รับทุนอุดหนุนการวิจัยและนวัตกรรมจากสำนักงานปลัดกระทรวงการอุดมศึกษา วิทยาศาสตร์ วิจัยและนวัตกรรม และกองทุนส่งเสริมการพัฒนาตลาดทุน 🪙</b></p>", unsafe_allow_html=True)
    st.markdown("---")
    
    query = st.text_input("🔍 พิมพ์ชื่อหุ้นที่ต้องการวิเคราะห์ (เช่น PTT, AOT, KBANK):", placeholder="พิมพ์แค่ชื่อหุ้นภาษาอังกฤษ...")
    
    if query:
        model = get_model()
        if not model:
            st.error("API Key ขัดข้อง")
            return
            
        symbols = re.findall(r'\b[A-Z]{2,6}\b', query.upper())
        symbol = symbols[0] if symbols else query.strip()
        
        # --- แบ่งหน้าจอเป็น 2 แท็บหลัก ---
        tab1, tab2 = st.tabs(["🤖 บทวิเคราะห์ ESG & AI", "📈 Financial Models (DCF & Comps)"])
        
        # --------------------------------------------------
        # แท็บ 1: ระบบ AI อัจฉริยะ (ของเดิม)
        # --------------------------------------------------
        with tab1:
            with st.spinner("🔄 กำลังดึงข้อมูลและประมวลผลด้วย AI..."):
                data = fetch_set_esg_news_info_cached(symbol)
                
            if 'error' in data:
                st.error(data['error'])
            else:
                prompt = f"{THAI_ESG_ADVISOR_PROMPT}\nคำถาม: วิเคราะห์ {symbol}\nข้อมูล:\n{data}"
                
                def stream_response():
                    for chunk in model.stream([HumanMessage(content=prompt)]):
                        text = chunk.content.replace("<think>", "").replace("</think>", "")
                        yield text

                st.markdown(f"### 🤖 บทวิเคราะห์เชิงโครงสร้าง: {symbol}")
                full_analysis = st.write_stream(stream_response)
                
                chart = extract_and_plot_sentiment(full_analysis)
                if chart: st.plotly_chart(chart, use_container_width=True)

        # --------------------------------------------------
        # แท็บ 2: โมเดลการเงินระดับสถาบัน (Comps & Interactive DCF)
        # --------------------------------------------------
        with tab2:
            st.markdown(f"### 📊 Comparable Company Analysis (Comps) - {symbol}")
            comps_df = get_comps_data(symbol)
            if not comps_df.empty:
                # ตกแต่งตาราง Comps ด้วยสีตามมาตรฐาน
                st.dataframe(
                    comps_df.style.highlight_max(subset=['Div Yield (%)'], color='lightgreen')
                                  .highlight_min(subset=['P/E (Forward)', 'EV/EBITDA'], color='lightgreen')
                                  .format({"ราคา (THB)": "{:.2f}", "Market Cap (B)": "{:.2f}", "EV/EBITDA": "{:.2f}", "P/E (Forward)": "{:.2f}"}),
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

if __name__ == "__main__":
    main()
