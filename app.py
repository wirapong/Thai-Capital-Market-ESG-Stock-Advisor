import re
import os
import time
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional

import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pycoingecko import CoinGeckoAPI

import urllib.parse
import xml.etree.ElementTree as ET
import requests

# ==========================================
# ⚙️ ตั้งค่าระบบ (Configuration)
# ==========================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CACHE_TTL = 300  # แคชข้อมูล 5 นาที
ESG_DB_FILE = "esg_database.csv" # ชื่อไฟล์ฐานข้อมูล ESG ของเรา

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
   - ทางเลือกรอง: DF-RNN และ Deep Renewal
   - ข้อห้าม: ไม่แนะนำโมเดลสถิติดั้งเดิม (ARIMA, SARIMA, SARIMAX) สำหรับความสัมพันธ์แบบ Nonlinear
2. การวิเคราะห์ผลกระทบเชิงโครงสร้าง (ESG สู่ ROA/ROCE):
   - บังคับใช้: Panel Regression Models (Fixed/Random Effects)
   - การหาความเป็นเหตุเป็นผล: ใช้ Granger Causality Test เพื่อดูว่า ESG นำร่องพยากรณ์ ROA ได้หรือไม่
3. เทคนิคเตรียมข้อมูล (Data Pipeline):
   - การถ่วงน้ำหนัก: Free-float-adjusted market capitalization
   - ข้อมูลสูญหาย (Missing Values): ใช้ Forward-fill 
   - การแบ่งข้อมูล: ใช้ Rolling-window และ Time-based Split เท่านั้น เพื่อป้องกัน Data Leakage

คำสั่งการทำงาน (Instructions):

กรณีที่ 1: การวิเคราะห์และพยากรณ์ราคาหุ้น (Data-driven)
วิเคราะห์ข้อมูลที่ได้รับอย่างครอบคลุม โดยต้องมีหัวข้อดังนี้:
  • ภาพรวมบริษัทและสถานะในตลาด (SET)
  • การวิเคราะห์อารมณ์ตลาด (News Sentiment Score): 
    - บังคับรูปแบบผลลัพธ์รวม 100% เสมอ: (🟢 Positive XX% | 🟡 Neutral XX% | 🔴 Negative XX%)
    - อธิบายเหตุผลสั้นๆ โดยดึง Keyword สำคัญจากข่าวมาสนับสนุนคะแนน
  • การวิเคราะห์เทคนิคอล (เช่น RSI, MACD, Bollinger Bands)
  • การบูรณาการปัจจัย ESG และความยั่งยืน
  • คาดการณ์ราคาหุ้นในอนาคต (Price Prediction) พร้อมเหตุผลประกอบ

กรณีที่ 2: การให้คำแนะนำด้านการสร้างโมเดล/วิเคราะห์ข้อมูล ESG
ให้ตอบโดยดึงหลักการจาก "Knowledge Base" ทั้ง 3 ข้อด้านบนมาอธิบายเท่านั้น พร้อมเน้นย้ำความสำคัญของ Data Pipeline เสมอ

รูปแบบการตอบกลับ (Formatting & Tone):
• ใช้ภาษาไทยที่เป็นทางการ สละสลวย และอ่านง่าย
• ใช้ Bullet points ในการแจกแจงรายละเอียด
• ท้ายสุดของทุกคำตอบ บังคับให้แสดงข้อความและอ้างอิงรูปแบบด้านล่างนี้เสมอ ห้ามใช้เปเปอร์อื่นเด็ดขาด ใช้รูปแบบการอ้างอิง APA 7th style ทั้งแบบแทรกในเนื้อหา เช่น Detthamrong et al., 2025 หรือ (Detthamrong et al., 2026):

ข้อมูลอ้างอิง
1. Detthamrong, U., Klangbunrueang, R., Chansanam, W., & Dasri, R. (2025). Deep Learning for Sustainable Finance: Robust ESG Index Forecasting in an Emerging Market Context. Sustainability, 18(1), 110. https://doi.org/10.3390/su18010110
2. Detthamrong, U., Klangbunrueang, R., Chansanam, W., & Dasri, R. (2026). The Impact of ESG Performance on Financial Performance: Evidence from Listed Companies in Thailand. Forecasting, 8(1), 14. https://doi.org/10.3390/forecast8010014
"""

# ==========================================
# 📊 ฟังก์ชันจัดการข้อมูล (Data Pipeline)
# ==========================================

def load_esg_data(symbol: str) -> dict:
    """อ่านข้อมูล ESG จากไฟล์ CSV ที่เราเตรียมไว้ (รวดเร็วและไม่โดนบล็อก)"""
    clean_symbol = symbol.upper().replace('.BK', '').strip()
    try:
        if os.path.exists(ESG_DB_FILE):
            df = pd.read_csv(ESG_DB_FILE)
            # แปลงชื่อหุ้นในไฟล์เป็นตัวพิมพ์ใหญ่ทั้งหมดเพื่อป้องกันข้อผิดพลาด
            df['symbol'] = df['symbol'].astype(str).str.upper()
            match = df[df['symbol'] == clean_symbol]
            
            if not match.empty:
                return {
                    "esg_rating": match.iloc[0].get('esg_rating', 'N/A'),
                    "cg_score": match.iloc[0].get('cg_score', 'N/A')
                }
    except Exception as e:
        logger.error(f"Error reading ESG DB: {e}")
        
    return {"esg_rating": "ไม่พบข้อมูลในฐานข้อมูล", "cg_score": "N/A"}

def calculate_technical_patterns(price_history: pd.DataFrame) -> Dict[str, Any]:
    if len(price_history) < 26: return {"error": "ข้อมูลย้อนหลังไม่เพียงพอ"}
    delta = price_history['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    current_rsi = rsi.iloc[-1]
    
    exp1 = price_history['Close'].ewm(span=12, adjust=False).mean()
    exp2 = price_history['Close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    
    sma = price_history['Close'].rolling(window=20).mean()
    std = price_history['Close'].rolling(window=20).std()
    upper_band = sma + (std * 2)
    lower_band = sma - (std * 2)
    current_price = price_history['Close'].iloc[-1]
    
    trend = "ขาขึ้น" if macd.iloc[-1] > signal.iloc[-1] else "ขาลง"
    rsi_status = "Overbought" if current_rsi > 70 else "Oversold" if current_rsi < 30 else "Neutral"
    bb_status = "ใกล้กรอบบน" if current_price > upper_band.iloc[-1] * 0.95 else "ใกล้กรอบล่าง" if current_price < lower_band.iloc[-1] * 1.05 else "อยู่กลางกรอบ"

    return {
        "RSI": round(current_rsi, 2), "RSI_Signal": rsi_status,
        "MACD_Trend": trend, "MACD_Value": round(macd.iloc[-1], 4),
        "BB_Status": bb_status,
        "Upper_Band": round(upper_band.iloc[-1], 2), "Lower_Band": round(lower_band.iloc[-1], 2)
    }

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def fetch_thai_stock_news(symbol: str, limit: int = 5) -> list:
    clean_symbol = symbol.replace('.BK', '').strip()
    query = urllib.parse.quote(f"{clean_symbol} หุ้น")
    url = f"https://news.google.com/rss/search?q={query}&hl=th&gl=TH&ceid=TH:th"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        root = ET.fromstring(response.text)
        news_list = []
        for item in root.findall('.//item')[:limit]:
            title_full = item.find('title').text if item.find('title') is not None else "ไม่มีหัวข้อข่าว"
            pub_date = item.find('pubDate').text if item.find('pubDate') is not None else ""
            if " - " in title_full:
                title, publisher = title_full.rsplit(" - ", 1)
            else:
                title, publisher = title_full, "แหล่งข่าวทั่วไป"
            news_list.append({"title": title.strip(), "publisher": publisher.strip(), "published_at": pub_date})
        return news_list
    except Exception as e:
        logger.warning(f"ไม่สามารถดึงข่าวสำหรับ {symbol} ได้: {e}")
        return []
        
@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def fetch_set_esg_news_info_cached(symbol: str) -> Dict[str, Any]:
    try:
        if not symbol.upper().endswith('.BK'):
            symbol = f"{symbol.upper().strip()}.BK"
            
        # 1. ดึงราคาจาก Yahoo
        stock = yf.Ticker(symbol)
        price_history = stock.history(period='6mo', interval='1d')
        if price_history.empty:
            raise ValueError(f"ไม่มีข้อมูลราคาสำหรับ {symbol}")
            
        time.sleep(0.5) 
        info = stock.info
        thai_news = fetch_thai_stock_news(symbol)
        technical_patterns = calculate_technical_patterns(price_history)
        
        # 2. ดึง ESG จากไฟล์ CSV Local
        clean_symbol = symbol.replace('.BK', '')
        esg_local_data = load_esg_data(clean_symbol)

        return {
            "symbol": clean_symbol,
            "company_name": info.get('longName', symbol),
            "current_price": round(price_history['Close'].iloc[-1], 2),
            "volume": info.get('volume', price_history['Volume'].iloc[-1] if not price_history.empty else 0),
            "technical_analysis": technical_patterns,
            "recent_news": thai_news,
            "esg_metrics": esg_local_data, 
            "price_stats": {
                "current_price": round(price_history['Close'].iloc[-1], 2),
                "min_price_last_year": round(price_history['Close'].min(), 2),
                "max_price_last_year": round(price_history['Close'].max(), 2),
                "average_price_last_year": round(price_history['Close'].mean(), 2),
            }
        }
    except Exception as e:
        logger.error(f"Error in fetch_set_esg: {e}")
        return {"error": f"Unable to fetch SET data for {symbol}. Error: {e}"}

# ==========================================
# 🤖 ระบบ AI และโมเดล
# ==========================================

@st.cache_resource
def get_model():
    try:
        api_key = st.secrets.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            st.warning("⚠️ ไม่พบ GOOGLE_API_KEY กรุณาไปตั้งค่าใน Advanced Settings > Secrets")
            return None
        os.environ["GOOGLE_API_KEY"] = api_key
        return ChatGoogleGenerativeAI(model=MODEL_CONFIG["model"], temperature=MODEL_CONFIG["temperature"])
    except Exception as e:
        st.error(f"Failed to initialize Gemini model: {e}")
        return None

class InvestmentAdvisor:
    def __init__(self, model):
        self.model = model
        
    def extract_symbol(self, query: str) -> str:
        symbols = re.findall(r'\b[A-Z]{2,6}\b', query.upper())
        return symbols[0] if symbols else 'PTT'
    
    def prepare_analysis(self, query: str) -> Dict[str, Any]:
        symbol = self.extract_symbol(query)
        data = fetch_set_esg_news_info_cached(symbol)
        if 'error' in data: return data
        
        data_str = f"""
        บริษัท: {data['company_name']} ({data['symbol']})
        ราคาปัจจุบัน: {data['current_price']} บาท | ปริมาณการซื้อขาย: {data['volume']:,}
        
        รูปแบบทางเทคนิค:
        - RSI: {data['technical_analysis'].get('RSI')} ({data['technical_analysis'].get('RSI_Signal')})
        - แนวโน้ม MACD: {data['technical_analysis'].get('MACD_Trend')}
        - Bollinger Bands: {data['technical_analysis'].get('BB_Status')}
        
        ข้อมูลความยั่งยืน (Local ESG Database):
        - เรตติ้ง ESG: {data['esg_metrics'].get('esg_rating')}
        - คะแนน CG: {data['esg_metrics'].get('cg_score')}
        
        ข่าวสารล่าสุด:
        {[news['title'] for news in data['recent_news']]}
        """
        prompt = f"{THAI_ESG_ADVISOR_PROMPT}\nคำถามของผู้ใช้: {query}\nข้อมูลที่ดึงได้จาก Pipeline:\n{data_str}"
        return {'prompt': prompt, 'data': data, 'type': 'thai_stock'}

# ==========================================
# 📈 ฟังก์ชันวาดกราฟ (UI Components)
# ==========================================

def create_price_chart(data: Dict, asset_type: str) -> go.Figure:
    stats = data.get("price_stats", {})
    source = "Yahoo Finance & Local ESG DB"
        
    tz_bkk = timezone(timedelta(hours=7))
    fetch_time = data.get("analysis_date", datetime.now(tz_bkk).strftime("%d/%m/%Y %H:%M:%S"))
    
    fig = go.Figure(go.Bar(
        x=['Min', 'Avg', 'Current', 'Max'],
        y=[stats.get('min_price_last_year', 0), stats.get('average_price_last_year', 0),
           stats.get('current_price', 0), stats.get('max_price_last_year', 0)],
        marker_color=['#FF4B4B', '#4B8BFF', '#00CC96', '#FFA15A']
    ))
    
    fig.update_layout(
        title={'text': f"Price Range Analysis (1 Year)<br><sup style='color:gray; font-size:12px'>แหล่งข้อมูล: {source} | ข้อมูล ณ เวลา: {fetch_time} (เวลาไทย)</sup>", 'x': 0.0},
        height=380, margin=dict(l=0, r=0, t=65, b=0)  
    )
    return fig

def extract_and_plot_sentiment(analysis_text: str) -> Optional[go.Figure]:
    try:
        pos_match = re.search(r'Positive\s*(\d+)', analysis_text, re.IGNORECASE)
        neu_match = re.search(r'Neutral\s*(\d+)', analysis_text, re.IGNORECASE)
        neg_match = re.search(r'Negative\s*(\d+)', analysis_text, re.IGNORECASE)

        if pos_match and neu_match and neg_match:
            pos_score, neu_score, neg_score = float(pos_match.group(1)), float(neu_match.group(1)), float(neg_match.group(1))
            if pos_score + neu_score + neg_score == 0: return None

            labels, values = ['Positive (เชิงบวก)', 'Neutral (เป็นกลาง)', 'Negative (เชิงลบ)'], [pos_score, neu_score, neg_score]
            colors = ['#00CC96', '#F4C145', '#FF4B4B']

            fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.55, marker_colors=colors, textinfo='percent', textfont_size=14, hoverinfo='label+percent')])
            fig.update_layout(title={'text': "News Sentiment Score", 'y': 0.9, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'}, height=350, margin=dict(t=40, b=20, l=20, r=20), showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5))
            fig.add_annotation(text="📊", x=0.5, y=0.5, font_size=40, showarrow=False)
            return fig
    except Exception as e:
        logger.error(f"Error plotting sentiment chart: {e}")
    return None

# ==========================================
# 🌐 ส่วนหน้าจอหลัก (Main Streamlit App)
# ==========================================

def main():
    st.set_page_config(page_title="Thai Capital Market ESG Advisor", page_icon="💰", layout="wide")
    
    # --- Sidebar: แสดงสถานะฐานข้อมูล ---
    with st.sidebar:
        st.markdown("### 📊 ฐานข้อมูล ESG (Local)")
        st.write("ระบบฐานข้อมูล SET ESG จาก https://www.set.or.th/")
        
        if os.path.exists(ESG_DB_FILE):
            try:
                df = pd.read_csv(ESG_DB_FILE)
                st.success(f"✅ โหลดฐานข้อมูลสำเร็จ ({len(df)} บริษัท)")
                # แสดงตัวอย่างข้อมูลให้ดูนิดหน่อย
                st.dataframe(df.head(10), use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f"❌ อ่านไฟล์ผิดพลาด: {e}")
        else:
            st.warning("⚠️ ไม่พบไฟล์ esg_database.csv ในระบบ")

    # --- Main Content ---
    st.markdown("<h1 style='text-align: center;'>⚡ Thai Capital Market ESG Stock Advisor (Data-Driven)</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'><b> 💰 ได้รับทุนอุดหนุนการวิจัยและนวัตกรรมจากสำนักงานปลัดกระทรวงการอุดมศึกษา วิทยาศาสตร์ วิจัยและนวัตกรรม และกองทุนส่งเสริมการพัฒนาตลาดทุน 🪙</b></p>", unsafe_allow_html=True)
    st.markdown("---")
    
    query = st.text_input("🔍 พิมพ์คำถามของคุณ:", placeholder="เช่น 'วิเคราะห์หุ้น AOT ด้านกราฟเทคนิคและ ESG' หรือ 'ภาพรวม PTT เป็นอย่างไร'")
    
    if query:
        model = get_model()
        if not model:
            return
            
        advisor = InvestmentAdvisor(model)
        
        with st.spinner("🔄 กำลังรัน Data Pipeline เพื่อดึงข้อมูล..."):
            result = advisor.prepare_analysis(query)
            
        if 'error' in result:
            st.error(result['error'])
            return
            
        data = result['data']
        # 1. แสดงกราฟแท่งราคาทันที
        st.plotly_chart(create_price_chart(data, result['type']), use_container_width=True)
        
        st.markdown("### 🤖 บทวิเคราะห์และพยากรณ์เชิงโครงสร้าง (AI)")
        
        def stream_response():
            is_thinking = False
            for chunk in advisor.model.stream([HumanMessage(content=result['prompt'])]):
                text = chunk.content
                if "<think>" in text: is_thinking = True
                if "</think>" in text: 
                    is_thinking = False
                    text = text.replace("</think>", "")
                    continue
                if not is_thinking:
                    yield text

        try:
            # 2. ให้ AI พิมพ์บทวิเคราะห์ (Streaming)
            full_analysis = st.write_stream(stream_response)
            
            # 3. พล็อตกราฟโดนัทอารมณ์ข่าว
            sentiment_chart = extract_and_plot_sentiment(full_analysis)
            
            if sentiment_chart:
                st.markdown("---")
                st.markdown("<h3 style='text-align: center;'>📊 สัดส่วนอารมณ์ตลาดจากข่าวสาร (Market Sentiment)</h3>", unsafe_allow_html=True)
                col_spacer1, col_chart, col_spacer2 = st.columns([1, 2, 1])
                with col_chart:
                    st.plotly_chart(sentiment_chart, use_container_width=True)
                    
        except Exception as e:
            st.error(f"⏳ เกิดข้อผิดพลาดในการประมวลผลของ AI: {e}")

if __name__ == "__main__":
    main()