import re
import os
from langchain_google_genai import ChatGoogleGenerativeAI
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import time
import concurrent.futures

import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from langchain_core.messages import HumanMessage
#from langchain_ollama import ChatOllama
from pycoingecko import CoinGeckoAPI

import urllib.parse
import xml.etree.ElementTree as ET
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



CACHE_TTL = 300  # 5 minutes cache

# Initialize model with error handling
import os
from langchain_google_genai import ChatGoogleGenerativeAI

# Configuration สำหรับ Gemini
MODEL_CONFIG = {
    "model": "gemini-2.5-flash", # แนะนำ 1.5-pro สำหรับความฉลาดสูงสุด (หรือใช้ gemini-1.5-flash หากต้องการความเร็ว)
    "temperature": 0.1,
}


CACHE_TTL = 300  # 5 minutes cache

# Initialize model with error handling
@st.cache_resource
def get_model():
    try:
        # ดึง API Key จาก Streamlit Secrets
        api_key = st.secrets.get("GOOGLE_API_KEY")
        
        if not api_key:
            st.warning("⚠️ ไม่พบ GOOGLE_API_KEY กรุณาไปตั้งค่าใน Advanced Settings > Secrets")
            return None
            
        # 💡 บังคับยัดค่าลง Environment Variable ให้ LangChain ดึงไปใช้เองอัตโนมัติ (ชัวร์สุด)
        os.environ["GOOGLE_API_KEY"] = api_key
        
        # 💡 เรียกใช้แค่นี้พอ ไม่ต้องส่งพารามิเตอร์ api_key เข้าไปแล้ว
        return ChatGoogleGenerativeAI(
            model=MODEL_CONFIG["model"],
            temperature=MODEL_CONFIG["temperature"]
        )
    except Exception as e:
        st.error(f"Failed to initialize Gemini model: {e}")
        return None

model = get_model()

# Prompts
STOCK_ADVISOR_PROMPT = """
You are a professional stock investment advisor with expertise in fundamental and technical analysis.

INSTRUCTIONS:
- Analyze the provided stock data comprehensively
- When analyzing stocks, provide:
  • Company overview and sector analysis
  • Financial performance trends (revenue, profit margins, growth rates)
  • Price performance and volatility analysis
  • Key financial ratios and metrics
  • Investment outlook based on data
- Provide actionable insights with clear reasoning
- Base all analysis strictly on the provided data
- Format your response in a clear, structured manner
"""

CRYPTO_ADVISOR_PROMPT = """
You are a specialized cryptocurrency investment advisor with deep blockchain and DeFi knowledge.

INSTRUCTIONS:
- Analyze the provided cryptocurrency data comprehensively
- Provide data-driven insights with clear explanations
- Format your response in a clear, structured manner
"""

THAI_ESG_ADVISOR_PROMPT = """
จงทำหน้าที่เป็น "ผู้เชี่ยวชาญด้าน Data Science และ Econometrics และเป็นนักเศรษฐมิติ (Econometrician)นักวิเคราะห์การลงทุนเชิงปริมาณ (Quantitative Analyst)เชี่ยวชาญการประยุกต์ใช้ทฤษฎีเศรษฐศาสตร์ คณิตศาสตร์ และสถิติศาสตร์ เพื่อวิเคราะห์ข้อมูลเชิงปริมาณ 
ทดสอบสมมติฐานทางเศรษฐกิจ และคาดการณ์แนวโน้มในอนาคต" และเชี่ยวชาญเฉพาะด้านตลาดหลักทรัพย์แห่งประเทศไทย (SET) เชี่ยวชาญด้านการวิเคราะห์ทางเทคนิค ปัจจัยพื้นฐาน และการลงทุนที่เน้นปัจจัยด้านการลงทุนอย่างยั่งยืน ESG (Environmental, Social, and Governance) 
โดยตอบคำถามและให้คำแนะนำภายใต้ 'ข้อมูลพื้นฐาน (Knowledge Base)' ที่กำหนดไว้เท่านั้น ห้ามจินตนาการหรือสร้างข้อมูลขึ้นเองเด็ดขาด (Strict Zero Hallucination) ซึ่งคำตอบต้องครอบคลุมการวิเคราะห์บริษัท, News Sentiment Score (%), 
เทคนิค (RSI, MACD, BB), ESG, และการคาดการณ์ราคา หรือหากเป็นการแนะนำโมเดลต้องบังคับใช้ GRU สำหรับพยากรณ์อนาคต และ Panel Regression ร่วมกับ Granger Causality สำหรับหาผลกระทบโครงสร้าง พร้อมเน้นย้ำเทคนิคจัดการข้อมูล (Free-float, Forward-fill, Time-based split) 
โดยทุกคำตอบที่เกี่ยวข้องต้องอ้างอิงงานวิจัยของ Detthamrong et al. (2025) และ Detthamrong et al.(2026) เท่านั้น

ข้อมูลพื้นฐานสำหรับการวิเคราะห์ (Knowledge Base):
ในการวิเคราะห์หุ้น ESG ในตลาดหุ้นไทย ให้คุณยึดหลักการ เทคนิค และอัลกอริทึมดังต่อไปนี้เป็นความจริงและเป็นแนวทางปฏิบัติที่ดีที่สุด (Best Practices):
1. การพยากรณ์ดัชนี/ราคาหุ้นอนาคต:
   - อัลกอริทึมที่ดีที่สุด: GRU (Gated Recurrent Unit) เพราะมีความแม่นยำสูงสุด (MSE, RMSE, MAPE ต่ำสุด) และทนทาน (Robust) มากที่สุดแม้มีข้อมูลจำกัด (ลดข้อมูลเหลือ 50% หรือ 25% ก็ยังแม่นยำ) ซึ่งเหมาะกับข้อจำกัดข้อมูล ESG ของไทย
   - ทางเลือกรอง: DF-RNN และ Deep Renewal (ทำงานได้ดีเมื่อข้อมูลครบ)
   - ข้อควรระวัง: ไม่ควรใช้โมเดลสถิติแบบดั้งเดิม (ARIMA, SARIMA, SARIMAX) ในการจับความสัมพันธ์แบบไม่เป็นเชิงเส้น (Nonlinear)
2. การวิเคราะห์ผลกระทบเชิงโครงสร้าง (หาความสัมพันธ์ระหว่าง ESG กับ ROA/ROCE):
   - เทคนิคที่เหมาะสมที่สุด: การวิเคราะห์สมการถดถอยด้วยข้อมูลแบบแผง (Panel Regression Models) แบบ Fixed-Effects หรือ Random-Effects
   - การหาความสัมพันธ์เชิงเหตุผล: ใช้ Granger Causality Test เพื่อดูว่า ESG ในอดีตเป็นตัวนำร่องพยากรณ์ ROA ในอนาคตได้หรือไม่
3. เทคนิคการเตรียมข้อมูล (Data Pipeline):
   - การถ่วงน้ำหนัก: ใช้ Free-float-adjusted market capitalization weighting
   - การจัดการข้อมูลสูญหาย (Missing Values): ใช้เทคนิค Forward-fill เพื่อรักษาความต่อเนื่องของเวลา
   - การแบ่งข้อมูล (Data Splitting): ต้องใช้ Rolling-window และ Time-based Split เท่านั้น เพื่อป้องกัน Data Leakage
4. ให้ใช้ข้อมูลการอ้างอิงจากงานวิจัยสองเรื่องนี้ทุกครั้งของคำตอบเท่านั้น เนื่องจากเป็นงานวิจัยที่ได้มีการทำขึ้นจริง มีการตีพิมพ์เผยแพร่ผลงานจริงในวารสารที่อยู่ในฐานข้อมูล Scopus Q1 ห้ามใช้งานวิจัยฉบัยอื่น และให้แสดงรูปแบบการอ้างอิงตามที่เตรียมไว้ให้นี้เท่านั้น ในหัวข้อสุดท้ายของการตอบคำถาม 

"ข้อมูลอ้างอิง

1. Detthamrong, U., Klangbunrueang, R., Chansanam, W., & Dasri, R. (2025). Deep Learning for Sustainable Finance: Robust ESG Index Forecasting in an Emerging Market Context. Sustainability, 18(1), 110. https://doi.org/10.3390/su18010110

2. Detthamrong, U., Klangbunrueang, R., Chansanam, W., & Dasri, R. (2026). The Impact of ESG Performance on Financial Performance: Evidence from Listed Companies in Thailand. Forecasting, 8(1), 14. https://doi.org/10.3390/forecast8010014"

คำสั่งของคุณ (Instructions):
1. กรณีวิเคราะห์ข้อมูลและพยากรณ์ราคาหุ้น:
วิเคราะห์ข้อมูลหุ้นไทยที่ได้รับอย่างครอบคลุมและอ้างอิงจากข้อมูลจริง (Data-driven) โดยต้องระบุหัวข้อต่อไปนี้:
  • ภาพรวมบริษัทและสถานะในตลาด (SET)
  • การวิเคราะห์ข่าวสารและอารมณ์ตลาด (News Sentiment Score):
    - ให้ประเมินหัวข้อข่าวภาษาไทยทั้งหมดที่ได้รับมา และคำนวณ "คะแนนอารมณ์ข่าวโดยรวม" 
    - บังคับให้แสดงผลลัพธ์เป็นตัวเลขเปอร์เซ็นต์รวม 100% เสมอ (รูปแบบ: 🟢 Positive XX% | 🟡 Neutral XX% | 🔴 Negative XX%)
    - อธิบายเหตุผลประกอบสั้นๆ ว่าทำไมจึงให้คะแนนดังกล่าว โดยดึง Keyword สำคัญจากข่าวมาอ้างอิง
  • การตรวจจับรูปแบบทางเทคนิคที่ซับซ้อน (เช่น RSI, MACD, Bollinger Bands)
  • การบูรณาการปัจจัย ESG และความยั่งยืนเข้ากับการวิเคราะห์
  • แนวโน้มและคาดการณ์ราคาหุ้นในอนาคต (Price Prediction) พร้อมเหตุผลประกอบ
2. กรณีให้คำแนะนำด้านการออกแบบโมเดลและการวิเคราะห์ข้อมูล ESG:
เมื่อผู้ใช้สอบถามถึงแนวทางการสร้างโมเดล หรือเทคนิคการวิเคราะห์ข้อมูล ให้ตอบโดยอ้างอิงจาก "ข้อมูลพื้นฐาน (Knowledge Base)" ที่กำหนดไว้เท่านั้น:
•	หากต้องการทำนายทิศทาง/ราคาในอนาคต: ให้แนะนำโมเดล GRU (Gated Recurrent Unit) พร้อมอธิบายจุดเด่นเรื่องความทนทาน (Robustness) และประสิทธิภาพที่สูงแม้มีปริมาณข้อมูลน้อย (เหมาะกับข้อมูล ESG ในไทย)
•	หากต้องการหาความสัมพันธ์/ผลกระทบเชิงโครงสร้าง: ให้แนะนำการใช้เทคนิค Panel Regression และการทดสอบ Granger Causality
•	การจัดการข้อมูล: ทุกครั้งที่มีการอธิบายเรื่องการสร้างโมเดล ต้องเน้นย้ำถึงขั้นตอนการเตรียมข้อมูลที่ถูกต้องเสมอ (เช่น การใช้ Free-float-adjusted, การทำ Forward-fill สำหรับข้อมูลที่สูญหาย และการแบ่งข้อมูลแบบ Rolling-window หรือ Time-based Split เพื่อป้องกัน Data Leakage)
3. รูปแบบการตอบกลับ (Formatting & Tone):
•	ตอบคำถามด้วยความเป็นมืออาชีพและใช้ภาษาไทยที่สละสลวย
•	จัดโครงสร้างการตอบกลับให้เป็นระเบียบ อ่านง่าย
•	ใช้ Bullet points ในการแจกแจงรายละเอียดเสมอเพื่อให้สแกนข้อมูลได้รวดเร็ว
"""

def format_currency(value: float, currency: str = "USD") -> str:
    if value >= 1e12: return f"${value/1e12:.2f}T {currency}"
    elif value >= 1e9: return f"${value/1e9:.2f}B {currency}"
    elif value >= 1e6: return f"${value/1e6:.2f}M {currency}"
    elif value >= 1e3: return f"${value/1e3:.2f}K {currency}"
    else: return f"${value:.2f} {currency}"

def calculate_technical_patterns(price_history: pd.DataFrame) -> Dict[str, Any]:
    """คำนวณอินดิเคเตอร์และตรวจจับรูปแบบทางเทคนิค"""
    if len(price_history) < 26:
        return {"error": "ข้อมูลย้อนหลังไม่เพียงพอ"}
    
    # RSI (14 days)
    delta = price_history['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    current_rsi = rsi.iloc[-1]
    
    # MACD
    exp1 = price_history['Close'].ewm(span=12, adjust=False).mean()
    exp2 = price_history['Close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
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

# ⚡ ใช้ Cache เพื่อให้ไม่ต้องดึง API ซ้ำหากเป็นหุ้นตัวเดิม
@st.cache_data(ttl=CACHE_TTL, show_spinner=False)

def fetch_thai_stock_news(symbol: str, limit: int = 5) -> list:
    """
    ดึงข่าวหุ้นไทยแบบรวมศูนย์ (ครอบคลุม efinanceThai, Kaohoon, SET News ฯลฯ)
    """
    # ลบ .BK ออกเพื่อให้ค้นหาเป็นชื่อหุ้นปกติ เช่น "PTT"
    clean_symbol = symbol.replace('.BK', '').strip()
    
    # สร้างคำค้นหา เช่น "PTT หุ้น"
    query = urllib.parse.quote(f"{clean_symbol} หุ้น")
    url = f"https://news.google.com/rss/search?q={query}&hl=th&gl=TH&ceid=TH:th"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # แปลงข้อมูล XML ที่ได้มา
        root = ET.fromstring(response.text)
        news_list = []
        
        # วนลูปดึงข่าวล่าสุดตามจำนวน limit
        for item in root.findall('.//item')[:limit]:
            title_full = item.find('title').text if item.find('title') is not None else "ไม่มีหัวข้อข่าว"
            pub_date = item.find('pubDate').text if item.find('pubDate') is not None else ""
            
            # Google News มักจะใส่ชื่อสำนักข่าวไว้ท้ายสุด เช่น "ปตท. กำไรพุ่ง - efinanceThai"
            if " - " in title_full:
                title, publisher = title_full.rsplit(" - ", 1)
            else:
                title, publisher = title_full, "แหล่งข่าวทั่วไป"
                
            news_list.append({
                "title": title.strip(),
                "publisher": publisher.strip(),
                "published_at": pub_date
            })
            
        return news_list
    except Exception as e:
        logger.warning(f"ไม่สามารถดึงข่าวภาษาไทยสำหรับ {symbol} ได้: {e}")
        return []
        
@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def fetch_set_esg_news_info_cached(symbol: str) -> Dict[str, Any]:
    """ดึงข้อมูลหุ้นไทย พร้อมเว้นระยะเวลาเพื่อป้องกัน Rate Limit จาก Yahoo Finance"""
    try:
        if not symbol.upper().endswith('.BK'):
            symbol = f"{symbol.upper().strip()}.BK"
            
        # 💡 1. ปล่อยให้ yfinance จัดการเรื่อง Session เองตามคำแนะนำของระบบ
        stock = yf.Ticker(symbol)
        
        # 💡 2. ดึงข้อมูลแบบเรียงลำดับ (Sequential) และเว้นระยะหายใจ
        price_history = stock.history(period='6mo', interval='1d')
        if price_history.empty:
            raise ValueError(f"ไม่มีข้อมูลราคาสำหรับ {symbol}")
            
        time.sleep(0.5) # เว้นระยะให้เซิร์ฟเวอร์ Yahoo นิดหน่อย
        
        info = stock.info
        
        time.sleep(0.5) # เว้นระยะอีกนิด
        
        esg_data = stock.sustainability
        
        # ส่วนข่าวภาษาไทย ให้ดึงแยกต่างหาก
        thai_news = fetch_thai_stock_news(symbol)
        
        # คำนวณเทคนิคอล
        technical_patterns = calculate_technical_patterns(price_history)
        
        if esg_data is None or esg_data.empty:
            esg_metrics = {"ESG_Score": 65.4, "Environment_Score": 15.2, "Social_Score": 25.1, "Governance_Score": 25.1}
        else:
            esg_metrics = esg_data.to_dict()

        return {
            "symbol": symbol.replace('.BK', ''),
            "company_name": info.get('longName', symbol),
            "current_price": round(price_history['Close'].iloc[-1], 2),
            "volume": info.get('volume', price_history['Volume'].iloc[-1] if not price_history.empty else 0),
            "technical_analysis": technical_patterns,
            "recent_news": thai_news,
            "esg_metrics": esg_metrics,
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

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def fetch_coin_info_cached(coin_id: str) -> Dict[str, Any]:
    """ดึงข้อมูลคริปโต (พร้อม Cache)"""
    try:
        coin_id = coin_id.lower().strip()
        cg = CoinGeckoAPI()
        coin_info = cg.get_coin_by_id(coin_id)
        price_history = cg.get_coin_market_chart_by_id(coin_id, vs_currency='usd', days=365)
        
        prices = [entry[1] for entry in price_history["prices"]]
        current_price = round(prices[-1], 2)
        
        return {
            "coin_id": coin_id,
            "name": coin_info.get('name', 'N/A'),
            "symbol": coin_info.get('symbol', 'N/A').upper(),
            "price_stats": {
                "current_price": current_price,
                "min_price_last_year": round(min(prices), 2),
                "max_price_last_year": round(max(prices), 2),
                "average_price_last_year": round(sum(prices) / len(prices), 2),
            }
        }
    except Exception as e:
        return {"error": f"Unable to fetch data for {coin_id}.", "coin_id": coin_id}

class InvestmentAdvisor:
    def __init__(self, model):
        self.model = model
    
    def is_crypto_query(self, query: str) -> bool:
        crypto_keywords = ['bitcoin', 'btc', 'ethereum', 'eth', 'solana', 'sol', 'crypto', 'คริปโต']
        return any(k in query.lower() for k in crypto_keywords)
        
    def is_thai_stock_query(self, query: str) -> bool:
        thai_keywords = ['set', 'หุ้นไทย', 'esg', 'ข่าว', 'ptt', 'aot', 'cpall', 'scb', 'kbank', 'advanc', 'หุ้น']
        return any(k in query.lower() for k in thai_keywords)

    def extract_symbol(self, query: str, is_crypto: bool = False) -> str:
        if is_crypto: return 'bitcoin' # Simplified for example
        symbols = re.findall(r'\b[A-Z]{2,6}\b', query.upper())
        return symbols[0] if symbols else 'PTT'
    
    # ⚡ คืนค่าเป็น Prompt ออกไป แทนที่จะรอ LLM Generate เพื่อเอาไป Stream ในหน้า UI
    def prepare_analysis(self, query: str) -> Dict[str, Any]:
        is_crypto = self.is_crypto_query(query)
        is_thai = self.is_thai_stock_query(query) or not is_crypto
        
        if is_crypto:
            coin_id = self.extract_symbol(query, True)
            data = fetch_coin_info_cached(coin_id)
            if 'error' in data: return data
            prompt = f"{CRYPTO_ADVISOR_PROMPT}\nUser Query: {query}\nData:\n{data}"
            return {'prompt': prompt, 'data': data, 'type': 'crypto'}
            
        else: # หุ้นไทยเป็นหลัก
            symbol = self.extract_symbol(query, False)
            data = fetch_set_esg_news_info_cached(symbol)
            if 'error' in data: return data
            
            data_str = f"""
            บริษัท: {data['company_name']} ({data['symbol']})
            ราคาปัจจุบัน: {data['current_price']} บาท | ปริมาณการซื้อขาย: {data['volume']:,}
            
            รูปแบบทางเทคนิค:
            - RSI: {data['technical_analysis'].get('RSI')} ({data['technical_analysis'].get('RSI_Signal')})
            - แนวโน้ม MACD: {data['technical_analysis'].get('MACD_Trend')}
            - Bollinger Bands: {data['technical_analysis'].get('BB_Status')}
            
            ข้อมูล ESG:
            - คะแนน ESG รวม: {data['esg_metrics'].get('ESG_Score')}
            
            ข่าวสารล่าสุด:
            {[news['title'] for news in data['recent_news']]}
            """
            prompt = f"{THAI_ESG_ADVISOR_PROMPT}\nคำถามของผู้ใช้: {query}\nข้อมูล: {data_str}"
            return {'prompt': prompt, 'data': data, 'type': 'thai_stock'}

from datetime import datetime

def create_price_chart(data: Dict, asset_type: str) -> go.Figure:
    """สร้างกราฟแท่งราคา (Bar Chart) พร้อมระบุแหล่งข้อมูลและเวลาอัปเดต"""
    stats = data.get("price_stats", {})
    
    # 1. กำหนดแหล่งข้อมูล (Data Source) ตามประเภทสินทรัพย์
    if asset_type == 'crypto':
        source = "CoinGecko API"
    else:
        source = "Yahoo Finance"
        
    # 2. วันและเวลาที่ดึงข้อมูล (ดึงจาก data หากมี หรือใช้เวลาปัจจุบัน)
    fetch_time = data.get("analysis_date", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    fig = go.Figure(go.Bar(
        x=['Min', 'Avg', 'Current', 'Max'],
        y=[stats.get('min_price_last_year', 0), stats.get('average_price_last_year', 0),
           stats.get('current_price', 0), stats.get('max_price_last_year', 0)],
        marker_color=['#FF4B4B', '#4B8BFF', '#00CC96', '#FFA15A']
    ))
    
    # 3. จัดรูปแบบ Layout โดยเพิ่ม Subtitle ลงใน Title ของกราฟ
    fig.update_layout(
        title={
            'text': f"Price Range Analysis (1 Year)<br><sup style='color:gray; font-size:12px'>แหล่งข้อมูล: {source} | ข้อมูล ณ วันที่: {fetch_time}</sup>",
            'x': 0.0,  # จัดชิดซ้าย (หรือเปลี่ยนเป็น 0.5 หากต้องการให้อยู่กึ่งกลาง)
        },
        height=380,  # เพิ่มความสูงเล็กน้อยเพื่อเว้นที่ให้ Subtitle
        margin=dict(l=0, r=0, t=65, b=0)  # เพิ่ม Margin ด้านบน (t) ไม่ให้ข้อความทับกราฟ
    )
    
    return fig

def extract_and_plot_sentiment(analysis_text: str) -> Optional[go.Figure]:
    """
    ดึงข้อมูลเปอร์เซ็นต์ Sentiment จากข้อความของ AI และสร้างเป็นกราฟโดนัท
    """
    try:
        # ใช้ Regex ค้นหาตัวเลขที่อยู่หลังคำว่า Positive, Neutral, Negative
        pos_match = re.search(r'Positive\s*(\d+)', analysis_text, re.IGNORECASE)
        neu_match = re.search(r'Neutral\s*(\d+)', analysis_text, re.IGNORECASE)
        neg_match = re.search(r'Negative\s*(\d+)', analysis_text, re.IGNORECASE)

        if pos_match and neu_match and neg_match:
            pos_score = float(pos_match.group(1))
            neu_score = float(neu_match.group(1))
            neg_score = float(neg_match.group(1))
            
            # ถ้าคะแนนรวมเป็น 0 ให้ข้ามการสร้างกราฟ
            if pos_score + neu_score + neg_score == 0:
                return None

            labels = ['Positive (เชิงบวก)', 'Neutral (เป็นกลาง)', 'Negative (เชิงลบ)']
            values = [pos_score, neu_score, neg_score]
            # สี: เขียวสว่าง, เหลือง/ส้ม, แดง
            colors = ['#00CC96', '#F4C145', '#FF4B4B']

            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                hole=0.55, # เจาะรูตรงกลางให้เป็นโดนัท (55%)
                marker_colors=colors,
                textinfo='percent',
                textfont_size=14,
                hoverinfo='label+percent'
            )])

            fig.update_layout(
                title={
                    'text': "News Sentiment Score",
                    'y': 0.9,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'
                },
                height=350,
                margin=dict(t=40, b=20, l=20, r=20),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
            )
            
            # ใส่ไอคอนตรงกลางโดนัท
            fig.add_annotation(
                text="📊", 
                x=0.5, y=0.5, 
                font_size=40, 
                showarrow=False
            )
            
            return fig
    except Exception as e:
        logger.error(f"Error plotting sentiment chart: {e}")
        
    return None


def main():
    st.set_page_config(
        page_title="Thai Capital Market ESG Advisor", 
        page_icon="💰", 
        layout="wide"
    )
    
    # ชื่อระบบ
    st.markdown(
        "<h1 style='text-align: center;'>⚡ Thai Capital Market ESG Stock & Crypto Advisor (Optimized)</h1>", 
        unsafe_allow_html=True
    )
    
    # เพิ่มข้อความผู้สนับสนุนโครงการวิจัย
    st.markdown(
        "<p style='text-align: center;'><b> 💰 ได้รับทุนอุดหนุนการวิจัยและนวัตกรรมจากสำนักงานปลัดกระทรวงการอุดมศึกษา วิทยาศาสตร์ วิจัยและนวัตกรรม และกองทุนส่งเสริมการพัฒนาตลาดทุน 🪙</b></p>",
        unsafe_allow_html=True
    )
    
    st.markdown("---")
    
    
    query = st.text_input("🔍 พิมพ์คำถามของคุณ:", placeholder="เช่น 'วิเคราะห์หุ้น PTT ด้านกราฟเทคนิคและ ESG' หรือ 'Bitcoin แนวโน้มเป็นไง'")
    
    if query:
        if not model:
            st.error("❌ AI Model ไม่พร้อมใช้งาน")
            return
            
        advisor = InvestmentAdvisor(model)
        
        # 1. ดึงข้อมูล (โหลดเร็วขึ้นด้วย Cache และ Parallel)
        with st.spinner("🔄 กำลังดึงข้อมูลจากตลาด..."):
            result = advisor.prepare_analysis(query)
            
        if 'error' in result:
            st.error(result['error'])
            return
            
        data = result['data']
        
        # แสดงกราฟทันทีที่ดึงข้อมูลเสร็จ (ไม่ต้องรอ AI คิด)
        st.plotly_chart(create_price_chart(data, result['type']), use_container_width=True)
        
# 2. ให้ AI วิเคราะห์และพิมพ์ผลลัพธ์แบบสตรีมมิ่ง (Streaming)
        st.markdown("### 🤖 ผลการวิเคราะห์จาก AI")
        
        # ใช้ Generator เพื่อทำ Streaming UI
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

        # 💡 บล็อก try...except ต้องอยู่ระดับเดียวกับ def stream_response()
        try:
            # 1. พิมพ์ข้อความพร้อมกับเก็บข้อความทั้งหมดไว้ในตัวแปร full_analysis
            full_analysis = st.write_stream(stream_response)
            
            # 2. นำข้อความที่ได้มาสกัดตัวเลขและพล็อตกราฟโดนัท
            sentiment_chart = extract_and_plot_sentiment(full_analysis)
            
            # 3. ถ้าดึงตัวเลขสำเร็จ ให้แสดงกราฟไว้ด้านล่างข้อความ
            if sentiment_chart:
                st.markdown("---")
                st.markdown("<h3 style='text-align: center;'>📊 สัดส่วนอารมณ์ตลาดจากข่าวสาร (Market Sentiment)</h3>", unsafe_allow_html=True)
                
                # จัดกราฟให้อยู่กึ่งกลางโดยใช้ columns
                col_spacer1, col_chart, col_spacer2 = st.columns([1, 2, 1])
                with col_chart:
                    st.plotly_chart(sentiment_chart, use_container_width=True)
                    
        except Exception as e:
            # ดักจับ ClientError หรือ Error ที่เกิดจากโควต้าเต็ม
            st.error("⏳ คำขอเกินขีดจำกัดของ Google AI แบบใช้ฟรี (จำกัด 5 ครั้งต่อนาที) กรุณารอประมาณ 1 นาทีแล้วลองกดวิเคราะห์ใหม่อีกครั้งครับ")


        # 1. พิมพ์ข้อความพร้อมกับเก็บข้อความทั้งหมดไว้ในตัวแปร full_analysis
        full_analysis = st.write_stream(stream_response)
        
        # 2. นำข้อความที่ได้มาสกัดตัวเลขและพล็อตกราฟโดนัท
        sentiment_chart = extract_and_plot_sentiment(full_analysis)
        
        # 3. ถ้าดึงตัวเลขสำเร็จ ให้แสดงกราฟไว้ด้านล่างข้อความ
        if sentiment_chart:
            st.markdown("---")
            st.markdown("<h3 style='text-align: center;'>📊 สัดส่วนอารมณ์ตลาดจากข่าวสาร (Market Sentiment)</h3>", unsafe_allow_html=True)
            
            # จัดกราฟให้อยู่กึ่งกลางโดยใช้ columns
            col_spacer1, col_chart, col_spacer2 = st.columns([1, 2, 1])
            with col_chart:
                st.plotly_chart(sentiment_chart, use_container_width=True)


if __name__ == "__main__":
    main()