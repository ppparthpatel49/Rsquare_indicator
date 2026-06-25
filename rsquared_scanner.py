import os
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import datetime
import warnings
from scipy.stats import linregress

warnings.filterwarnings('ignore')

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
ACCOUNT_SIZE = 100000  
RISK_PER_TRADE = 0.01  

nse_stocks = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "HINDUNILVR.NS", 
    "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS", "L&T.NS", "BAJFINANCE.NS", 
    "HCLTECH.NS", "ASIANPAINT.NS", "AXISBANK.NS", "MARUTI.NS", "SUNPHARMA.NS", 
    "TITAN.NS", "DMART.NS", "ULTRACEMCO.NS", "BAJAJFINSV.NS", "WIPRO.NS", "M&M.NS", 
    "TATASTEEL.NS", "ADANIENT.NS", "POWERGRID.NS", "NTPC.NS", "TATAMOTORS.NS",
    "TECHM.NS", "NESTLEIND.NS", "ONGC.NS", "GRASIM.NS", "HINDALCO.NS", "JSWSTEEL.NS",
    "TRENT.NS", "BHEL.NS", "HAL.NS", "BEL.NS", "DLF.NS", "PFC.NS", "RECLTD.NS", "ZOMATO.NS"
]

def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try: requests.post(url, json=payload, timeout=10)
    except: pass

def calc_atr(df, period=20):
    ranges = pd.concat([df['High'] - df['Low'], np.abs(df['High'] - df['Close'].shift()), np.abs(df['Low'] - df['Close'].shift())], axis=1)
    return ranges.max(axis=1).rolling(period).mean()

triggered_stocks = []

for stock in nse_stocks:
    try:
        df = yf.download(stock, period="1y", progress=False)
        if len(df) < 150: continue
        if isinstance(df.columns, pd.MultiIndex): df.columns = [c[0] for c in df.columns]
            
        close_90 = df['Close'].iloc[-90:].values
        x_time = np.arange(90)
        slope, intercept, r_value, p_value, std_err = linregress(x_time, close_90)
        r_squared = r_value ** 2
        
        sma_90 = df['Close'].rolling(90).mean().iloc[-1]
        is_smooth_trend = (r_squared > 0.50) and (df['Close'].iloc[-1] > sma_90)

        df['Daily_Delta'] = np.where(df['Close'] > df['Open'], df['Volume'], np.where(df['Close'] < df['Open'], -df['Volume'], 0))
        df['CVD'] = df['Daily_Delta'].cumsum()
        df['CVD_SMA'] = df['CVD'].rolling(50).mean()
        
        cvd_slope = df['CVD_SMA'].iloc[-1] - df['CVD_SMA'].iloc[-10]
        is_accumulated = (df['CVD'].iloc[-1] > df['CVD_SMA'].iloc[-1]) and (cvd_slope > 0)

        df['SMA_20'] = df['Close'].rolling(20).mean()
        df['STD_20'] = df['Close'].rolling(20).std()
        df['BBW'] = ((df['SMA_20'] + (df['STD_20'] * 2)) - (df['SMA_20'] - (df['STD_20'] * 2))) / df['SMA_20']
        
        bbw_lowest_120 = df['BBW'].rolling(120).min()
        is_squeeze = df['BBW'].iloc[-5:].min() <= (bbw_lowest_120.iloc[-1] * 1.2)

        highest_50 = df['High'].shift(1).rolling(50).max().iloc[-1]
        c = df['Close'].iloc[-1]
        is_breakout = c > highest_50
        
        if is_smooth_trend and is_accumulated and is_squeeze and is_breakout:
            atr = calc_atr(df, 20).iloc[-1]
            stop_loss = c - (atr * 2.5)
            
            risk_amt = ACCOUNT_SIZE * RISK_PER_TRADE
            risk_per_share = c - stop_loss
            shares = int(risk_amt / risk_per_share) if risk_per_share > 0 else 0
            
            triggered_stocks.append({
                "ticker": stock.replace('.NS', ''),
                "entry": c,
                "stop": stop_loss,
                "shares": shares,
                "r2": r_squared
            })
            
    except Exception as e:
        continue

date_today = datetime.datetime.now().strftime("%Y-%m-%d")
if triggered_stocks:
    msg = f"🟢 *R-SQUARED SYSTEMATIC BREAKOUT* ({date_today})\n\n"
    for t in triggered_stocks:
        msg += f"📊 *{t['ticker']}* (R²: {t['r2']:.2f})\n🔹 Entry: ₹{t['entry']:.2f}\n🔴 Stop: ₹{t['stop']:.2f}\n📦 Qty: {t['shares']}\n\n"
    msg += "_Risk sized at 1%. Trail stop loss upwards daily._"
else:
    msg = f"⬛️ *R-SQUARED SYSTEM* ({date_today})\n\n⚠️ No valid statistical breakouts today.\nConditions not met."

send_telegram(msg)
