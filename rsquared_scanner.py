import os
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import datetime
import warnings
from scipy.stats import linregress

warnings.filterwarnings('ignore')

# ==========================================
# 1. CREDENTIALS & SETTINGS
# ==========================================
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
ACCOUNT_SIZE = 100000  
RISK_PER_TRADE = 0.01  

# Zero-Lag Kinetic Settings
MIN_MICRO_R2 = 0.65
MIN_VELOCITY = 60.0
MIN_HURST = 0.50
ATR_MULTIPLIER = 1.5 # Tighter stop for early entry

# ==========================================
# 2. WATCHLIST
# ==========================================
MY_PORTFOLIO = ["ITC.NS"] # Add stocks you own here

nse_stocks = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "HINDUNILVR.NS", 
    "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS", "LT.NS", "BAJFINANCE.NS", 
    "HCLTECH.NS", "ASIANPAINT.NS", "AXISBANK.NS", "MARUTI.NS", "SUNPHARMA.NS", 
    "TITAN.NS", "DMART.NS", "ULTRACEMCO.NS", "BAJAJFINSV.NS", "WIPRO.NS", "M&M.NS", 
    "TATASTEEL.NS", "ADANIENT.NS", "POWERGRID.NS", "NTPC.NS", "TMCV.NS",
    "TECHM.NS", "NESTLEIND.NS", "ONGC.NS", "GRASIM.NS", "HINDALCO.NS", "JSWSTEEL.NS",
    "TRENT.NS", "BHEL.NS", "HAL.NS", "BEL.NS", "DLF.NS", "PFC.NS", "RECLTD.NS", "INDIGO.NS",
    "BSE.NS", "SUZLON.NS", "DIXON.NS", "CHOLAFIN.NS", "BANKBARODA.NS"
]

nse_stocks = list(set(nse_stocks + MY_PORTFOLIO))

def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try: requests.post(url, json=payload, timeout=10)
    except: pass

def calc_atr(df, period=14):
    ranges = pd.concat([df['High'] - df['Low'], np.abs(df['High'] - df['Close'].shift()), np.abs(df['Low'] - df['Close'].shift())], axis=1)
    return ranges.max(axis=1).rolling(period).mean()

def get_hurst(ts):
    lags = range(2, min(10, len(ts)//2))
    if len(lags) == 0: return 0.5
    variances = [np.var(ts[lag:] - ts[:-lag]) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(variances), 1)
    return poly[0] / 2

triggered_stocks = []
portfolio_warnings = []

print("⚙️ Booting Zero-Lag Kinetic Engine...")

for stock in nse_stocks:
    try:
        # Fetch 4 months to comfortably cover 45-day window + 30-day AVWAP history
        df = yf.download(stock, period="4mo", progress=False)
        if len(df) < 60: continue
        if isinstance(df.columns, pd.MultiIndex): df.columns = [c[0] for c in df.columns]
            
        c = df['Close'].iloc[-1]
        
        # 1. LIQUIDITY FILTER
        df['Turnover'] = (df['Close'] * df['Volume']) / 10000000
        avg_turnover = df['Turnover'].rolling(20).mean().iloc[-1]
        if avg_turnover < 10.0: continue

        # 2. GAPLESS PRICE (Removes False Gap-Ups)
        df['Intraday'] = df['Close'] - df['Open']
        df['Gapless'] = df['Intraday'].cumsum()
        
        c14_gapless = df['Gapless'].iloc[-14:].values
        c45_gapless = df['Gapless'].iloc[-45:].values
        
        x14 = np.arange(14)
        _, _, r_micro, _, _ = linregress(x14, c14_gapless)
        r2_micro = r_micro**2
        
        x45 = np.arange(45)
        _, _, r_med, _, _ = linregress(x45, c45_gapless)
        r2_med = r_med**2
        
        # 3. KINETIC VELOCITY (Real Closing Prices)
        c14_real = df['Close'].iloc[-14:].values
        slope_real, _, _, _, _ = linregress(x14, c14_real)
        base_price = c14_real[0]
        vel = (slope_real / base_price) * 252 * 100 if base_price != 0 else 0
        
        # 4. MICRO HURST EXPONENT
        hurst = get_hurst(c14_real)
        
        # 5. ANCHORED VWAP
        last_30_lows = df['Low'].iloc[-30:]
        lowest_idx = last_30_lows.idxmin()
        vwap_df = df.loc[lowest_idx:]
        typical_price = (vwap_df['High'] + vwap_df['Low'] + vwap_df['Close'] * 2) / 4
        vwap = (typical_price * vwap_df['Volume']).sum() / vwap_df['Volume'].sum()
        
        vwap_surge = c > vwap

        # --- THE MASTER KINETIC IGNITION LOGIC ---
        is_ignition = (r2_micro > MIN_MICRO_R2) and (r2_med > 0.20) and (vel > MIN_VELOCITY) and (hurst > MIN_HURST) and vwap_surge
        
        if is_ignition:
            atr = calc_atr(df, 14).iloc[-1]
            stop_loss = c - (atr * ATR_MULTIPLIER)
            risk_amt = ACCOUNT_SIZE * RISK_PER_TRADE
            risk_per_share = c - stop_loss
            shares = int(risk_amt / risk_per_share) if risk_per_share > 0 else 0
            
            triggered_stocks.append({
                "ticker": stock.replace('.NS', ''),
                "entry": c, "stop": stop_loss, "shares": shares,
                "r2": r2_micro, "vel": vel, "vwap": vwap
            })

        # --- PORTFOLIO DANGER WARNING ---
        if stock in MY_PORTFOLIO:
            # Sell warning if the micro momentum violently dies, or drops below the Anchored VWAP
            if r2_micro < 0.20 or c < vwap:
                portfolio_warnings.append({"ticker": stock.replace('.NS', ''), "r2": r2_micro, "vwap": vwap})
            
    except Exception as e:
        continue

# ==========================================
# 3. FIRE TELEGRAM MESSAGE
# ==========================================
date_today = datetime.datetime.now().strftime("%Y-%m-%d")
msg = ""

if triggered_stocks:
    msg += f"🔥 *ZERO-LAG KINETIC IGNITION* ({date_today})\n"
    msg += f"〰️〰️〰️〰️〰️〰️〰️〰️〰️〰️〰️〰️\n"
    for t in triggered_stocks:
        msg += f"🚀 *{t['ticker']}*\n"
        msg += f"└ R²: {t['r2']:.2f} | Vel: {t['vel']:.0f}% \n"
        msg += f"└ > VWAP Support: ₹{t['vwap']:.1f}\n"
        msg += f"🔹 Entry: ₹{t['entry']:.2f} | 🔴 Stop: ₹{t['stop']:.2f}\n"
        msg += f"📦 Qty: {t['shares']} shares\n\n"

if portfolio_warnings:
    msg += f"⚠️ *PORTFOLIO EXIT WARNINGS*\n"
    msg += f"〰️〰️〰️〰️〰️〰️〰️〰️〰️〰️〰️〰️\n"
    for w in portfolio_warnings:
        msg += f"📉 *{w['ticker']}* has lost Ignition Momentum!\n"
        msg += f"└ R² collapsed or Price fell below VWAP.\n\n"

if not triggered_stocks and not portfolio_warnings:
    msg = f"⬛️ *KINETIC IGNITION SYSTEM* ({date_today})\n\n⚠️ No new 14-day ignitions found today."

send_telegram(msg)
print("Finished. Message sent to Telegram.")
