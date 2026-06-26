import os
import pandas as pd
import numpy as np
import requests
import datetime
import time
import warnings
from scipy.stats import linregress
from tvDatafeed import TvDatafeed, Interval

warnings.filterwarnings('ignore')

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

# Kinetic Settings for Indices (Slightly lower velocity because indices move slower than stocks)
MIN_MICRO_R2 = 0.65
MIN_VELOCITY = 40.0 
MIN_HURST = 0.50

def load_indices_from_csv(filename):
    if not os.path.exists(filename): return []
    try:
        df = pd.read_csv(filename, header=None)
        return df[0].dropna().astype(str).str.strip().tolist()
    except Exception as e:
        return []

indices = load_indices_from_csv("index_watchlist.csv")
if not indices:
    print("Error: index_watchlist.csv is empty or missing!")
    exit()

def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try: requests.post(url, json=payload, timeout=10)
    except: pass

def get_hurst(ts):
    lags = range(2, min(10, len(ts)//2))
    if len(lags) == 0: return 0.5
    variances = [np.var(ts[lag:] - ts[:-lag]) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(variances), 1)
    return poly[0] / 2

triggered_indices = []

print(f"⚙️ Booting Index Kinetic Engine... (Scanning {len(indices)} Indices)")
tv = TvDatafeed()

for idx in indices:
    try:
        df = tv.get_hist(symbol=idx, exchange='NSE', interval=Interval.in_daily, n_bars=100)
        if df is None or len(df) < 50: continue
            
        c = df['close'].iloc[-1]
        
        # 1. GAPLESS PRICE 
        df['intraday'] = df['close'] - df['open']
        df['gapless'] = df['intraday'].cumsum()
        
        c14_gapless = df['gapless'].iloc[-14:].values
        c45_gapless = df['gapless'].iloc[-45:].values
        
        x14 = np.arange(14)
        _, _, r_micro, _, _ = linregress(x14, c14_gapless)
        r2_micro = r_micro**2
        
        x45 = np.arange(45)
        _, _, r_med, _, _ = linregress(x45, c45_gapless)
        r2_med = r_med**2
        
        # 2. KINETIC VELOCITY 
        c14_real = df['close'].iloc[-14:].values
        slope_real, _, _, _, _ = linregress(x14, c14_real)
        base_price = c14_real[0]
        vel = (slope_real / base_price) * 252 * 100 if base_price != 0 else 0
        
        # 3. MICRO HURST EXPONENT
        hurst = get_hurst(c14_real)
        
        # 4. ANCHORED VWAP (Or TWAP if Volume is missing)
        last_30_lows = df['low'].iloc[-30:]
        lowest_idx = last_30_lows.idxmin()
        vwap_df = df.loc[lowest_idx:]
        
        typical_price = (vwap_df['high'] + vwap_df['low'] + vwap_df['close'] * 2) / 4
        
        # Safe VWAP calculation for indices
        if vwap_df['volume'].sum() == 0 or np.isnan(vwap_df['volume'].sum()):
            vwap = typical_price.mean() # Fallback to Time-Weighted
        else:
            vwap = (typical_price * vwap_df['volume']).sum() / vwap_df['volume'].sum()
        
        vwap_surge = c > vwap

        # --- THE MASTER KINETIC IGNITION LOGIC ---
        is_ignition = (r2_micro > MIN_MICRO_R2) and (r2_med > 0.20) and (vel > MIN_VELOCITY) and (hurst > MIN_HURST) and vwap_surge
        
        if is_ignition:
            triggered_indices.append({
                "ticker": idx.replace('CNX', ''),
                "price": c, "r2": r2_micro, "vel": vel, "vwap": vwap
            })
            
        time.sleep(1) # Prevent TradingView rate limits
        
    except Exception as e:
        continue

# ==========================================
# FIRE TELEGRAM MESSAGE
# ==========================================
date_today = datetime.datetime.now().strftime("%Y-%m-%d")
msg = ""

if triggered_indices:
    msg += f"📈 *INDEX KINETIC IGNITION* ({date_today})\n"
    msg += f"〰️〰️〰️〰️〰️〰️〰️〰️〰️〰️〰️〰️\n"
    for t in triggered_indices:
        msg += f"👑 *{t['ticker']}*\n"
        msg += f"└ R²: {t['r2']:.2f} | Vel: {t['vel']:.0f}% \n"
        msg += f"└ > VWAP Support: ₹{t['vwap']:.1f}\n"
        msg += f"🔹 Current Level: ₹{t['price']:.1f}\n\n"
else:
    msg = f"⬛️ *INDEX IGNITION SYSTEM* ({date_today})\n\n⚠️ No Sector Indices are experiencing a 14-day Ignition today."

send_telegram(msg)
print("Finished. Message sent to Telegram.")
