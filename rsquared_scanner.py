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

# Quantitative Thresholds
MIN_R2 = 0.50
MIN_VELOCITY = 40.0
MIN_HURST = 0.45
MIN_SNR = 0.25
MIN_SHARPE = 1.0

# ==========================================
# 2. WATCHLIST
# ==========================================
MY_PORTFOLIO = ["ITC.NS", "HDFCBANK.NS"] 

nse_stocks = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "HINDUNILVR.NS", 
    "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS", "LT.NS", "BAJFINANCE.NS", 
    "HCLTECH.NS", "ASIANPAINT.NS", "AXISBANK.NS", "MARUTI.NS", "SUNPHARMA.NS", 
    "TITAN.NS", "DMART.NS", "ULTRACEMCO.NS", "BAJAJFINSV.NS", "WIPRO.NS", "M&M.NS", 
    "TATASTEEL.NS", "ADANIENT.NS", "POWERGRID.NS", "NTPC.NS", "TMCV.NS",
    "TECHM.NS", "NESTLEIND.NS", "ONGC.NS", "GRASIM.NS", "HINDALCO.NS", "JSWSTEEL.NS",
    "TRENT.NS", "BHEL.NS", "HAL.NS", "BEL.NS", "DLF.NS", "PFC.NS", "RECLTD.NS", "INDIGO.NS",
    "BSE.NS", "SUZLON.NS", "DIXON.NS", "CHOLAFIN.NS","AADHARHFC.NS", "AAVAS.NS", "ACE.NS", "AEGISLOG.NS", "AFFLE.NS", 
    "AFCONS.NS", "AMARAJABAT.NS", "AMBER.NS", "ANGELONE.NS", "ANANDRATHI.NS", 
    "APARIND.NS", "ARTIIND.NS", "ASTERDM.NS", "ATHER.NS", "ATUL.NS", 
    "BATAINDIA.NS", "BEML.NS", "BLS.NS", "BLUESTARCO.NS", "BRIGADE.NS", 
    "CASTROLIND.NS", "CENTRALBK.NS", "CESC.NS", "CHAMBLFERT.NS", "CHENNPETRO.NS", 
    "CAMS.NS", "CREDITACC.NS", "CROMPTON.NS", "DATAPATTERNS.NS", "DLINK.NS", 
    "DOMS.NS", "EASEMYTRIP.NS", "ENGINERSIN.NS", "FINCABLES.NS", "FINPIPE.NS", 
    "FIRSTSOURCE.NS", "FIVESTAR.NS", "FORCEMOT.NS", "GMDCLTD.NS", "GODIGIT.NS", 
    "GRSE.NS", "GESHIP.NS", "GLENMARK.NS", "HAPPIESTMND.NS", "HBLPOWER.NS", 
    "HFCL.NS", "HSCL.NS", "IFCI.NS", "IEX.NS", "IIFL.NS", 
    "INOXWIND.NS", "INTELLECT.NS", "IRCON.NS", "ITI.NS", "J&KBANK.NS", 
    "JBMAUTO.NS", "JSL.NS", "JUPITERWAG.NS", "JUSTDIAL.NS", "JYOTHYLAB.NS", 
    "KALPATARU.NS", "KARURVYSYA.NS", "KAYNES.NS", "KEC.NS", "LAURUSLABS.NS", 
    "LALPATHLAB.NS", "MGL.NS", "MAPMYINDIA.NS", "METROPOLIS.NS", "NATCOPHARM.NS", 
    "NAVINFLUOR.NS", "NBCC.NS", "NCC.NS", "NURTURE.NS", "NUVAMA.NS", 
    "OLECTRA.NS", "PNCINFRA.NS", "PNBHOUSING.NS", "PVRINOX.NS", "RAILTEL.NS", 
    "RAMCOIND.NS", "RAMKAFORG.NS", "RAYMOND.NS", "REDINGTON.NS", "RITES.NS", 
    "RBLBANK.NS", "SAGILITY.NS", "SANSERA.NS", "SARDAEN.NS", "SHYAMMETL.NS", 
    "SIGNATURE.NS", "SONATSOFTW.NS", "SWANENERGY.NS", "TANLA.NS", "TEJASNET.NS", 
    "TITAGARH.NS", "TRIDENT.NS", "TRIVENITURB.NS", "TTML.NS", "WELCORP.NS", 
    "WELSPUNLIV.NS", "ZEEL.NS", "ZENSARTECH.NS","ACC.NS", "ADANITOTAL.NS", "ABBOTINDIA", "ABCAPITAL.NS", "ABFRL.NS", 
    "ALKEM.NS", "APLLTD.NS", "APOLLOTYRE.NS", "ASHOKLEY.NS", "ASTRAL.NS", 
    "AUROPHARMA.NS", "AUBANK.NS", "BALKRISIND.NS", "BANDHANBNK.NS", "BANKBARODA.NS", 
    "BANKINDIA.NS", "BHEL.NS", "BIOCON.NS", "BSE.NS", "CANBK.NS", 
    "CGPOWER.NS", "COCHINSHIP.NS", "COFORGE.NS", "CONCOR.NS", "CUMMINSIND.NS", 
    "CYIENT.NS", "DABUR.NS", "DALBHARAT.NS", "DEEPAKNTR.NS", "DELHIVERY.NS", 
    "DIXON.NS", "ESCORTS.NS", "EXIDEIND.NS", "FEDERALBNK.NS", "FORTIS.NS", 
    "GMRINFRA.NS", "GODREJPROP.NS", "GUJGASLTD.NS", "HAVELLS.NS", "HDFCAMC.NS", 
    "HINDCOPPER.NS", "HINDPETRO.NS", "HINDZINC.NS", "HUDCO.NS", "IDBI.NS", 
    "IDFCFIRSTB.NS", "INDIAMART.NS", "INDIANB.NS", "INDHOTEL.NS", "IOB.NS", 
    "IRB.NS", "IREDA.NS", "IRFC.NS", "JAIBALAJI.NS", "JINDALSTEL.NS", 
    "JUBLFOOD.NS", "KALYANKJIL.NS", "KPITTECH.NS", "L&TFH.NS", "LICHSGFIN.NS", 
    "LUPIN.NS", "M&MFIN.NS", "MAHABANK.NS", "MANAPPURAM.NS", "MARICO.NS", 
    "MAXHEALTH.NS", "MAZDOCK.NS", "MCX.NS", "MOTILALOFS.NS", "MPHASIS.NS", 
    "MRF.NS", "MRPL.NS", "MUTHOOTFIN.NS", "NATIONALUM.NS", "NHPC.NS", 
    "NLCINDIA.NS", "NMDC.NS", "NYKAA.NS", "OBEROIRLTY.NS", "OIL.NS", 
    "OFSS.NS", "PAGEIND.NS", "PATANJALI.NS", "PAYTM.NS", "PBFINTECH.NS", 
    "PERSISTENT.NS", "PETRONET.NS", "PHOENIXLTD.NS", "PIIND.NS", "POLYCAB.NS", 
    "POONAWALLA.NS", "PRESTIGE.NS", "RADICO.NS", "RVNL.NS", "SAIL.NS", 
    "SJVN.NS", "SOLARINDS.NS", "SONACOMS.NS", "SRF.NS", "SUNDARMFIN.NS", 
    "SUPREMEIND.NS", "SUZLON.NS", "TATACHEM.NS", "TATACOMM.NS", "TATAELXSI.NS", 
    "TIINDIA.NS", "TORNTPOWER.NS", "UBL.NS", "UCOBANK.NS", "UNIONBANK.NS", 
    "UPL.NS", "VOLTAS.NS", "WAAREEENER.NS", "YESBANK.NS", "ZOMATO.NS"
]

nse_stocks = list(set(nse_stocks + MY_PORTFOLIO))

def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try: requests.post(url, json=payload, timeout=10)
    except: pass

def calc_atr(df, period=20):
    ranges = pd.concat([df['High'] - df['Low'], np.abs(df['High'] - df['Close'].shift()), np.abs(df['Low'] - df['Close'].shift())], axis=1)
    return ranges.max(axis=1).rolling(period).mean()

def get_hurst(ts):
    lags = range(2, 20)
    variances = [np.var(ts[lag:] - ts[:-lag]) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(variances), 1)
    return poly[0] / 2

all_green_stocks = []
triggered_stocks = []
portfolio_warnings = []

print("⚙️ Booting 5-Pillar Master Quant Engine...")

for stock in nse_stocks:
    try:
        df = yf.download(stock, period="1y", progress=False)
        if len(df) < 150: continue
        if isinstance(df.columns, pd.MultiIndex): df.columns = [c[0] for c in df.columns]
            
        c = df['Close'].iloc[-1]
        close_90 = df['Close'].iloc[-90:].values
        
        # 1. & 2. R-Squared and Velocity
        x_time = np.arange(90)
        slope, intercept, r_value, p_value, std_err = linregress(x_time, close_90)
        r_squared = r_value ** 2
        base_price = df['Close'].iloc[-90]
        annualized_velocity = (slope / base_price) * 252 * 100
        
        # 3. Hurst Exponent
        hurst = get_hurst(close_90)
        
        # 4. Signal-to-Noise Ratio (SNR)
        price_move = abs(c - base_price)
        path_length = np.sum(np.abs(np.diff(close_90)))
        snr = price_move / path_length if path_length != 0 else 0
        
        # 5. Annualized Sharpe Ratio
        daily_ret = df['Close'].pct_change().iloc[-90:]
        mean_ret = daily_ret.mean()
        std_ret = daily_ret.std()
        sharpe = (mean_ret / std_ret) * np.sqrt(252) if std_ret != 0 else 0
        
        sma_90 = df['Close'].rolling(90).mean().iloc[-1]
        
        # --- THE 5-PILLAR "ALL GREEN" CHECK ---
        is_all_green = (r_squared > MIN_R2) and (annualized_velocity > MIN_VELOCITY) and \
                       (hurst > MIN_HURST) and (snr > MIN_SNR) and \
                       (sharpe > MIN_SHARPE) and (c > sma_90)

        if is_all_green:
            all_green_stocks.append({
                "ticker": stock.replace('.NS', ''),
                "r2": r_squared,
                "vel": annualized_velocity,
                "hurst": hurst,
                "snr": snr,
                "sharpe": sharpe
            })

        # --- PORTFOLIO DIVERGENCE WARNING ---
        if stock in MY_PORTFOLIO and r_squared < 0.35:
            portfolio_warnings.append({"ticker": stock, "r2": r_squared})

        # --- THE BREAKOUT TRIGGER CHECK ---
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
        is_breakout = c > highest_50
        
        # If it is ALL GREEN *and* Breaking Out
        if is_all_green and is_accumulated and is_squeeze and is_breakout:
            atr = calc_atr(df, 20).iloc[-1]
            stop_loss = c - (atr * 2.5)
            risk_amt = ACCOUNT_SIZE * RISK_PER_TRADE
            risk_per_share = c - stop_loss
            shares = int(risk_amt / risk_per_share) if risk_per_share > 0 else 0
            
            triggered_stocks.append({
                "ticker": stock.replace('.NS', ''), "entry": c, "stop": stop_loss, "shares": shares
            })
            
    except Exception as e:
        continue

# ==========================================
# 3. FIRE TELEGRAM MESSAGE
# ==========================================
date_today = datetime.datetime.now().strftime("%Y-%m-%d")
msg = ""

if triggered_stocks:
    msg += f"🔥 *5-PILLAR BREAKOUTS* ({date_today})\n"
    msg += f"〰️〰️〰️〰️〰️〰️〰️〰️〰️〰️\n"
    for t in triggered_stocks:
        msg += f"🚀 *{t['ticker']}*\n"
        msg += f"🔹 Entry: ₹{t['entry']:.2f} | 🔴 Stop: ₹{t['stop']:.2f}\n"
        msg += f"📦 Qty: {t['shares']} shares\n\n"

if all_green_stocks:
    msg += f"🟢 *ALL-GREEN WATCHLIST (Super Trends)*\n"
    msg += f"〰️〰️〰️〰️〰️〰️〰️〰️〰️〰️\n"
    for g in all_green_stocks:
        msg += f"💎 *{g['ticker']}*\n"
        msg += f"└ R²: {g['r2']:.2f} | Vel: {g['vel']:.0f}% | Hurst: {g['hurst']:.2f}\n"
        msg += f"└ SNR: {g['snr']:.2f} | Sharpe: {g['sharpe']:.2f}\n\n"

if portfolio_warnings:
    msg += f"⚠️ *PORTFOLIO EXIT WARNINGS*\n"
    msg += f"〰️〰️〰️〰️〰️〰️〰️〰️〰️〰️\n"
    for w in portfolio_warnings:
        msg += f"📉 *{w['ticker'].replace('.NS','')}* R² dropped to {w['r2']:.2f}\n\n"

if not triggered_stocks and not all_green_stocks and not portfolio_warnings:
    msg = f"⬛️ *MASTER 5-PILLAR SYSTEM* ({date_today})\n\n⚠️ No stocks met the All-Green criteria today."

send_telegram(msg)
print("Finished. Message sent to Telegram.")
