import os
import re
import json
import datetime
import requests
import pandas as pd
import numpy as np
import yfinance as yf

# ==========================================
# 1. 22 NSE SECTORS + ETFs MAP
# ==========================================
SECTOR_MAP = {
    "Automobile and Auto Components": [
        ("MARUTI.NS", "Large"), ("M&M.NS", "Large"), ("BAJAJ-AUTO.NS", "Large"),
        ("TVSMOTOR.NS", "Mid"), ("ASHOKLEY.NS", "Mid"), ("BALKRISIND.NS", "Mid"),
        ("SANSERA.NS", "Small"), ("GABRIEL.NS", "Small"), ("SUBROS.NS", "Small"),
        ("JAMNAAUTO.NS", "Micro"), ("SUPRAJIT.NS", "Micro")
    ],
    "Capital Goods": [
        ("SIEMENS.NS", "Large"), ("ABB.NS", "Large"), ("HAL.NS", "Large"),
        ("CUMMINSIND.NS", "Mid"), ("BHEL.NS", "Mid"), ("POLYCAB.NS", "Mid"),
        ("APARINDS.NS", "Mid"), ("KIRLOSENG.NS", "Small"), ("GRSE.NS", "Small"),
        ("TEGA.NS", "Small"), ("WENDT.NS", "Micro"), ("ELECON.NS", "Micro")
    ],
    "Chemicals": [
        ("PIDILITIND.NS", "Large"), ("SRF.NS", "Large"), ("UPL.NS", "Large"),
        ("DEEPAKNTR.NS", "Mid"), ("TATACHEM.NS", "Mid"), ("AARTIIND.NS", "Mid"),
        ("NEOGEN.NS", "Small"), ("FINEORG.NS", "Small"), ("CLEAN.NS", "Small"),
        ("BODALCHEM.NS", "Micro"), ("ANDHRAPET.NS", "Micro"), ("GSFC.NS", "Small")
    ],
    "Construction": [
        ("LT.NS", "Large"), ("NBCC.NS", "Mid"), ("IRB.NS", "Mid"),
        ("KEC.NS", "Mid"), ("NCC.NS", "Small"), ("RVNL.NS", "Mid"),
        ("PNCINFRA.NS", "Small"), ("HGINFRA.NS", "Small"), ("KALPATARU.NS", "Small"),
        ("ASHOKA.NS", "Micro"), ("GPTINFRA.NS", "Micro")
    ],
    "Construction Materials": [
        ("ULTRACEMCO.NS", "Large"), ("AMBUJACEM.NS", "Large"), ("SHREECEM.NS", "Large"),
        ("DALBHARAT.NS", "Mid"), ("JKCEMENT.NS", "Mid"), ("RAMCOCEM.NS", "Mid"),
        ("HEIDELBERG.NS", "Small"), ("STARCEMENT.NS", "Small"), ("NUVOCO.NS", "Small"),
        ("SAURASHCEM.NS", "Micro"), ("DECCANCEM.NS", "Micro")
    ],
    "Consumer Durables": [
        ("TITAN.NS", "Large"), ("HAVELLS.NS", "Large"), ("DIXON.NS", "Large"),
        ("VOLTAS.NS", "Mid"), ("CROMPTON.NS", "Mid"), ("BLUESTARCO.NS", "Mid"),
        ("SYMPHONY.NS", "Small"), ("ORIENTELEC.NS", "Small"), ("AMBER.NS", "Small"),
        ("BUTTERFLY.NS", "Micro"), ("IFBIND.NS", "Micro")
    ],
    "Consumer Services": [
        ("TRENT.NS", "Large"), ("DMART.NS", "Large"), ("INDHOTEL.NS", "Large"),
        ("NYKAA.NS", "Mid"), ("JUBLFOOD.NS", "Mid"), ("LEMONTREE.NS", "Mid"),
        ("WESTLIFE.NS", "Small"), ("DEVYANI.NS", "Small"), ("SAPPHIRE.NS", "Small"),
        ("BARBEQUE.NS", "Micro"), ("KHADIM.NS", "Micro")
    ],
    "Diversified": [
        ("3MINDIA.NS", "Large"), ("GODREJIND.NS", "Mid"), ("DCMSHRIRAM.NS", "Mid"),
        ("BALMERLAWRI.NS", "Small"), ("GRAVITA.NS", "Small"),
        ("SHIVALIK.NS", "Micro"), ("HINDCOPPER.NS", "Mid"),
        ("ANDREWYULE.NS", "Micro"), ("MODIRUBBER.NS", "Micro")
    ],
    "Fast Moving Consumer Goods": [
        ("ITC.NS", "Large"), ("HINDUNILVR.NS", "Large"), ("NESTLEIND.NS", "Large"),
        ("BRITANNIA.NS", "Large"), ("DABUR.NS", "Large"), ("COLPAL.NS", "Mid"),
        ("EMAMILTD.NS", "Small"), ("BIKAJI.NS", "Small"), ("GILLETTE.NS", "Small"),
        ("HERITGFOOD.NS", "Micro"), ("TASTYBITE.NS", "Micro")
    ],
    "Financial Services": [
        ("HDFCBANK.NS", "Large"), ("ICICIBANK.NS", "Large"), ("SBIN.NS", "Large"),
        ("AUBANK.NS", "Mid"), ("FEDERALBNK.NS", "Mid"), ("MUTHOOTFIN.NS", "Mid"),
        ("CREDITACC.NS", "Small"), ("UJJIVANSFB.NS", "Small"), ("CDSL.NS", "Small"),
        ("MANAPPURAM.NS", "Micro"), ("GEOJITFSL.NS", "Micro")
    ],
    "Forest Materials": [
        ("CENTURYPLY.NS", "Mid"), ("GREENPANEL.NS", "Small"), ("GREENPLY.NS", "Small"),
        ("STYLAMIND.NS", "Small"), ("ARCHIDPLY.NS", "Micro")
    ],
    "Healthcare": [
        ("SUNPHARMA.NS", "Large"), ("CIPLA.NS", "Large"), ("DRREDDY.NS", "Large"),
        ("LUPIN.NS", "Mid"), ("ALKEM.NS", "Mid"), ("AUROPHARMA.NS", "Mid"),
        ("GRANULES.NS", "Small"), ("JBCHEPHARM.NS", "Small"), ("LAURUSLABS.NS", "Small"),
        ("NATCOPHARM.NS", "Micro"), ("ORCHPHARMA.NS", "Micro")
    ],
    "Information Technology": [
        ("TCS.NS", "Large"), ("INFY.NS", "Large"), ("HCLTECH.NS", "Large"),
        ("PERSISTENT.NS", "Mid"), ("COFORGE.NS", "Mid"), ("MPHASIS.NS", "Mid"),
        ("SONATSOFTW.NS", "Small"), ("ZENSARTECH.NS", "Small"), ("BSOFT.NS", "Small"),
        ("DATAMATICS.NS", "Micro"), ("CIGNITITEC.NS", "Micro")
    ],
    "Media, Entertainment & Publication": [
        ("ZEEL.NS", "Mid"), ("SUNTV.NS", "Mid"), ("PVRINOX.NS", "Mid"),
        ("NETWORK18.NS", "Small"), ("TV18BRDCST.NS", "Small"), ("HATHWAY.NS", "Small"),
        ("DBCORP.NS", "Small"), ("NAZARA.NS", "Small"), ("SAREGAMA.NS", "Small"),
        ("TVTODAY.NS", "Micro"), ("UFO.NS", "Micro")
    ],
    "Metals & Mining": [
        ("TATASTEEL.NS", "Large"), ("JSWSTEEL.NS", "Large"), ("HINDALCO.NS", "Large"),
        ("VEDL.NS", "Large"), ("NMDC.NS", "Mid"), ("JINDALSTEL.NS", "Mid"),
        ("SAIL.NS", "Mid"), ("NATIONALUM.NS", "Small"), ("APLAPOLLO.NS", "Mid"),
        ("GPIL.NS", "Small"), ("MAHSEAMLES.NS", "Micro")
    ],
    "Oil, Gas & Consumable Fuels": [
        ("RELIANCE.NS", "Large"), ("ONGC.NS", "Large"), ("BPCL.NS", "Large"),
        ("IOC.NS", "Large"), ("GAIL.NS", "Mid"), ("PETRONET.NS", "Mid"),
        ("IGL.NS", "Small"), ("MGL.NS", "Small"), ("GUJGASLTD.NS", "Mid"),
        ("GULFOILLUB.NS", "Micro"), ("CASTROLIND.NS", "Small")
    ],
    "Power": [
        ("NTPC.NS", "Large"), ("POWERGRID.NS", "Large"), ("TATAPOWER.NS", "Large"),
        ("ADANIPOWER.NS", "Large"), ("JSWENERGY.NS", "Mid"), ("TORNTPOWER.NS", "Mid"),
        ("CESC.NS", "Small"), ("NLCINDIA.NS", "Small"), ("JPPOWER.NS", "Micro"),
        ("RTNPOWER.NS", "Micro"), ("GIPCL.NS", "Micro")
    ],
    "Realty": [
        ("DLF.NS", "Large"), ("GODREJPROP.NS", "Large"), ("OBEROIRLTY.NS", "Mid"),
        ("PRESTIGE.NS", "Mid"), ("LODHA.NS", "Large"), ("PHOENIXLTD.NS", "Mid"),
        ("BRIGADE.NS", "Small"), ("SOBHA.NS", "Small"), ("MAHLIFE.NS", "Small"),
        ("ANANTRAJ.NS", "Small"), ("SUNTECK.NS", "Micro")
    ],
    "Services": [
        ("ADANIPORTS.NS", "Large"), ("INDIGO.NS", "Large"), ("IRCTC.NS", "Mid"),
        ("CONCOR.NS", "Mid"), ("DELHIVERY.NS", "Mid"), ("GMRAIRPORT.NS", "Mid"),
        ("REDINGTON.NS", "Small"), ("BLUEDART.NS", "Small"), ("TCIEXP.NS", "Small"),
        ("ALLCARGO.NS", "Micro"), ("GATEWAY.NS", "Micro")
    ],
    "Telecommunication": [
        ("BHARTIARTL.NS", "Large"), ("INDUSTOWER.NS", "Large"), ("IDEA.NS", "Mid"),
        ("TATACOMM.NS", "Mid"), ("HFCL.NS", "Small"), ("TEJASNET.NS", "Small"),
        ("ROUTE.NS", "Small"), ("GTLINFRA.NS", "Micro"), ("ITI.NS", "Small"),
        ("OPTIEMUS.NS", "Micro"), ("UNITECH.NS", "Micro")
    ],
    "Textiles": [
        ("PAGEIND.NS", "Large"), ("KPRMILL.NS", "Mid"), ("TRIDENT.NS", "Small"),
        ("WELSPUNLIV.NS", "Small"), ("VTL.NS", "Small"), ("ARVIND.NS", "Small"),
        ("SUTLEJTEX.NS", "Micro"), ("GOKEX.NS", "Small"), ("SHIVTEX.NS", "Micro"),
        ("FILATEX.NS", "Micro"), ("BANSWARA.NS", "Micro")
    ],
    "Utilities": [
        ("ADANIGREEN.NS", "Large"), ("NHPC.NS", "Mid"), ("SJVN.NS", "Small"),
        ("ADANIENSOL.NS", "Large"), ("SUZLON.NS", "Mid"), ("INOXWIND.NS", "Small"),
        ("WAAREEENER.NS", "Large"), ("ORIENTGEN.NS", "Micro"),
        ("RPOWER.NS", "Micro"), ("KPIGREEN.NS", "Small")
    ],
    "ETFs": [
        ("NIFTYBEES.NS", "Large"), ("JUNIORBEES.NS", "Large"), ("SETFNIF50.NS", "Large"),
        ("UTINIFTETF.NS", "Large"), ("NEXT50IETF.NS", "Large"), ("ICICIB22.NS", "Large"),
        ("MID150BEES.NS", "Mid"), ("HDFCMID150.NS", "Mid"), ("HDFCSML250.NS", "Small"),
        ("BANKBEES.NS", "Large"), ("PSUBNKBEES.NS", "Mid"), ("PVTBNKADD.NS", "Large"),
        ("FINIETF.NS", "Large"), ("ITBEES.NS", "Large"), ("PHARMABEES.NS", "Large"),
        ("HEALTHY.NS", "Mid"), ("AUTOBEES.NS", "Large"), ("FMCGIETF.NS", "Large"),
        ("CONSUMBEES.NS", "Large"), ("INFRABEES.NS", "Large"), ("OILIETF.NS", "Large"),
        ("METALIETF.NS", "Mid"), ("CPSEETF.NS", "Large"), ("MODEFENCE.NS", "Mid"),
        ("EVINDIA.NS", "Mid"), ("MAKEINDIA.NS", "Mid"), ("MOM100.NS", "Mid"),
        ("MOM30IETF.NS", "Large"), ("ALPHA.NS", "Mid"), ("ALPL30IETF.NS", "Large"),
        ("LOWVOLIETF.NS", "Large"), ("QUAL30IETF.NS", "Large"), ("NV20IETF.NS", "Large"),
        ("DIVOPPBEES.NS", "Large"), ("GOLDBEES.NS", "Large"), ("GOLDCASE.NS", "Large"),
        ("SETFGOLD.NS", "Large"), ("SILVERBEES.NS", "Large"), ("SILVERIETF.NS", "Large"),
        ("MON100.NS", "Large"), ("MAFANG.NS", "Large"), ("MASPTOP50.NS", "Large"),
        ("HNGSNGBEES.NS", "Large")
    ]
}

# ==========================================
# 2. TECHNICAL INDICATOR & MKR / SAHIL BREAKOUT HELPERS
# ==========================================
def calculate_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def analyze_rsi_cross(rsi_series, level):
    if len(rsi_series.dropna()) < 2:
        return "", ""
    valid = rsi_series.dropna()
    is_above = bool(valid.iloc[-1] >= level)
    cross = "UP" if is_above else "DN"
    ago = 0
    for i in range(len(valid) - 1, -1, -1):
        if (valid.iloc[i] >= level) == is_above:
            ago += 1
        else:
            break
    return cross, max(0, ago - 1)

def calculate_mkr(series, bandwidth=14, deviation=2.0, repaint=True):
    """
    TradingView 'Multi Kernel Regression [ChartPrime]':
    Laplace Kernel, Bandwidth=14, Source=Close, Deviation=2, Repaint=True
    """
    vals = series.values.astype(float)
    n = len(vals)
    out = np.copy(vals)

    if repaint and n > 1:
        W = min(n, 250)
        start = n - W
        sub = vals[start:]
        idx = np.arange(W)
        dist = np.abs(idx[:, None] - idx[None, :])
        weights = np.exp(-dist / float(bandwidth))
        out[start:] = (weights @ sub) / weights.sum(axis=1)

        for i in range(start):
            k = min(i + 1, 60)
            d = np.arange(k)
            w = np.exp(-d / float(bandwidth))
            w_vals = vals[i - k + 1 : i + 1][::-1]
            out[i] = np.dot(w_vals, w) / np.sum(w)
    else:
        W = min(max(int(bandwidth * 4), 30), 100)
        idx = np.arange(W)
        weights = np.exp(-idx / float(bandwidth))
        for i in range(n):
            k = min(i + 1, W)
            window_vals = vals[i - k + 1 : i + 1][::-1]
            w_slice = weights[:k]
            out[i] = np.dot(window_vals, w_slice) / np.sum(w_slice)

    line = pd.Series(out, index=series.index)
    resid_std = (series - line).rolling(window=bandwidth, min_periods=1).std().fillna(0)
    upper = line + (deviation * resid_std)
    lower = line - (deviation * resid_std)
    return line, upper, lower

def analyze_mkr_signal(close_series, line_series):
    if len(line_series) < 3:
        return "BUY", 0, "UP", 0

    diff = line_series.diff()
    is_rising = bool(diff.iloc[-1] > 0)
    sig = "BUY" if is_rising else "SELL"

    turn_ago = 0
    for i in range(len(diff) - 1, 0, -1):
        if (diff.iloc[i] > 0) == is_rising:
            turn_ago += 1
        else:
            break
    turn_ago = max(0, turn_ago - 1)

    is_above = bool(close_series.iloc[-1] >= line_series.iloc[-1])
    cross = "UP" if is_above else "DN"
    cross_ago = 0
    for i in range(len(close_series) - 1, -1, -1):
        if (close_series.iloc[i] >= line_series.iloc[i]) == is_above:
            cross_ago += 1
        else:
            break
    cross_ago = max(0, cross_ago - 1)

    return sig, turn_ago, cross, cross_ago

def calculate_sahil_breakout(df, entry_len=50, exit_len=20):
    """
    Sahil's Video Strategy (50-bar Highest High BUY / 20-bar Lowest Low EXIT):
    - Upper Green Line: Highest High of previous 50 bars
    - Lower Red Line: Lowest Low of previous 20 bars (Trailing Stop-Loss)
    - Returns: (state, bars_ago, breakout_level, trailing_sl)
    """
    if len(df) < entry_len + 1:
        return "", "", "", ""

    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    upper = high.rolling(entry_len).max().shift(1)
    lower_exit = low.rolling(exit_len).min().shift(1)

    state = ""
    bars_ago = ""
    last_sig_idx = None

    for i in range(entry_len, len(df)):
        c_val = close.iloc[i]
        u_val = upper.iloc[i]
        l_exit_val = lower_exit.iloc[i]

        if pd.isna(u_val) or pd.isna(l_exit_val):
            continue

        if c_val > u_val:
            if state != "BUY":
                state = "BUY"
                last_sig_idx = i
        elif state == "BUY" and c_val < l_exit_val:
            state = "SELL"
            last_sig_idx = i

    if last_sig_idx is not None:
        bars_ago = (len(df) - 1) - last_sig_idx

    breakout_level = round(float(high.rolling(entry_len).max().iloc[-1]), 2)
    trailing_sl = round(float(low.rolling(exit_len).min().iloc[-1]), 2)

    return state, bars_ago, breakout_level, trailing_sl

# ==========================================
# 3. FPIDATA.IN FPI AUTO-UPDATE (NO EXTERNAL DEPENDENCY)
# ==========================================
def update_nsdl_json():
    print("fpidata.in પરથી લેટેસ્ટ FPI ફ્લો ડેટા તપાસી રહ્યા છીએ...")
    nsdl_file = "nsdl.json"
    curr_data = {}
    if os.path.exists(nsdl_file):
        try:
            with open(nsdl_file, "r", encoding="utf-8") as f:
                curr_data = json.load(f)
        except Exception:
            curr_data = {}

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
    }

    try:
        url_fpi = "https://fpidata.in/"
        r = requests.get(url_fpi, headers=headers, timeout=15)
        if r.status_code == 200:
            html_text = r.text
            plain_text = re.sub(r'<[^>]+>', ' ', html_text)
            date_match = re.search(r'(\d{1,2}\s+[A-Za-z]{3}\s+20\d\d)', plain_text)
            fpi_date = date_match.group(1) if date_match else datetime.datetime.now().strftime("%d %b %Y")

            parsed_sectors = {}
            rows_html = re.findall(r'<tr[^>]*>(.*?)</tr>', html_text, flags=re.S | re.I)
            for tr in rows_html:
                cells = re.findall(r'<t[dh][^>]*>(.*?)</t[dh]>', tr, flags=re.S | re.I)
                cols = [re.sub(r'<[^>]+>', '', c).replace('&amp;', '&').replace(',', '').strip() for c in cells]
                if len(cols) >= 2:
                    name = cols[0].strip()
                    matched_sec = None
                    for s_key in SECTOR_MAP.keys():
                        if s_key == "ETFs":
                            continue
                        if s_key.lower() in name.lower() or name.lower() in s_key.lower():
                            matched_sec = s_key
                            break
                    if matched_sec:
                        nums = []
                        for c_val in cols[1:]:
                            c_clean = c_val.replace("₹", "").replace("Cr", "").replace("+", "").replace("−", "-").strip()
                            try:
                                nums.append(float(c_clean))
                            except ValueError:
                                continue
                        if nums:
                            net_flow = nums[0]
                            auc = nums[1] if len(nums) > 1 else (curr_data.get("data", {}).get(matched_sec, [0, 50000])[1])
                            parsed_sectors[matched_sec] = [net_flow, auc]

            if len(parsed_sectors) >= 10:
                curr_data["date"] = fpi_date
                curr_data["data"] = parsed_sectors
                with open(nsdl_file, "w", encoding="utf-8") as f:
                    json.dump(curr_data, f, indent=2)
                print(f"fpidata.in પરથી {len(parsed_sectors)} સેક્ટર્સનો FPI ડેટા અપડેટ થયો ({fpi_date}).")
                return
    except Exception as e:
        print(f"fpidata.in સ્ક્રેપિંગમાં ચેતવણી: {e}. વર્તમાન ડેટા સુરક્ષિત રાખવામાં આવ્યો છે.")

# ==========================================
# 4. MAIN SCREENER & DATA GENERATOR
# ==========================================
def main():
    update_nsdl_json()
    print("સ્ટોક્સ અને ETFs નું માર્કેટ સ્કેનિંગ શરૂ થાય છે...")
    all_tickers = []
    meta = {}

    for sec, stocks in SECTOR_MAP.items():
        for sym, cap in stocks:
            all_tickers.append(sym)
            meta[sym.replace(".NS", "")] = (sec, cap)

    if os.path.exists("extra_symbols.txt"):
        with open("extra_symbols.txt", "r", encoding="utf-8") as ef:
            for line in ef:
                sym = line.strip().upper()
                if sym and not sym.startswith("#"):
                    t_ns = f"{sym}.NS"
                    if t_ns not in all_tickers:
                        all_tickers.append(t_ns)
                    if sym not in meta:
                        meta[sym] = ("Others", "Small")

    bench_sym = "^CRSLDX"
    tickers_to_download = list(set(all_tickers + [bench_sym]))

    print(f"કુલ {len(tickers_to_download)} સિમ્બોલ્સનો ૨ વર્ષનો ડેટા ડાઉનલોડ થઈ રહ્યો છે...")
    data = yf.download(tickers_to_download, period="2y", interval="1d", group_by="ticker", threads=True)

    out_rows = []
    bench_rets = {"1W": 0.0, "1M": 0.0, "3M": 0.0, "6M": 0.0}
    b_c = None
    try:
        if bench_sym in data and len(data[bench_sym]["Close"].dropna()) > 30:
            b_c = data[bench_sym]["Close"].dropna()
            b_cur = b_c.iloc[-1]
            if len(b_c) >= 6: bench_rets["1W"] = round(float((b_cur - b_c.iloc[-6]) / b_c.iloc[-6] * 100), 2)
            if len(b_c) >= 22: bench_rets["1M"] = round(float((b_cur - b_c.iloc[-22]) / b_c.iloc[-22] * 100), 2)
            if len(b_c) >= 66: bench_rets["3M"] = round(float((b_cur - b_c.iloc[-66]) / b_c.iloc[-66] * 100), 2)
            if len(b_c) >= 130: bench_rets["6M"] = round(float((b_cur - b_c.iloc[-130]) / b_c.iloc[-130] * 100), 2)
    except Exception as e:
        print(f"Benchmark error: {e}")

    out_rows.append({
        "Symbol": "NIFTY 500", "Industry": "", "Cap": "", "Price": "",
        "1W": bench_rets["1W"], "1M": bench_rets["1M"], "3M": bench_rets["3M"], "6M": bench_rets["6M"],
        "Daily": "", "Weekly": "", "DFresh": "", "WFresh": "",
        "DCross": "", "DCrossAgo": "", "WCross": "", "WCrossAgo": "",
        "DTurnAgo": "", "WTurnAgo": "", "EMA20": "", "EMA50": "", "EMA100": "", "EMA200": "",
        "RSState": "", "RSPos": "", "RSCross": "", "RSCrossAgo": "",
        "RS_Score": 0.0, "RS_Rank": 50,
        "DelivPct": "", "DelivHigh": "", "RSI_D": "", "RSI_W": "", "RSI_M": "",
        "RSI40Cross": "", "RSI40CrossAgo": "", "RSI50Cross": "", "RSI50CrossAgo": "",
        "RSI60Cross": "", "RSI60CrossAgo": "", "DConfirmed": "", "WConfirmed": "",
        "DLine": "", "DUpper": "", "DLower": "", "WLine": "", "WUpper": "", "WLower": "",
        "SahilD": "", "SahilDAgo": "", "SahilD_BO": "", "SahilD_TSL": "",
        "SahilW": "", "SahilWAgo": "", "SahilW_BO": "", "SahilW_TSL": ""
    })

    for t in all_tickers:
        raw_sym = t.replace(".NS", "")
        sec, cap = meta.get(raw_sym, ("Others", "Small"))

        try:
            if t not in data:
                continue
            df = data[t].dropna(subset=["Close", "High", "Low"])
            if len(df) < 50:
                continue

            c = df["Close"]
            cur_price = round(float(c.iloc[-1]), 2)

            r1w = round(float((cur_price - c.iloc[-6]) / c.iloc[-6] * 100), 2) if len(c) >= 6 else 0.0
            r1m = round(float((cur_price - c.iloc[-22]) / c.iloc[-22] * 100), 2) if len(c) >= 22 else 0.0
            r3m = round(float((cur_price - c.iloc[-66]) / c.iloc[-66] * 100), 2) if len(c) >= 66 else 0.0
            r6m = round(float((cur_price - c.iloc[-130]) / c.iloc[-130] * 100), 2) if len(c) >= 130 else 0.0
            r1y = round(float((cur_price - c.iloc[-250]) / c.iloc[-250] * 100), 2) if len(c) >= 250 else r6m

            rs_score = round((0.4 * r3m) + (0.3 * r6m) + (0.3 * r1y), 2)

            ema20 = round(float(calculate_ema(c, 20).iloc[-1]), 2)
            ema50 = round(float(calculate_ema(c, 50).iloc[-1]), 2)
            ema100 = round(float(calculate_ema(c, 100).iloc[-1]), 2)
            ema200 = round(float(calculate_ema(c, 200).iloc[-1]), 2)

            # Daily MKR
            d_line_s, d_upper_s, d_lower_s = calculate_mkr(c, bandwidth=14, deviation=2.0, repaint=True)
            d_sig, d_turn_ago, d_cross, d_cross_ago = analyze_mkr_signal(c, d_line_s)
            d_fresh = "YES" if d_turn_ago <= 3 else ""
            d_confirmed = "YES" if (d_sig == "BUY" and d_cross == "UP") else ""

            # Weekly Resample & Weekly MKR
            agg_dict = {"High": "max", "Low": "min", "Close": "last"}
            if "Open" in df.columns:
                agg_dict["Open"] = "first"
            if "Volume" in df.columns:
                agg_dict["Volume"] = "sum"
            df_w = df.resample("W-FRI").agg(agg_dict).dropna(subset=["Close", "High", "Low"])
            w_c = df_w["Close"]

            w_line_s, w_upper_s, w_lower_s = calculate_mkr(w_c, bandwidth=14, deviation=2.0, repaint=True)
            w_sig, w_turn_ago, w_cross, w_cross_ago = analyze_mkr_signal(w_c, w_line_s)
            w_fresh = "YES" if w_turn_ago <= 2 else ""
            w_confirmed = "YES" if (w_sig == "BUY" and w_cross == "UP") else ""

            # Mansfield Relative Strength (vs Nifty 500)
            rs_state, rs_pos, rs_cross, rs_cross_ago = "", "", "", ""
            if b_c is not None and len(b_c) > 50:
                aligned_b = b_c.reindex(df.index).ffill()
                rs_ratio = (c / aligned_b) * 100
                rs_ma = rs_ratio.rolling(50).mean()
                mrs = ((rs_ratio / rs_ma) - 1) * 100
                mrs_valid = mrs.dropna()
                if len(mrs_valid) > 1:
                    rs_pos = round(float(mrs_valid.iloc[-1]), 2)
                    is_rs_pos = rs_pos >= 0
                    rs_state = "BUY" if is_rs_pos else "SELL"
                    rs_cross = "UP" if is_rs_pos else "DN"
                    c_ago = 0
                    for idx_i in range(len(mrs_valid) - 1, -1, -1):
                        if (mrs_valid.iloc[idx_i] >= 0) == is_rs_pos:
                            c_ago += 1
                        else:
                            break
                    rs_cross_ago = max(0, c_ago - 1)

            # RSI (Daily, Weekly, Monthly + 40/50/60 Crosses)
            rsi_d_s = calculate_rsi(c, 14)
            rsi_w_s = calculate_rsi(w_c, 14)
            try:
                df_m = df.resample("ME").agg({"Close": "last"}).dropna()
            except Exception:
                df_m = df.resample("M").agg({"Close": "last"}).dropna()
            rsi_m_s = calculate_rsi(df_m["Close"], 14)

            rsi_d_val = round(float(rsi_d_s.iloc[-1]), 1) if not pd.isna(rsi_d_s.iloc[-1]) else ""
            rsi_w_val = round(float(rsi_w_s.iloc[-1]), 1) if not pd.isna(rsi_w_s.iloc[-1]) else ""
            rsi_m_val = round(float(rsi_m_s.iloc[-1]), 1) if len(rsi_m_s) > 0 and not pd.isna(rsi_m_s.iloc[-1]) else ""

            r40_c, r40_ago = analyze_rsi_cross(rsi_d_s, 40)
            r50_c, r50_ago = analyze_rsi_cross(rsi_d_s, 50)
            r60_c, r60_ago = analyze_rsi_cross(rsi_d_s, 60)

            # Sahil 50-20 Breakout & Trailing Stop-Loss (Daily & Weekly)
            sahil_d, sahil_d_ago, sahil_d_bo, sahil_d_tsl = calculate_sahil_breakout(df, 50, 20)
            sahil_w, sahil_w_ago, sahil_w_bo, sahil_w_tsl = calculate_sahil_breakout(df_w, 50, 20)

            out_rows.append({
                "Symbol": raw_sym,
                "Industry": sec,
                "Cap": cap,
                "Price": cur_price,
                "1W": r1w,
                "1M": r1m,
                "3M": r3m,
                "6M": r6m,
                "Daily": d_sig,
                "Weekly": w_sig,
                "DFresh": d_fresh,
                "WFresh": w_fresh,
                "DCross": d_cross,
                "DCrossAgo": d_cross_ago,
                "WCross": w_cross,
                "WCrossAgo": w_cross_ago,
                "DTurnAgo": d_turn_ago,
                "WTurnAgo": w_turn_ago,
                "EMA20": ema20,
                "EMA50": ema50,
                "EMA100": ema100,
                "EMA200": ema200,
                "RSState": rs_state,
                "RSPos": rs_pos,
                "RSCross": rs_cross,
                "RSCrossAgo": rs_cross_ago,
                "RS_Score": rs_score,
                "RS_Rank": 50,
                "DelivPct": "",
                "DelivHigh": "",
                "RSI_D": rsi_d_val,
                "RSI_W": rsi_w_val,
                "RSI_M": rsi_m_val,
                "RSI40Cross": r40_c,
                "RSI40CrossAgo": r40_ago,
                "RSI50Cross": r50_c,
                "RSI50CrossAgo": r50_ago,
                "RSI60Cross": r60_c,
                "RSI60CrossAgo": r60_ago,
                "DConfirmed": d_confirmed,
                "WConfirmed": w_confirmed,
                "DLine": round(float(d_line_s.iloc[-1]), 2),
                "DUpper": round(float(d_upper_s.iloc[-1]), 2),
                "DLower": round(float(d_lower_s.iloc[-1]), 2),
                "WLine": round(float(w_line_s.iloc[-1]), 2),
                "WUpper": round(float(w_upper_s.iloc[-1]), 2),
                "WLower": round(float(w_lower_s.iloc[-1]), 2),
                "SahilD": sahil_d,
                "SahilDAgo": sahil_d_ago,
                "SahilD_BO": sahil_d_bo,
                "SahilD_TSL": sahil_d_tsl,
                "SahilW": sahil_w,
                "SahilWAgo": sahil_w_ago,
                "SahilW_BO": sahil_w_bo,
                "SahilW_TSL": sahil_w_tsl
            })
        except Exception as e:
            print(f"Error processing {raw_sym}: {e}")
            continue

    # Calculate RS_Rank (0-100 percentile) for all stocks (excluding NIFTY 500 row at index 0)
    if len(out_rows) > 1:
        scores = pd.Series([row["RS_Score"] for row in out_rows[1:]])
        ranks = (scores.rank(pct=True) * 100).round().astype(int)
        for idx, r_val in enumerate(ranks):
            out_rows[idx + 1]["RS_Rank"] = int(r_val)

    # Save Output Files (JSON & CSV)
    with open("data.json", "w", encoding="utf-8") as jf:
        json.dump(out_rows, jf, indent=2)

    pd.DataFrame(out_rows).to_csv("data.csv", index=False)
    print(f"સ્કેનિંગ પૂર્ણ! કુલ {len(out_rows) - 1} સ્ટોક્સનો ડેટા data.json અને data.csv માં સેવ થયો.")

if __name__ == "__main__":
    main()
