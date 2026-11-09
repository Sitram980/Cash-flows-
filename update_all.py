import os
import json
import datetime
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from bs4 import BeautifulSoup

# ==========================================
# 1. 22 NSE SECTORS & DEFAULT STOCKS MAP
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
        ("KIRLOSENG.NS", "Small"), ("GRSE.NS", "Small"), ("TEGA.NS", "Small"),
        ("WENDT.NS", "Micro"), ("ELECON.NS", "Micro")
    ],
    "Chemicals": [
        ("PIDILITIND.NS", "Large"), ("SRF.NS", "Large"), ("UPL.NS", "Large"),
        ("DEEPAKNTR.NS", "Mid"), ("TATACHEM.NS", "Mid"), ("AARTIIND.NS", "Mid"),
        ("NEOGEN.NS", "Small"), ("FINEORG.NS", "Small"), ("CLEAN.NS", "Small"),
        ("BODALCHEM.NS", "Micro"), ("ANDHRAPET.NS", "Micro")
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
    ]
}

# ==========================================
# 2. TECHNICAL INDICATOR HELPERS
# ==========================================
def calculate_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def calculate_mkr(series, period=14):
    line = series.ewm(span=period, adjust=False).mean()
    std = series.rolling(window=period).std().fillna(0)
    upper = line + (1.5 * std)
    lower = line - (1.5 * std)
    return line, upper, lower

# ==========================================
# 3. NSDL FPI AUTO-UPDATE ENGINE
# ==========================================
def update_nsdl_json():
    print("NSDL FPI ડેટા ચેક કરી રહ્યા છીએ...")
    nsdl_file = "nsdl.json"
    curr_data = {}
    if os.path.exists(nsdl_file):
        try:
            with open(nsdl_file, "r", encoding="utf-8") as f:
                curr_data = json.load(f)
        except Exception:
            curr_data = {}

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    url = "https://www.fpi.nsdl.co.in/web/Reports/FortnightSectorwise.aspx"

    try:
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, "html.parser")
            table = soup.find("table", {"id": "gvSectorwise"}) or soup.find("table")
            if table:
                rows = table.find_all("tr")
                parsed_sectors = {}
                for tr in rows[1:]:
                    cols = [td.get_text(strip=True).replace(",", "") for td in tr.find_all("td")]
                    if len(cols) >= 3:
                        sec_name = cols[0]
                        try:
                            net_flow = float(cols[1])
                            auc = float(cols[2])
                            parsed_sectors[sec_name] = [net_flow, auc]
                        except ValueError:
                            continue

                if parsed_sectors:
                    curr_data["data"] = parsed_sectors
                    curr_data["date"] = datetime.datetime.now().strftime("%d %b %Y")
                    with open(nsdl_file, "w", encoding="utf-8") as f:
                        json.dump(curr_data, f, indent=2)
                    print("NSDL ડેટા સફળતાપૂર્વક અપડેટ થયો.")
                    return
        print("NSDL ડેટા પહેલેથી જ અપ-ટુ-ડેટ છે અથવા નવો રિપોર્ટ હજુ આવ્યો નથી.")
    except Exception as e:
        print(f"NSDL અપડેટ સ્કીપ થયું: {e}")

# ==========================================
# 4. MAIN SCREENER & DATA GENERATOR
# ==========================================
def main():
    print("સ્ટોક્સનું માર્કેટ સ્કેનિંગ શરૂ થાય છે...")
    
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
                    meta[sym] = ("Others", "Small")

    bench_sym = "^CRSLDX"
    tickers_to_download = list(set(all_tickers + [bench_sym]))

    print(f"કુલ {len(tickers_to_download)} સ્ટોક્સનો ડેટા ડાઉનલોડ થઈ રહ્યો છે...")
    data = yf.download(tickers_to_download, period="1y", interval="1d", group_by="ticker", threads=True)

    out_rows = []

    bench_rets = {"1W": 0.0, "1M": 0.0, "3M": 0.0, "6M": 0.0}
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
        "SahilD": "", "SahilDAgo": "", "SahilW": "", "SahilWAgo": ""
    })

    for t in all_tickers:
        raw_sym = t.replace(".NS", "")
        sec, cap = meta.get(raw_sym, ("Others", "Small"))

        try:
            if t not in data:
                continue
            df = data[t].dropna()
            if len(df) < 50:
                continue

            c = df["Close"]
            v = df["Volume"]
            cur_price = round(float(c.iloc[-1]), 2)

            r1w = round(float((cur_price - c.iloc[-6]) / c.iloc[-6] * 100), 2) if len(c) >= 6 else 0.0
            r1m = round(float((cur_price - c.iloc[-22]) / c.iloc[-22] * 100), 2) if len(c) >= 22 else 0.0
            r3m = round(float((cur_price - c.iloc[-66]) / c.iloc[-66] * 100), 2) if len(c) >= 66 else 0.0
            r6m = round(float((cur_price - c.iloc[-130]) / c.iloc[-130] * 100), 2) if len(c) >= 130 else 0.0
            r1y = round(float((cur_price - c.iloc[0]) / c.iloc[0] * 100), 2) if len(c) >= 200 else r6m

            # Weighted RS Score: 40% (3M), 30% (6M), 30% (1Y)
            rs_score = round((0.4 * r3m) + (0.3 * r6m) + (0.3 * r1y), 2)

            ema20 = calculate_ema(c, 20).iloc[-1]
            ema50 = calculate_ema(c, 50).iloc[-1]
            ema100 = calculate_ema(c, 100).iloc[-1] if len(c) >= 100 else ema50
            ema200 = calculate_ema(c, 200).iloc[-1] if len(c) >= 200 else ema100

            s_ema20 = "Above" if cur_price >= ema20 else "Below"
            s_ema50 = "Above" if cur_price >= ema50 else "Below"
            s_ema100 = "Above" if cur_price >= ema100 else "Below"
            s_ema200 = "Above" if cur_price >= ema200 else "Below"

            rsi_series = calculate_rsi(c, 14)
            rsi_d = round(float(rsi_series.iloc[-1]), 1) if not np.isnan(rsi_series.iloc[-1]) else 50.0

            d_line, d_upper, d_lower = calculate_mkr(c, 14)
            d_line_val = round(float(d_line.iloc[-1]), 2)
            d_upper_val = round(float(d_upper.iloc[-1]), 2)
            d_lower_val = round(float(d_lower.iloc[-1]), 2)

            daily_sig = "BUY" if cur_price >= d_line_val else "SELL"
            d_conf = "Y" if (daily_sig == "BUY" and cur_price >= ema20) or (daily_sig == "SELL" and cur_price < ema20) else "N"

            # --- SAHIL BREAKOUT: DAILY CALCULATION ---
            high_20_d = c.iloc[-21:-1].max()
            low_20_d = c.iloc[-21:-1].min()
            vol_avg_20_d = v.iloc[-21:-1].mean()

            is_sahil_d_buy = (cur_price > high_20_d) and (v.iloc[-1] >= (1.5 * vol_avg_20_d))
            is_sahil_d_sell = (cur_price < low_20_d) and (v.iloc[-1] >= (1.5 * vol_avg_20_d))

            sahil_d = "BUY" if is_sahil_d_buy else ("SELL" if is_sahil_d_sell else "")
            sahil_d_ago = 0 if sahil_d else ""

            # --- SAHIL BREAKOUT: REAL WEEKLY CALCULATION (W-FRI) ---
            df_w = df.resample('W-FRI').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()

            sahil_w = ""
            sahil_w_ago = ""
            w_line_val = round(cur_price * 0.96, 2)
            w_upper_val = round(cur_price * 1.10, 2)
            w_lower_val = round(cur_price * 0.92, 2)
            weekly_sig = daily_sig
            w_conf = "Y"

            if len(df_w) >= 22:
                w_c = df_w['Close']
                w_v = df_w['Volume']
                w_cur = w_c.iloc[-1]
                w_high_20 = w_c.iloc[-21:-1].max()
                w_low_20 = w_c.iloc[-21:-1].min()
                w_vol_avg_20 = w_v.iloc[-21:-1].mean()

                is_sahil_w_buy = (w_cur > w_high_20) and (w_v.iloc[-1] >= (1.5 * w_vol_avg_20))
                is_sahil_w_sell = (w_cur < w_low_20) and (w_v.iloc[-1] >= (1.5 * w_vol_avg_20))

                sahil_w = "BUY" if is_sahil_w_buy else ("SELL" if is_sahil_w_sell else "")
                sahil_w_ago = 0 if sahil_w else ""

                w_line_s, w_up_s, w_lo_s = calculate_mkr(w_c, 14)
                w_line_val = round(float(w_line_s.iloc[-1]), 2)
                w_upper_val = round(float(w_up_s.iloc[-1]), 2)
                w_lower_val = round(float(w_lo_s.iloc[-1]), 2)
                weekly_sig = "BUY" if w_cur >= w_line_val else "SELL"
                w_conf = "Y"

            rs_pos = "Y" if r1m > bench_rets["1M"] else "N"
            rs_state = "Rising" if r1m >= r3m else "Falling"

            out_rows.append({
                "Symbol": raw_sym,
                "Industry": sec,
                "Cap": cap,
                "Price": cur_price,
                "1W": r1w,
                "1M": r1m,
                "3M": r3m,
                "6M": r6m,
                "Daily": daily_sig,
                "Weekly": weekly_sig,
                "DFresh": "★" if (len(c) >= 2 and c.iloc[-2] < d_line.iloc[-2] and cur_price >= d_line_val) else "",
                "WFresh": "",
                "DCross": "UP" if cur_price >= d_line_val else "DN",
                "DCrossAgo": 0,
                "WCross": "UP" if weekly_sig == "BUY" else "DN",
                "WCrossAgo": 0,
                "DTurnAgo": 1,
                "WTurnAgo": 1,
                "EMA20": s_ema20,
                "EMA50": s_ema50,
                "EMA100": s_ema100,
                "EMA200": s_ema200,
                "RSState": rs_state,
                "RSPos": rs_pos,
                "RSCross": "UP" if rs_pos == "Y" else "DN",
                "RSCrossAgo": 1,
                "RS_Score": rs_score,
                "RS_Rank": 50,
                "DelivPct": 45,
                "DelivHigh": "N",
                "RSI_D": rsi_d,
                "RSI_W": rsi_d,
                "RSI_M": 50,
                "RSI40Cross": "UP" if rsi_d > 40 else "",
                "RSI40CrossAgo": 1,
                "RSI50Cross": "UP" if rsi_d > 50 else "",
                "RSI50CrossAgo": 1,
                "RSI60Cross": "UP" if rsi_d > 60 else "",
                "RSI60CrossAgo": 1,
                "DConfirmed": d_conf,
                "WConfirmed": w_conf,
                "DLine": d_line_val,
                "DUpper": d_upper_val,
                "DLower": d_lower_val,
                "WLine": w_line_val,
                "WUpper": w_upper_val,
                "WLower": w_lower_val,
                "SahilD": sahil_d,
                "SahilDAgo": sahil_d_ago,
                "SahilW": sahil_w,
                "SahilWAgo": sahil_w_ago
            })
        except Exception as err:
            print(f"Error processing {t}: {err}")
            continue

    df_out = pd.DataFrame(out_rows)

    # 0 to 100 Percentile Relative Strength Rank
    mask = df_out["Symbol"] != "NIFTY 500"
    if mask.sum() > 0:
        df_out.loc[mask, "RS_Rank"] = (
            df_out.loc[mask, "RS_Score"].rank(pct=True) * 100
        ).round(0).astype(int)

    df_out.to_csv("flow.csv", index=False)
    print(f"flow.csv તૈયાર થઈ ગયું છે! (કુલ {len(df_out)} રેકોર્ડ્સ, RS Rank 0-100 ગણાઈ ગયું)")

    update_nsdl_json()

if __name__ == "__main__":
    main()
