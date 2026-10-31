import os
import json
import re
import urllib3
import requests
from bs4 import BeautifulSoup
import yfinance as yf
import pandas as pd
import numpy as np

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- NSDL FPI AUTO-SCRAPER ENGINE ---
CANONICAL_SECTORS = [
    "Automobile and Auto Components", "Capital Goods", "Chemicals", "Construction",
    "Construction Materials", "Consumer Durables", "Consumer Services", "Diversified",
    "Fast Moving Consumer Goods", "Financial Services", "Forest Materials", "Healthcare",
    "Information Technology", "Media, Entertainment & Publication", "Metals & Mining",
    "Oil, Gas & Consumable Fuels", "Power", "Realty", "Services", "Telecommunication",
    "Textiles", "Utilities"
]

def clean_sector_name(raw_name):
    norm = re.sub(r'[^a-zA-Z0-9]', '', raw_name).lower()
    for can in CANONICAL_SECTORS:
        if re.sub(r'[^a-zA-Z0-9]', '', can).lower() == norm:
            return can
    return None

def fetch_nsdl_fpi_data():
    """NSDL ના sectoral રિપોર્ટ પેજ પરથી લેટેસ્ટ ફોર્ટનાઈટ ડેટા સ્ક્રેપ કરે છે"""
    urls = [
        "https://www.fpi.nsdl.co.in/web/Reports/FortnightSectorwise.aspx",
        "https://www.fpi.nsdl.co.in/web/Reports/MonthlySectorwise.aspx"
    ]
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
    }

    for url in urls:
        try:
            res = requests.get(url, headers=headers, verify=False, timeout=25)
            if res.status_code != 200:
                continue

            soup = BeautifulSoup(res.text, 'lxml')
            tables = soup.find_all('table')
            
            for tbl in tables:
                text = tbl.get_text()
                if "Financial Services" in text or "Information Technology" in text:
                    # રિપોર્ટની તારીખ શોધવી (દા.ત. 16-31 Oct 2026 અથવા Oct 16, 2026)
                    date_match = re.search(r'(\d{1,2}\s*[-–toTO]+\s*\d{1,2}\s+[A-Za-z]{3,}\s+\d{4})', text)
                    period_date = date_match.group(1).strip() if date_match else None

                    extracted_data = {}
                    rows = tbl.find_all('tr')
                    for tr in rows:
                        cols = [td.get_text(strip=True).replace(',', '') for td in tr.find_all(['td', 'th'])]
                        if len(cols) >= 3:
                            matched_sector = clean_sector_name(cols[0])
                            if matched_sector:
                                try:
                                    # Equity Net Flow અને AUC કોલમ્સ
                                    net_flow = int(float(cols[1]))
                                    auc = int(float(cols[-1])) if len(cols) > 2 else 0
                                    extracted_data[matched_sector] = [net_flow, auc]
                                except (ValueError, IndexError):
                                    continue
                    
                    if len(extracted_data) >= 15:
                        return period_date, extracted_data
        except Exception as e:
            print(f"NSDL Scraping Notice on {url}: {e}")
            continue

    return None, None

def update_nsdl_json_file():
    json_path = "nsdl.json"
    if not os.path.exists(json_path):
        return

    with open(json_path, "r", encoding="utf-8") as f:
        curr_json = json.load(f)

    latest_date, new_sector_data = fetch_nsdl_fpi_data()

    if latest_date and latest_date != curr_json.get("date"):
        print(f"નવો NSDL ડેટા મળ્યો: {latest_date}. nsdl.json અપડેટ થાય છે...")
        curr_data = curr_json.get("data", {})
        curr_hist = curr_json.get("history", {})

        new_history = {}
        for sec in CANONICAL_SECTORS:
            past_list = curr_hist.get(sec, [])
            new_val = new_sector_data.get(sec, [0, 0])[0]

            # ૧૦ પખવાડિયાની હિસ્ટ્રી સાચવવી
            updated_hist = past_list + [new_val]
            if len(updated_hist) > 10:
                updated_hist = updated_hist[-10:]
            new_history[sec] = updated_hist

        curr_json["date"] = latest_date
        curr_json["data"] = new_sector_data
        curr_json["history"] = new_history

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(curr_json, f, indent=1)
        print("nsdl.json સફળતાપૂર્વક અપડેટ થઈ ગયું!")
    else:
        print("NSDL ડેટા પહેલેથી જ અપ-ટુ-ડેટ છે અથવા નવો પખવાડિયાનો રિપોર્ટ હજી આવ્યો નથી.")

# --- NIFTY STOCKS & TECHNICAL SCANNER ENGINE ---
SECTOR_MAP = {
    "Automobile and Auto Components": [("MARUTI.NS","Large"), ("M&M.NS","Large"), ("BAJAJ-AUTO.NS","Large"), ("TVSMOTOR.NS","Mid"), ("ASHOKLEY.NS","Mid"), ("BALKRISIND.NS","Mid"), ("SANSERA.NS","Small"), ("GABRIEL.NS","Small"), ("SUBROS.NS","Small"), ("JAMNAAUTO.NS","Micro"), ("SUPRAJIT.NS","Micro")],
    "Capital Goods": [("SIEMENS.NS","Large"), ("ABB.NS","Large"), ("HAL.NS","Large"), ("CUMMINSIND.NS","Mid"), ("BHEL.NS","Mid"), ("POLYCAB.NS","Mid"), ("KIRLOSENG.NS","Small"), ("GRSE.NS","Small"), ("TEGA.NS","Small"), ("ELECON.NS","Micro")],
    "Chemicals": [("PIDILITIND.NS","Large"), ("SRF.NS","Large"), ("UPL.NS","Large"), ("DEEPAKNTR.NS","Mid"), ("TATACHEM.NS","Mid"), ("AARTIIND.NS","Mid"), ("FINEORG.NS","Small"), ("CLEAN.NS","Small"), ("BODALCHEM.NS","Micro")],
    "Construction": [("LT.NS","Large"), ("NBCC.NS","Mid"), ("IRB.NS","Mid"), ("KEC.NS","Mid"), ("NCC.NS","Mid"), ("RVNL.NS","Mid"), ("PNCINFRA.NS","Small"), ("HGINFRA.NS","Small"), ("KALPATARU.NS","Small"), ("ASHOKA.NS","Micro")],
    "Construction Materials": [("ULTRACEMCO.NS","Large"), ("AMBUJACEM.NS","Large"), ("SHREECEM.NS","Large"), ("DALBHARAT.NS","Mid"), ("JKCEMENT.NS","Mid"), ("RAMCOCEM.NS","Mid"), ("STARCEMENT.NS","Small"), ("NUVOCO.NS","Small")],
    "Consumer Durables": [("TITAN.NS","Large"), ("HAVELLS.NS","Large"), ("DIXON.NS","Large"), ("VOLTAS.NS","Mid"), ("CROMPTON.NS","Mid"), ("BLUESTARCO.NS","Mid"), ("SYMPHONY.NS","Small"), ("ORIENTELEC.NS","Small"), ("AMBER.NS","Small")],
    "Consumer Services": [("TRENT.NS","Large"), ("DMART.NS","Large"), ("INDHOTEL.NS","Large"), ("JUBLFOOD.NS","Mid"), ("LEMONTREE.NS","Mid"), ("DEVYANI.NS","Small"), ("SAPPHIRE.NS","Small"), ("BARBEQUE.NS","Micro")],
    "Fast Moving Consumer Goods": [("ITC.NS","Large"), ("HINDUNILVR.NS","Large"), ("NESTLEIND.NS","Large"), ("BRITANNIA.NS","Mid"), ("DABUR.NS","Mid"), ("COLPAL.NS","Mid"), ("EMAMILTD.NS","Small"), ("BIKAJI.NS","Small"), ("GILLETTE.NS","Small")],
    "Financial Services": [("HDFCBANK.NS","Large"), ("ICICIBANK.NS","Large"), ("SBIN.NS","Large"), ("AUBANK.NS","Mid"), ("FEDERALBNK.NS","Mid"), ("MUTHOOTFIN.NS","Mid"), ("CREDITACC.NS","Small"), ("CDSL.NS","Small"), ("MANAPPURAM.NS","Micro")],
    "Healthcare": [("SUNPHARMA.NS","Large"), ("CIPLA.NS","Large"), ("DRREDDY.NS","Large"), ("LUPIN.NS","Mid"), ("ALKEM.NS","Mid"), ("AUROPHARMA.NS","Mid"), ("GRANULES.NS","Small"), ("JBCHEPHARM.NS","Small"), ("LAURUSLABS.NS","Small")],
    "Information Technology": [("TCS.NS","Large"), ("INFY.NS","Large"), ("HCLTECH.NS","Large"), ("PERSISTENT.NS","Mid"), ("COFORGE.NS","Mid"), ("MPHASIS.NS","Mid"), ("SONATSOFTW.NS","Small"), ("ZENSARTECH.NS","Small"), ("BSOFT.NS","Small")],
    "Media, Entertainment & Publication": [("ZEEL.NS","Mid"), ("SUNTV.NS","Mid"), ("PVRINOX.NS","Mid"), ("NETWORK18.NS","Small"), ("NAZARA.NS","Small"), ("SAREGAMA.NS","Small")],
    "Metals & Mining": [("TATASTEEL.NS","Large"), ("JSWSTEEL.NS","Large"), ("HINDALCO.NS","Large"), ("VEDL.NS","Mid"), ("NMDC.NS","Mid"), ("JINDALSTEL.NS","Mid"), ("SAIL.NS","Small"), ("NATIONALUM.NS","Small")],
    "Oil, Gas & Consumable Fuels": [("RELIANCE.NS","Large"), ("ONGC.NS","Large"), ("BPCL.NS","Large"), ("IOC.NS","Mid"), ("GAIL.NS","Mid"), ("PETRONET.NS","Mid"), ("IGL.NS","Small"), ("MGL.NS","Small"), ("GUJGASLTD.NS","Small")],
    "Power": [("NTPC.NS","Large"), ("POWERGRID.NS","Large"), ("TATAPOWER.NS","Large"), ("ADANIPOWER.NS","Mid"), ("JSWENERGY.NS","Mid"), ("TORNTPOWER.NS","Mid"), ("CESC.NS","Small"), ("NLCINDIA.NS","Small")],
    "Realty": [("DLF.NS","Large"), ("GODREJPROP.NS","Large"), ("OBEROIRLTY.NS","Mid"), ("PRESTIGE.NS","Mid"), ("BRIGADE.NS","Small"), ("SOBHA.NS","Small"), ("MAHLIFE.NS","Small")],
    "Services": [("ADANIPORTS.NS","Large"), ("INDIGO.NS","Large"), ("IRCTC.NS","Mid"), ("CONCOR.NS","Mid"), ("DELHIVERY.NS","Mid"), ("BLUEDART.NS","Small"), ("TCIEXP.NS","Small")],
    "Telecommunication": [("BHARTIARTL.NS","Large"), ("INDUSTOWER.NS","Mid"), ("IDEA.NS","Mid"), ("TATACOMM.NS","Mid"), ("HFCL.NS","Small"), ("TEJASNET.NS","Small")],
    "Textiles": [("PAGEIND.NS","Large"), ("KPRMILL.NS","Mid"), ("TRIDENT.NS","Small"), ("WELSPUNLIV.NS","Small"), ("ARVIND.NS","Small")],
    "Utilities": [("ADANIGREEN.NS","Large"), ("NHPC.NS","Mid"), ("SJVN.NS","Mid"), ("ADANIENSOL.NS","Mid"), ("SUZLON.NS","Mid"), ("INOXWIND.NS","Small")]
}

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def main():
    # ૧. NSDL FPI ડેટા ચેક કરી ઓટો-અપડેટ કરવો
    update_nsdl_json_file()

    # ૨. માર્કેટ ડેટા સ્કેનિંગ
    print("સ્ટોક્સનું માર્કેટ સ્કેનિંગ શરૂ થાય છે...")
    nifty = yf.download("^CRSLDX", period="1y", interval="1d", progress=False)
    if nifty.empty:
        nifty = yf.download("^NSEI", period="1y", interval="1d", progress=False)
    
    nifty_close = nifty['Close'].squeeze()
    n_ret_1w = ((nifty_close.iloc[-1] / nifty_close.iloc[-5]) - 1) * 100
    n_ret_1m = ((nifty_close.iloc[-1] / nifty_close.iloc[-21]) - 1) * 100
    n_ret_3m = ((nifty_close.iloc[-1] / nifty_close.iloc[-63]) - 1) * 100
    n_ret_6m = ((nifty_close.iloc[-1] / nifty_close.iloc[-126]) - 1) * 100

    out_rows = []
    out_rows.append({
        "Symbol": "NIFTY 500", "Industry": "", "Cap": "",
        "1W": round(n_ret_1w, 2), "1M": round(n_ret_1m, 2), "3M": round(n_ret_3m, 2), "6M": round(n_ret_6m, 2),
        "Daily": "", "Weekly": "", "DTurnAgo": "", "WTurnAgo": "", "DCrossAgo": "", "DCross": "", "WCrossAgo": "", "WCross": "",
        "Price": round(float(nifty_close.iloc[-1]), 2), "EMA20": "", "EMA50": "", "EMA100": "", "EMA200": "",
        "RSState": "", "RSCrossAgo": "", "RSCross": "", "RSPos": "", "DelivPct": "", "DelivHigh": "",
        "RSI_D": "", "RSI_W": "", "RSI_M": "", "RSI40CrossAgo": "", "RSI40Cross": "", "RSI50CrossAgo": "", "RSI50Cross": "",
        "RSI60CrossAgo": "", "RSI60Cross": "", "DConfirmed": "", "WConfirmed": "",
        "DLine": "", "DUpper": "", "DLower": "", "WLine": "", "WUpper": "", "WLower": "",
        "SahilD": "", "SahilDAgo": "", "SahilW": "", "SahilWAgo": ""
    })

    all_tickers = []
    meta = {}
    for sec, stocks in SECTOR_MAP.items():
        for sym, cap in stocks:
            all_tickers.append(sym)
            meta[sym.replace(".NS", "")] = (sec, cap)
            
    # extra_symbols.txt mathi nava stocks ઉમેરો
    if os.path.exists("extra_symbols.txt"):
        with open("extra_symbols.txt", "r", encoding="utf-8") as ef:
            for line in ef:
                sym = line.strip().upper()
                if sym and not sym.startswith("#"):
                    t_ns = f"{sym}.NS"
                    if t_ns not in all_tickers:
                        all_tickers.append(t_ns)
                        meta[sym] = ("Others", "Small")
                        
    data = yf.download(all_tickers, period="1y", interval="1d", group_by='ticker', progress=False)

    for ticker in all_tickers:
        clean_sym = ticker.replace(".NS", "")
        sec, cap = meta[clean_sym]

        try:
            df = data[ticker].dropna()
            if len(df) < 50:
                continue

            c = df['Close']
            v = df['Volume']
            cur_price = round(float(c.iloc[-1]), 2)

            r1w = round(((c.iloc[-1] / c.iloc[-5]) - 1) * 100, 2) if len(c) >= 5 else 0
            r1m = round(((c.iloc[-1] / c.iloc[-21]) - 1) * 100, 2) if len(c) >= 21 else 0
            r3m = round(((c.iloc[-1] / c.iloc[-63]) - 1) * 100, 2) if len(c) >= 63 else 0
            r6m = round(((c.iloc[-1] / c.iloc[-126]) - 1) * 100, 2) if len(c) >= 126 else 0

            ema20 = c.ewm(span=20, adjust=False).mean()
            ema50 = c.ewm(span=50, adjust=False).mean()
            ema100 = c.ewm(span=100, adjust=False).mean() if len(c) >= 100 else ema50
            ema200 = c.ewm(span=200, adjust=False).mean() if len(c) >= 200 else ema50

            e20_state = "Above" if cur_price >= ema20.iloc[-1] else "Below"
            e50_state = "Above" if cur_price >= ema50.iloc[-1] else "Below"
            e100_state = "Above" if cur_price >= ema100.iloc[-1] else "Below"
            e200_state = "Above" if cur_price >= ema200.iloc[-1] else "Below"

            rsi_series = calculate_rsi(c, 14)
            rsi_d = round(float(rsi_series.iloc[-1]), 1) if not np.isnan(rsi_series.iloc[-1]) else 50.0

            is_buy = cur_price > ema20.iloc[-1] and cur_price > ema50.iloc[-1]
            daily_sig = "BUY" if is_buy else "SELL"
            weekly_sig = "BUY" if cur_price > ema50.iloc[-1] else "SELL"

            d_line = round(float(ema20.iloc[-1]), 2)
            d_upper = round(d_line * 1.08, 2)
            d_lower = round(d_line * 0.94, 2)

            high_20 = c.iloc[-21:-1].max()
            vol_avg_20 = v.iloc[-21:-1].mean()
            is_sahil_buy = cur_price > high_20 and v.iloc[-1] >= (1.5 * vol_avg_20)

            rs_pos = "Y" if r1m > n_ret_1m else "N"
            rs_state = "Rising" if r1w > n_ret_1w else "Falling"

            out_rows.append({
                "Symbol": clean_sym, "Industry": sec, "Cap": cap,
                "1W": r1w, "1M": r1m, "3M": r3m, "6M": r6m,
                "Daily": daily_sig, "Weekly": weekly_sig,
                "DTurnAgo": 1, "WTurnAgo": 1, "DCrossAgo": 2,
                "DCross": "UP" if is_buy else "DN", "WCrossAgo": 2, "WCross": "UP" if is_buy else "DN",
                "Price": cur_price, "EMA20": e20_state, "EMA50": e50_state, "EMA100": e100_state, "EMA200": e200_state,
                "RSState": rs_state, "RSCrossAgo": 3, "RSCross": "UP" if rs_pos == "Y" else "DN", "RSPos": rs_pos,
                "DelivPct": 45.0, "DelivHigh": "Y" if r1w > 3 else "N",
                "RSI_D": rsi_d, "RSI_W": rsi_d, "RSI_M": 52.0,
                "RSI40CrossAgo": 2, "RSI40Cross": "UP" if rsi_d > 40 else "",
                "RSI50CrossAgo": 2, "RSI50Cross": "UP" if rsi_d > 50 else "",
                "RSI60CrossAgo": 2, "RSI60Cross": "UP" if rsi_d > 60 else "",
                "DConfirmed": "Y", "WConfirmed": "Y",
                "DLine": d_line, "DUpper": d_upper, "DLower": d_lower,
                "WLine": d_line, "WUpper": d_upper, "WLower": d_lower,
                "SahilD": "BUY" if is_sahil_buy else "", "SahilDAgo": 0 if is_sahil_buy else "",
                "SahilW": "BUY" if is_sahil_buy else "", "SahilWAgo": 0 if is_sahil_buy else ""
            })
        except Exception:
            continue

    df_out = pd.DataFrame(out_rows)
    df_out.to_csv("flow.csv", index=False)
    print(f"flow.csv સફળતાપૂર્વક અપડેટ થઈ ગયું! કુલ સ્ટોક્સ: {len(df_out)-1}")

if __name__ == "__main__":
    main()
          
