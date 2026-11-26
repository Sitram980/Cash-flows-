import os, io
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import requests
import yfinance as yf

IST = timezone(timedelta(hours=5, minutes=30))
NOW_IST = datetime.now(IST)

NSDL_SECTORS = {
    "Automobile and Auto Components", "Capital Goods", "Chemicals", "Construction",
    "Construction Materials", "Consumer Durables", "Consumer Services", "Diversified",
    "Fast Moving Consumer Goods", "Financial Services", "Forest Materials", "Healthcare",
    "Information Technology", "Media, Entertainment & Publication", "Metals & Mining",
    "Oil, Gas & Consumable Fuels", "Power", "Realty", "Services", "Telecommunication",
    "Textiles", "Utilities"
}

SECTOR_MAP = {
    "automobile": "Automobile and Auto Components", "auto": "Automobile and Auto Components",
    "capital goods": "Capital Goods", "industrials": "Capital Goods",
    "chemicals": "Chemicals", "construction materials": "Construction Materials",
    "construction": "Construction", "basic materials": "Metals & Mining",
    "consumer durables": "Consumer Durables", "consumer cyclical": "Consumer Durables",
    "consumer services": "Consumer Services", "diversified": "Diversified",
    "fast moving consumer goods": "Fast Moving Consumer Goods", "fmcg": "Fast Moving Consumer Goods",
    "consumer defensive": "Fast Moving Consumer Goods", "financial": "Financial Services",
    "forest": "Forest Materials", "healthcare": "Healthcare", "health care": "Healthcare",
    "information technology": "Information Technology", "technology": "Information Technology",
    "media": "Media, Entertainment & Publication", "communication": "Telecommunication",
    "metals": "Metals & Mining", "oil, gas": "Oil, Gas & Consumable Fuels",
    "oil gas": "Oil, Gas & Consumable Fuels", "energy": "Oil, Gas & Consumable Fuels",
    "power": "Power", "utilities": "Utilities", "realty": "Realty", "real estate": "Realty",
    "services": "Services", "telecommunication": "Telecommunication", "telecom": "Telecommunication",
    "textiles": "Textiles"
}

def map_sector(raw):
    s = str(raw or "").strip()
    if s in NSDL_SECTORS:
        return s
    k = s.lower().replace(",", "")
    for pat, target in SECTOR_MAP.items():
        if pat.replace(",", "") in k:
            return target
    return "Diversified"

def calc_rsi(s, period=14):
    if s is None or len(s) < period + 2:
        return np.nan
    d = s.diff()
    ag = d.clip(lower=0.0).ewm(alpha=1.0/period, min_periods=period, adjust=False).mean()
    al = (-d.clip(upper=0.0)).ewm(alpha=1.0/period, min_periods=period, adjust=False).mean()
    rsi = 100.0 - (100.0 / (1.0 + ag / al.replace(0.0, np.nan)))
    v = rsi.iloc[-1]
    return round(float(v), 1) if pd.notna(v) else np.nan

def calc_atr(h, l, c, period=14):
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    v = tr.ewm(alpha=1.0/period, min_periods=period, adjust=False).mean().iloc[-1]
    return float(v) if pd.notna(v) else float((h - l).iloc[-1])

def kernel_reg(prices, h=8.0, r=8.0):
    y = np.asarray(prices, dtype=float)[-120:]
    idx = np.arange(len(y), dtype=float)
    diff = idx[:, None] - idx[None, :]
    w = (1.0 + (diff ** 2) / (2.0 * r * (h ** 2))) ** (-r)
    return (w @ y) / w.sum(axis=1)

def analyze_mkr(cs, h=8.0):
    y = np.asarray(cs.dropna().values, dtype=float)
    if len(y) < 15:
        return "SELL", "N", "", "", "", np.nan
    yh = kernel_reg(y, h=h)
    dirs = np.where(np.diff(yh) >= 0, "BUY", "SELL")
    curr = dirs[-1]
    tago = 0
    for i in range(len(dirs) - 2, -1, -1):
        if dirs[i] != curr:
            break
        tago += 1
    if len(dirs) > 1 and all(d == curr for d in dirs):
        tago = len(dirs) - 1
    m = min(len(y), len(yh))
    ab = y[-m:] > yh[-m:]
    cdir, cago = "", ""
    for k in range(m - 1, 0, -1):
        if ab[k] != ab[k - 1]:
            cdir, cago = ("UP" if ab[k] else "DN"), (m - 1) - k
            break
    fresh = "Y" if (tago <= 3 or (cago != "" and cago <= 3)) else "N"
    return curr, fresh, tago, cdir, cago, float(yh[-1])

def analyze_sahil(hs, ls, cs, elen=55, xlen=20):
    h, l, c = hs.values, ls.values, cs.values
    n = len(c)
    if n < elen + 2:
        return "", "", "", ""
    st, lbar, bo, tsl = "", -1, "", ""
    for i in range(max(elen, n - 90), n):
        hi55, lo20 = np.max(h[i-elen:i]), np.min(l[i-xlen:i])
        if c[i] > hi55:
            if st != "BUY":
                st, lbar, bo = "BUY", i, round(float(hi55), 2)
            tsl = round(float(lo20), 2)
        elif c[i] < lo20:
            if st != "SELL":
                st, lbar, bo = "SELL", i, round(float(lo20), 2)
            tsl = round(float(np.max(h[i-xlen:i])), 2)
        elif st == "BUY":
            tsl = round(float(lo20), 2)
        elif st == "SELL":
            tsl = round(float(np.max(h[i-xlen:i])), 2)
    return st, ((n - 1 - lbar) if lbar >= 0 else ""), bo, tsl

def fetch_nse_delivery():
    dmap = {}
    sess = requests.Session()
    sess.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/128.0.0.0", "Referer": "https://www.nseindia.com/"})
    for d in range(8):
        dt = NOW_IST - timedelta(days=d)
        if dt.weekday() >= 5:
            continue
        ds = dt.strftime("%d%m%Y")
        try:
            r = sess.get(f"https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{ds}.csv", timeout=12)
            if r.status_code == 200 and "DELIV_PER" in r.text:
                dfb = pd.read_csv(io.StringIO(r.text))
                dfb.columns = [c.strip() for c in dfb.columns]
                if "SERIES" in dfb.columns:
                    dfb = dfb[dfb["SERIES"].astype(str).str.strip().isin(["EQ", "BE"])]
                for _, rw in dfb.iterrows():
                    sym = str(rw.get("SYMBOL", "")).strip().upper()
                    val = pd.to_numeric(str(rw.get("DELIV_PER", "")).strip(), errors="coerce")
                    if sym and pd.notna(val) and 0 < val <= 100:
                        dmap[sym] = round(float(val), 1)
                if len(dmap) > 100:
                    return dmap
        except Exception:
            pass
        try:
            r = sess.get(f"https://nsearchives.nseindia.com/archives/equities/mto/MTO_{ds}.DAT", timeout=12)
            if r.status_code == 200 and "20," in r.text:
                for ln in r.text.splitlines():
                    p = [x.strip() for x in ln.split(",")]
                    if len(p) >= 7 and p[0] == "20" and p[3] in ("EQ", "BE"):
                        val = float(p[-1])
                        if 0 < val <= 100:
                            dmap[p[2].upper()] = round(val, 1)
                if len(dmap) > 100:
                    return dmap
        except Exception:
            pass
    return dmap

def get_symbols():
    syms = {}
    for u in ["https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv", "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"]:
        try:
            r = requests.get(u, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
            if r.status_code == 200 and "Symbol" in r.text:
                df = pd.read_csv(io.StringIO(r.text))
                for idx, rw in df.iterrows():
                    s = str(rw["Symbol"]).strip().upper()
                    if s and s != "NAN":
                        syms[s] = (map_sector(rw.get("Industry", "")), "Large" if idx < 100 else ("Mid" if idx < 250 else "Small"))
                if len(syms) >= 400:
                    break
        except Exception:
            pass
    if len(syms) < 100 and os.path.exists("flow.csv"):
        old = pd.read_csv("flow.csv")
        for _, rw in old.iterrows():
            s = str(rw.get("Symbol", "")).strip().upper()
            if s and not s.startswith("NIFTY"):
                syms[s] = (map_sector(rw.get("Industry", "")), str(rw.get("Cap", "Mid")).strip() or "Mid")
    if os.path.exists("extra_symbols.txt"):
        with open("extra_symbols.txt", encoding="utf-8") as f:
            for ln in f:
                s = ln.strip().upper().replace(".NS", "")
                if s and not s.startswith("#") and s not in syms:
                    syms[s] = ("Diversified", "Small")
    return syms

def pct_ret(s, b):
    if len(s) <= b:
        return np.nan
    p, c = float(s.iloc[-1 - b]), float(s.iloc[-1])
    return round((c / p - 1.0) * 100.0, 2) if p > 0 else np.nan

def main():
    sym_meta = get_symbols()
    deliv_live = fetch_nse_delivery()
    deliv_prev = {}
    if os.path.exists("flow.csv"):
        try:
            old = pd.read_csv("flow.csv")
            for _, rw in old.iterrows():
                s = str(rw.get("Symbol", "")).strip().upper()
                v = pd.to_numeric(rw.get("DelivPct", np.nan), errors="coerce")
                if s and pd.notna(v) and v != 45.0 and 0 < v <= 100:
                    deliv_prev[s] = round(float(v), 1)
        except Exception:
            pass

    tickers = ["^CRSLDX"] + [f"{s}.NS" for s in sym_meta]
    data = yf.download(tickers, period="18mo", interval="1d", group_by="ticker", auto_adjust=True, threads=True, progress=False)

    def get_df(t):
        try:
            df = data[t].dropna(subset=["Close"]) if isinstance(data.columns, pd.MultiIndex) else data.dropna(subset=["Close"])
            return df if len(df) >= 30 else None
        except Exception:
            return None

    bdf = get_df("^CRSLDX")
    bclose = bdf["Close"] if bdf is not None else None
    out_rows = []
    if bclose is not None:
        out_rows.append({"Symbol": "NIFTY 500", "Industry": "Index", "Cap": "", "Price": round(float(bclose.iloc[-1]), 2),
                         "1W": pct_ret(bclose, 5), "1M": pct_ret(bclose, 21), "3M": pct_ret(bclose, 63), "6M": pct_ret(bclose, 126)})

    mrs_raw = {}
    for sym, (ind, cap) in sym_meta.items():
        df = get_df(f"{sym}.NS")
        if df is None:
            continue
        c, h, l = df["Close"], df["High"], df["Low"]
        vol = df["Volume"] if "Volume" in df.columns else pd.Series(0, index=c.index)
        px = round(float(c.iloc[-1]), 2)
        r1w, r1m, r3m, r6m = pct_ret(c, 5), pct_ret(c, 21), pct_ret(c, 63), pct_ret(c, 126)

        wc = c.resample("W-FRI").last().dropna()
        wh = h.resample("W-FRI").max().dropna()
        wl = l.resample("W-FRI").min().dropna()
        mc = c.resample("M").last().dropna()

        dsig, dfresh, dtago, dcross, dcago, dline = analyze_mkr(c, 8.0)
        wsig, wfresh, wtago, wcross, wcago, _ = analyze_mkr(wc, 8.0)
        atr = calc_atr(h, l, c, 14)

        rs_pos, rs_state = "N", "Falling"
        mrs_score = r3m if pd.notna(r3m) else (r1m if pd.notna(r1m) else 0.0)
        if bclose is not None:
            al = pd.concat([c.rename("s"), bclose.rename("b")], axis=1).dropna()
            if len(al) >= 60:
                rat = al["s"] / al["b"]
                mrs = ((rat / rat.rolling(252, min_periods=60).mean()) - 1.0) * 100.0
                mc_rs = mrs.dropna()
                if len(mc_rs) >= 6:
                    rs_pos = "Y" if float(mc_rs.iloc[-1]) > 0 else "N"
                    rs_state = "Rising" if float(mc_rs.iloc[-1]) > float(mc_rs.iloc[-6]) else "Falling"
                    mrs_score = float(mc_rs.iloc[-1])

        if sym in deliv_live:
            dp = deliv_live[sym]
        elif sym in deliv_prev:
            dp = deliv_prev[sym]
        else:
            v20 = float(vol.iloc[-20:].mean()) if len(vol) >= 20 else 0.0
            vr = (float(vol.iloc[-1]) / v20) if v20 > 0 else 1.0
            dp = round(min(85.0, max(25.0, 42.0 + (vr - 1.0) * 12.0)), 1)

        sd, sd_ago, sd_bo, sd_tsl = analyze_sahil(h, l, c, 55, 20)
        sw, sw_ago, sw_bo, sw_tsl = analyze_sahil(wh, wl, wc, 55, 20)
        ep = lambda sp: ("Above" if px >= float(c.ewm(span=sp, adjust=False).mean().iloc[-1]) else "Below") if len(c) >= sp else ""

        mrs_raw[sym] = mrs_score
        out_rows.append({
            "Symbol": sym, "Industry": ind, "Cap": cap, "Price": px,
            "1W": r1w, "1M": r1m, "3M": r3m, "6M": r6m,
            "Daily": dsig, "Weekly": wsig, "DFresh": dfresh, "WFresh": wfresh,
            "DTurnAgo": dtago, "WTurnAgo": wtago, "DCross": dcross, "DCrossAgo": dcago,
            "WCross": wcross, "WCrossAgo": wcago,
            "EMA20": ep(20), "EMA50": ep(50), "EMA100": ep(100), "EMA200": ep(200),
            "RSState": rs_state, "RSPos": rs_pos, "RS_Rank": 50,
            "DelivPct": dp, "DelivHigh": "Y" if dp >= 55.0 else "N",
            "RSI_D": calc_rsi(c, 14), "RSI_W": calc_rsi(wc, 14), "RSI_M": calc_rsi(mc, 14),
            "DConfirmed": "Y" if (pd.notna(dline) and ((dsig == "BUY" and px >= dline) or (dsig == "SELL" and px <= dline))) else "N",
            "WConfirmed": "Y" if len(wc) >= 2 and ((wsig == "BUY" and wc.iloc[-1] >= wc.iloc[-2]) or (wsig == "SELL" and wc.iloc[-1] <= wc.iloc[-2])) else "N",
            "DLine": round(dline, 2) if pd.notna(dline) else "",
            "DUpper": round(dline + 2.0 * atr, 2) if pd.notna(dline) else "",
            "DLower": round(max(0.01, dline - 1.5 * atr), 2) if pd.notna(dline) else "",
            "SahilD": sd, "SahilDAgo": sd_ago, "SahilD_BO": sd_bo, "SahilD_TSL": sd_tsl,
            "SahilW": sw, "SahilWAgo": sw_ago, "SahilW_BO": sw_bo, "SahilW_TSL": sw_tsl,
        })

    if mrs_raw:
        srk = pd.Series(mrs_raw).rank(pct=True) * 98.0 + 1.0
        for rw in out_rows:
            if rw["Symbol"] in srk:
                rw["RS_Rank"] = int(round(srk[rw["Symbol"]]))

    pd.DataFrame(out_rows).to_csv("flow.csv", index=False)
    print(f"Done: {len(out_rows)} rows written.")

if __name__ == "__main__":
    main()
    
