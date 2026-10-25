"""Builds flow.csv (Symbol, Industry, Cap, 1W, 1M, 3M, 6M) for the Capital Flow Map site."""
import io, requests, numpy as np, pandas as pd, yfinance as yf

BASE = 'https://nsearchives.nseindia.com/content/indices/'
CAP_FILES = {'Large': 'ind_nifty100list.csv', 'Mid': 'ind_niftymidcap150list.csv',
             'Small': 'ind_niftysmallcap250list.csv', 'Micro': 'ind_niftymicrocap250_list.csv'}
PERIODS = [('1W', 5), ('1M', 21), ('3M', 63), ('6M', 126)]

s = requests.Session()
s.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36',
                  'Referer': 'https://www.nseindia.com/'})
s.get('https://www.nseindia.com', timeout=15)

# ---- NSE delivery data (sec_bhavdata_full): delivery % and quantity for every stock ----
# NSE publishes this a few hours after close, so if today's isn't out yet, this tries recent days back.
def fetch_delivery():
    base = 'https://archives.nseindia.com/products/content/'
    for days_back in range(0, 7):
        d = pd.Timestamp.now() - pd.Timedelta(days=days_back)
        fname = f"sec_bhavdata_full_{d.strftime('%d%m%Y')}.csv"
        try:
            r = s.get(base + fname, timeout=20)
            if r.status_code != 200 or len(r.text) < 1000:
                continue
            df = pd.read_csv(io.StringIO(r.text))
            df.columns = [c.strip() for c in df.columns]
            for c in ['SYMBOL', 'SERIES']:
                df[c] = df[c].astype(str).str.strip()
            df = df[df['SERIES'] == 'EQ'].copy()
            df['DELIV_PER'] = pd.to_numeric(df['DELIV_PER'], errors='coerce')
            df['DELIV_QTY'] = pd.to_numeric(df['DELIV_QTY'].astype(str).str.replace(',', ''), errors='coerce')
            print(f'Delivery data: {fname}, {len(df)} EQ rows')
            return df.set_index('SYMBOL')[['DELIV_PER', 'DELIV_QTY']], d.strftime('%Y-%m-%d')
        except Exception:
            continue
    print('Delivery data: could not find a recent sec_bhavdata_full file; DelivPct will be blank')
    return None, None

deliv_df, deliv_date = fetch_delivery()

parts = []
for cap, f in CAP_FILES.items():
    try:
        r = s.get(BASE + f, timeout=30); r.raise_for_status()
        d = pd.read_csv(io.StringIO(r.text))[['Symbol', 'Industry']]
        d['Symbol'] = d['Symbol'].str.strip(); d['Cap'] = cap; parts.append(d)
        print(cap, len(d))
    except Exception as e:
        print('skipped', cap, e)
assert parts, 'No NSE lists could be loaded'
uni = pd.concat(parts).drop_duplicates('Symbol').reset_index(drop=True)

cframes, hframes, lframes, vframes, syms = [], [], [], [], uni['Symbol'].tolist()
for i in range(0, len(syms), 100):
    raw = yf.download([x + '.NS' for x in syms[i:i+100]], period='3y', interval='1d',
                      auto_adjust=True, progress=False, threads=True)
    def pick(field):
        d = raw[field] if isinstance(raw.columns, pd.MultiIndex) else raw[[field]]
        d.columns = [str(x).replace('.NS', '') for x in d.columns]
        return d
    cframes.append(pick('Close')); hframes.append(pick('High')); lframes.append(pick('Low')); vframes.append(pick('Volume'))
close = pd.concat(cframes, axis=1).dropna(how='all')
high = pd.concat(hframes, axis=1).dropna(how='all')
low = pd.concat(lframes, axis=1).dropna(how='all')
volume = pd.concat(vframes, axis=1).dropna(how='all')

# ---- Relative Strength vs Nifty 500 (Mansfield-style RS, oscillates around 0) ----
try:
    bench = yf.Ticker('^CRSLDX').history(period='3y', interval='1d')['Close']
    bench.index = bench.index.tz_localize(None)
except Exception as e:
    bench = None
    print(f'RS benchmark download failed ({e}); RS columns will be blank')

def relative_strength(series, bench, rs_window=50, lookback=10):
    """Mansfield RS: (price/benchmark) relative to its own N-day average, in %.
    Returns (RS state 'Rising'/'Falling', bars since it crossed 0, cross direction, 'Y' if RS is positive right now)."""
    if bench is None:
        return '', '', '', ''
    s_ = series.dropna()
    b_ = bench.reindex(s_.index).ffill()
    ratio = (s_ / b_).dropna()
    if len(ratio) < rs_window + lookback + 2:
        return '', '', '', ''
    rs_sma = ratio.rolling(rs_window).mean()
    mansfield = ((ratio / rs_sma) - 1) * 100
    m = mansfield.dropna().values
    if len(m) < lookback + 2:
        return '', '', '', ''
    state = 'Rising' if m[-1] > m[-5] else 'Falling'
    sign = m > 0
    crosses = np.where(sign[1:] != sign[:-1])[0]
    if len(crosses):
        cross_ago = len(sign) - 1 - crosses[-1]
        cross_dir = 'UP' if sign[-1] else 'DN'
    else:
        cross_ago, cross_dir = '', ''
    return state, cross_ago, cross_dir, ('Y' if m[-1] > 0 else '')

# ---- Multi Kernel Regression [ChartPrime] signal (Gaussian kernel, non-repaint) ----
def mkr(series, bw, lookback=200):
    """Two-sided Laplace kernel regression (matches the indicator's Repaint=ON setting).
    Bandwidth 14 is used for both daily and weekly, same as the TradingView indicator --
    Pine's 'bandwidth' is a bar count, not a time count, so it doesn't change with timeframe.
    Returns (regime, bars since the line last turned, bars since price last crossed it + direction,
    'Y' if confirmed -- the line's slope AND price's position above/below it currently agree,
    the line's current price level, upper band, lower band -- for stop-loss/target suggestions)."""
    y = series.dropna().values.astype(float)
    n = len(y)
    if n < 20: return '', '', '', '', '', '', '', ''
    m = min(lookback, n)
    seg = y[-m:]
    idx = np.arange(m)
    U = np.abs(idx[:, None] - idx[None, :]) / bw
    W = np.exp(-U)
    line = (W @ seg) / W.sum(axis=1)
    up = np.diff(line) > 0
    turns = np.where(up[1:] != up[:-1])[0]
    turn_ago = (len(up) - 1 - turns[-1]) if len(turns) else ''
    above = seg > line
    crosses = np.where(above[1:] != above[:-1])[0]
    if len(crosses):
        cross_ago = len(above) - 1 - crosses[-1]
        cross_dir = 'UP' if above[-1] else 'DN'
    else:
        cross_ago, cross_dir = '', ''
    confirmed = 'Y' if up[-1] == above[-1] else ''   # both rising+above, or both falling+below
    std = float(np.std(seg - line))
    line_val = round(float(line[-1]), 2)
    upper, lower = round(line_val + 2 * std, 2), round(line_val - 2 * std, 2)
    return ('BUY' if up[-1] else 'SELL'), turn_ago, cross_ago, cross_dir, confirmed, line_val, upper, lower

def rsi_series(series, period=14):
    """Standard Wilder's RSI (0-100) as a full array. Empty array if there isn't enough history yet."""
    s_ = series.dropna()
    if len(s_) < period + 5:
        return np.array([])
    delta = s_.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs_ = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs_)).dropna().values

def rsi(series, period=14):
    """Just the latest RSI value, rounded. Blank if there isn't enough history yet."""
    r = rsi_series(series, period)
    return round(float(r[-1]), 1) if len(r) else ''

def rsi_level_crosses(r, levels=(40, 50, 60), lookback=10):
    """For each level, how many bars ago RSI last crossed it, and in which direction (within `lookback` bars)."""
    out = {}
    for lvl in levels:
        if len(r) < lookback + 2:
            out[lvl] = ('', '')
            continue
        above = r > lvl
        crosses = np.where(above[1:] != above[:-1])[0]
        if len(crosses) and (len(above) - 1 - crosses[-1]) <= lookback:
            cross_ago = len(above) - 1 - crosses[-1]
            cross_dir = 'UP' if above[-1] else 'DN'
            out[lvl] = (cross_ago, cross_dir)
        else:
            out[lvl] = ('', '')
    return out

def breakout_signal(c, h, l, v, n=20, vol_len=20, vol_mult=1.5, lookback=10):
    """Port of 'SAHIL Breakout Vol Pro' (Pine v6): a fresh BUY fires the bar price closes above
    the highest high of the prior N bars with volume >= vol_mult x its N-bar average; SELL is the
    mirror image on the lows. Returns ('BUY'/'SELL'/'', bars since that fresh signal)."""
    n_ = min(len(c), len(h), len(l), len(v))
    if n_ < max(n, vol_len) + 5:
        return '', ''
    c, h, l, v = c[-n_:], h[-n_:], l[-n_:], v[-n_:]
    hiN = pd.Series(h).rolling(n).max().shift(1).values
    loN = pd.Series(l).rolling(n).min().shift(1).values
    avgVol = pd.Series(v).rolling(vol_len).mean().shift(1).values
    with np.errstate(invalid='ignore', divide='ignore'):
        volOk = (v / avgVol) >= vol_mult
    volOk = np.where(np.isnan(avgVol), True, volOk)   # if no volume history yet, don't block on it
    buyCond = (c > hiN) & volOk
    sellCond = (c < loN) & volOk
    buySig = buyCond & ~np.concatenate(([False], buyCond[:-1]))
    sellSig = sellCond & ~np.concatenate(([False], sellCond[:-1]))
    for i in range(len(c) - 1, max(len(c) - 1 - lookback, -1), -1):
        if buySig[i]: return 'BUY', len(c) - 1 - i
        if sellSig[i]: return 'SELL', len(c) - 1 - i
    return '', ''

weekly = close.resample('W-FRI').last()
weekly_high = high.resample('W-FRI').max()
weekly_low = low.resample('W-FRI').min()
weekly_vol = volume.resample('W-FRI').sum()
monthly = close.resample('ME').last()
EMA_SPANS = [20, 50, 100, 200]
sig = {}
for sym in close.columns:
    d, dta, dca, dc, dconf, dline, dupper, dlower = mkr(close[sym], 14, lookback=200)
    w, wta, wca, wc, wconf, wline, wupper, wlower = mkr(weekly[sym], 14, lookback=100)
    s_ = close[sym].dropna()
    rs_state, rs_cross_ago, rs_cross_dir, rs_pos = relative_strength(close[sym], bench)
    if sym in high.columns and sym in low.columns and sym in volume.columns:
        od = pd.DataFrame({'c': close[sym], 'h': high[sym], 'l': low[sym], 'v': volume[sym]}).dropna()
        sahil_d, sahil_d_ago = breakout_signal(od['c'].values, od['h'].values, od['l'].values, od['v'].values)
    else:
        sahil_d, sahil_d_ago = '', ''
    if sym in weekly_high.columns and sym in weekly_low.columns and sym in weekly_vol.columns:
        ow = pd.DataFrame({'c': weekly[sym], 'h': weekly_high[sym], 'l': weekly_low[sym], 'v': weekly_vol[sym]}).dropna()
        sahil_w, sahil_w_ago = breakout_signal(ow['c'].values, ow['h'].values, ow['l'].values, ow['v'].values)
    else:
        sahil_w, sahil_w_ago = '', ''
    row = {'Daily': d, 'Weekly': w, 'DTurnAgo': dta, 'WTurnAgo': wta,
          'DCrossAgo': dca, 'DCross': dc, 'WCrossAgo': wca, 'WCross': wc,
          'DConfirmed': dconf, 'WConfirmed': wconf,
          'DLine': dline, 'DUpper': dupper, 'DLower': dlower,
          'WLine': wline, 'WUpper': wupper, 'WLower': wlower,
          'Price': round(float(s_.iloc[-1]), 2) if len(s_) else '',
          'RSState': rs_state, 'RSCrossAgo': rs_cross_ago, 'RSCross': rs_cross_dir, 'RSPos': rs_pos,
          'SahilD': sahil_d, 'SahilDAgo': sahil_d_ago, 'SahilW': sahil_w, 'SahilWAgo': sahil_w_ago}
    r_d = rsi_series(close[sym])
    row['RSI_D'] = round(float(r_d[-1]), 1) if len(r_d) else ''
    row['RSI_W'] = rsi(weekly[sym])
    row['RSI_M'] = rsi(monthly[sym])
    xlevels = rsi_level_crosses(r_d)
    for lvl in (40, 50, 60):
        ago_, dir_ = xlevels[lvl]
        row[f'RSI{lvl}CrossAgo'], row[f'RSI{lvl}Cross'] = ago_, dir_
    for n in EMA_SPANS:
        if len(s_) >= n:
            row[f'EMA{n}'] = 'Above' if s_.iloc[-1] > s_.ewm(span=n, adjust=False).mean().iloc[-1] else 'Below'
        else:
            row[f'EMA{n}'] = ''
    sig[sym] = row
sig = pd.DataFrame(sig).T

rets = pd.DataFrame({p: (close.iloc[-1] / close.iloc[-1 - d] - 1) * 100 for p, d in PERIODS}).round(1)
out = uni.set_index('Symbol').join(rets, how='inner').join(sig, how='left').rename_axis('Symbol').reset_index()
if deliv_df is not None:
    out = out.merge(deliv_df.rename(columns={'DELIV_PER': 'DelivPct', 'DELIV_QTY': 'DelivQty'}),
                    left_on='Symbol', right_index=True, how='left')
    out['DelivPct'] = out['DelivPct'].round(1)
    out['DelivHigh'] = np.where(out['DelivPct'] > 60, 'Y', '')
else:
    out['DelivPct'], out['DelivQty'], out['DelivHigh'] = '', '', ''
out['1W'] = out['1W']  # keep columns in order
out.to_csv('flow.csv', index=False)
print('saved', len(out), 'stocks; last price date', close.index[-1].date())

# ---- Auto-update NSDL sector flow data (nsdl.json) ----
# Primary: CDSL's own fortnightly report page (official source, not blocked, gives real AUC too).
# Fallback: a free API mirror (net flow only, AUC carried forward from last known value).
# If both fail, the existing nsdl.json is left untouched -- update it by hand instead.
import json as _json, re
from datetime import datetime as _dt
BROWSER_HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                                 '(KHTML, like Gecko) Chrome/124.0 Safari/537.36'}

def existing_nsdl():
    return _json.load(open('nsdl.json'))

def save_nsdl(data, history, date_label, n_sectors, source):
    out_ = existing_nsdl()
    out_['data'] = data
    if history:
        out_['history'] = history
    out_['date'] = date_label
    _json.dump(out_, open('nsdl.json', 'w'), indent=1)
    print(f'nsdl.json auto-updated from {source}: {n_sectors} sectors, period: {date_label}')

def try_cdsl():
    """Parses CDSL's official fortnightly sector table directly. Each sector row has 96 numeric
    cells in 4 blocks (AUC start, net inv 1st half, net inv 2nd half, AUC end), each in INR Cr then
    USD Mn (12 values each): index 48 = Equity net flow for the latest half, index 72 = Equity AUC
    at period end. See the Action log for the parsed count if this ever needs re-checking."""
    idx = requests.get('https://www.cdslindia.com/Publications/ForeignPortInvestor.html', timeout=20, headers=BROWSER_HEADERS)
    idx.raise_for_status()
    m = re.search(r'href="(https://www\.cdslindia\.com/publications/FII/FortnightlySecWisePages/[^"]+\.html)"[^>]*>\s*([^<]+?)\s*<', idx.text, re.I)
    if not m:
        raise ValueError('could not find latest fortnightly report link on CDSL index page')
    page_url, date_label = m.group(1), m.group(2).strip()
    r = requests.get(page_url, timeout=20, headers=BROWSER_HEADERS)
    r.raise_for_status()
    html = r.text
    rows = re.findall(r'<tr[^>]*>(.*?)</tr>', html, re.S)
    data, history = {}, {}
    old_data = existing_nsdl().get('data', {})
    for row in rows:
        cells = [re.sub(r'<[^>]+>', '', c).replace('&nbsp;', ' ').strip() for c in re.findall(r'<td[^>]*>(.*?)</td>', row, re.S)]
        cells = [c for c in cells if c != '']
        if len(cells) < 2 or not re.match(r'^\d{1,2}$', cells[0]):
            continue
        name = cells[1]
        nums_raw = cells[2:]
        if not all(re.match(r'^-?[\d,]+$', c) for c in nums_raw) or len(nums_raw) < 73:
            continue
        nums = [int(c.replace(',', '')) for c in nums_raw]
        flow, auc = nums[48], nums[72]
        data[name] = [flow, auc]
        old_flow = old_data.get(name, [None])[0]
        history[name] = ([old_flow] if old_flow is not None else []) + [flow]
    if len(data) < 15:
        raise ValueError(f'only parsed {len(data)} sectors from CDSL table (layout may have changed)')
    save_nsdl(data, history, date_label, len(data), 'CDSL')

def try_mirror():
    r2 = requests.get('https://fii-diidata.mrchartist.com/api/sectors', timeout=20,
                      headers={**BROWSER_HEADERS, 'Referer': 'https://fii-diidata.mrchartist.com/', 'Accept': 'application/json'})
    r2.raise_for_status()
    api = r2.json()
    items = api if isinstance(api, list) else (api.get('sectors') or api.get('data') or [])
    old_data = existing_nsdl().get('data', {})
    new_data, history, last_date_raw = {}, {}, ''
    for it in items:
        name = it.get('name') or it.get('sector') or it.get('Industry')
        flow = it.get('fortnightCr', it.get('net_flow', it.get('flow', it.get('net'))))
        last_date_raw = it.get('lastDate', last_date_raw)
        old_auc = old_data.get(name, [None, 0])[1] if name in old_data else 0
        if name and flow is not None:
            new_data[name] = [round(float(flow)), old_auc]
        hist = it.get('historyCr')
        if name and isinstance(hist, list) and len(hist):
            history[name] = [round(float(x)) for x in hist[-10:]]
    if len(new_data) < 15:
        raise ValueError(f'only parsed {len(new_data)} sectors from mirror; sample: {str(items[0])[:300] if items else str(api)[:300]}')
    m = re.match(r'([A-Za-z]{3})(\d{1,2})(\d{4})', last_date_raw or '')
    date_label = f'Fortnight ending {m.group(2)} {m.group(1)} {m.group(3)}' if m else (last_date_raw or _dt.now().strftime('%d %b %Y'))
    save_nsdl(new_data, history, date_label, len(new_data), 'mirror API')

try:
    try_cdsl()
except Exception as e1:
    print(f'CDSL auto-update failed ({e1}); trying mirror API instead...')
    try:
        try_mirror()
    except Exception as e2:
        print(f'NSDL auto-update skipped (CDSL: {e1}; mirror: {e2}); nsdl.json left as-is -- update it by hand if needed.')

