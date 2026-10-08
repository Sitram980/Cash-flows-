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

frames, syms = [], uni['Symbol'].tolist()
for i in range(0, len(syms), 100):
    raw = yf.download([x + '.NS' for x in syms[i:i+100]], period='2y', interval='1d',
                      auto_adjust=True, progress=False, threads=True)
    c = raw['Close'] if isinstance(raw.columns, pd.MultiIndex) else raw[['Close']]
    c.columns = [str(x).replace('.NS', '') for x in c.columns]; frames.append(c)
close = pd.concat(frames, axis=1).dropna(how='all')

# ---- Multi Kernel Regression [ChartPrime] signal (Gaussian kernel, non-repaint) ----
def mkr(series, bw, lookback=200):
    """Two-sided Laplace kernel regression (matches the indicator's Repaint=ON setting).
    Bandwidth 14 is used for both daily and weekly, same as the TradingView indicator --
    Pine's 'bandwidth' is a bar count, not a time count, so it doesn't change with timeframe.
    Returns (regime, bars since the line last turned, bars since price last crossed it + direction)."""
    y = series.dropna().values.astype(float)
    n = len(y)
    if n < 20: return '', '', '', ''
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
    return ('BUY' if up[-1] else 'SELL'), turn_ago, cross_ago, cross_dir

weekly = close.resample('W-FRI').last()
EMA_SPANS = [20, 50, 100, 200]
sig = {}
for sym in close.columns:
    d, dta, dca, dc = mkr(close[sym], 14, lookback=200)
    w, wta, wca, wc = mkr(weekly[sym], 14, lookback=100)
    s_ = close[sym].dropna()
    row = {'Daily': d, 'Weekly': w, 'DTurnAgo': dta, 'WTurnAgo': wta,
          'DCrossAgo': dca, 'DCross': dc, 'WCrossAgo': wca, 'WCross': wc,
          'Price': round(float(s_.iloc[-1]), 2) if len(s_) else ''}
    for n in EMA_SPANS:
        if len(s_) >= n:
            row[f'EMA{n}'] = 'Above' if s_.iloc[-1] > s_.ewm(span=n, adjust=False).mean().iloc[-1] else 'Below'
        else:
            row[f'EMA{n}'] = ''
    sig[sym] = row
sig = pd.DataFrame(sig).T

rets = pd.DataFrame({p: (close.iloc[-1] / close.iloc[-1 - d] - 1) * 100 for p, d in PERIODS}).round(1)
out = uni.set_index('Symbol').join(rets, how='inner').join(sig, how='left').rename_axis('Symbol').reset_index()
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

