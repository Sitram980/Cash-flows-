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
def mkr(series, bw, lookback=500):
    """Returns (regime, days since the line last turned, days since price last crossed the line + direction)."""
    y = series.dropna().values.astype(float)
    if len(y) < 60: return '', '', '', ''
    L = min(lookback, len(y)); w = np.exp(-np.arange(L) / bw)  # Laplace kernel
    line = np.convolve(y, w)[:len(y)] / np.cumsum(w)[np.minimum(np.arange(len(y)), L - 1)]
    up = np.diff(line) > 0
    turns = np.where(up[1:] != up[:-1])[0]
    turn_ago = (len(up) - 1 - turns[-1]) if len(turns) else ''
    above = y > line
    crosses = np.where(above[1:] != above[:-1])[0]
    if len(crosses):
        cross_ago = len(above) - 1 - crosses[-1]
        cross_dir = 'UP' if above[-1] else 'DN'
    else:
        cross_ago, cross_dir = '', ''
    return ('BUY' if up[-1] else 'SELL'), turn_ago, cross_ago, cross_dir

weekly = close.resample('W-FRI').last()
sig = {}
for sym in close.columns:
    d, dta, dca, dc = mkr(close[sym], 14)
    w, wta, wca, wc = mkr(weekly[sym], 5)
    sig[sym] = {'Daily': d, 'Weekly': w, 'DTurnAgo': dta, 'WTurnAgo': wta,
               'DCrossAgo': dca, 'DCross': dc, 'WCrossAgo': wca, 'WCross': wc}
sig = pd.DataFrame(sig).T

rets = pd.DataFrame({p: (close.iloc[-1] / close.iloc[-1 - d] - 1) * 100 for p, d in PERIODS}).round(1)
out = uni.set_index('Symbol').join(rets, how='inner').join(sig, how='left').rename_axis('Symbol').reset_index()
out['1W'] = out['1W']  # keep columns in order
out.to_csv('flow.csv', index=False)
print('saved', len(out), 'stocks; last price date', close.index[-1].date())

# ---- Try to auto-update NSDL sector flow data (nsdl.json) from a free API mirror ----
# If this fails for any reason, the existing nsdl.json is left untouched -- update it by hand instead.
import json as _json, re
from datetime import datetime as _dt
try:
    r2 = requests.get('https://fii-diidata.mrchartist.com/api/sectors', timeout=20,
                      headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                                             '(KHTML, like Gecko) Chrome/124.0 Safari/537.36',
                              'Referer': 'https://fii-diidata.mrchartist.com/',
                              'Accept': 'application/json'})
    r2.raise_for_status()
    api = r2.json()
    # NOTE: this API's exact field names are unverified -- adjust the keys below if the run fails,
    # by checking the Action log (it prints the raw response on error).
    items = api if isinstance(api, list) else (api.get('sectors') or api.get('data') or [])
    existing = _json.load(open('nsdl.json'))
    old_data = existing.get('data', {})
    new_data = {}
    last_date_raw = ''
    for it in items:
        name = it.get('name') or it.get('sector') or it.get('Industry')
        flow = it.get('fortnightCr', it.get('net_flow', it.get('flow', it.get('net'))))
        last_date_raw = it.get('lastDate', last_date_raw)
        # This API doesn't return AUC in Rs Cr directly, so keep the last known AUC for that sector
        # (only the net flow number is refreshed each fortnight).
        old_auc = old_data.get(name, [None, 0])[1] if name in old_data else 0
        if name and flow is not None:
            new_data[name] = [round(float(flow)), old_auc]
    if len(new_data) >= 15:   # sanity check: only accept if most sectors came through
        existing['data'] = new_data
        m = re.match(r'([A-Za-z]{3})(\d{1,2})(\d{4})', last_date_raw or '')
        existing['date'] = f'Fortnight ending {m.group(2)} {m.group(1)} {m.group(3)}' if m else (last_date_raw or _dt.now().strftime('%d %b %Y'))
        _json.dump(existing, open('nsdl.json', 'w'), indent=1)
        print(f'nsdl.json auto-updated: {len(new_data)} sectors, period: {existing["date"]}')
    else:
        print(f'NSDL auto-update skipped: only {len(new_data)} sectors parsed, response shape unexpected:')
        print('sample item:', str(items[0])[:500] if items else str(api)[:500])
except Exception as e:
    print(f'NSDL auto-update skipped ({e}); nsdl.json left as-is -- update it by hand if needed.')

