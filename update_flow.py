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
    d, df_, dc = mkr(close[sym], 14, 5, 3); w, wf, wc = mkr(weekly[sym], 5, 2, 2)
    sig[sym] = {'Daily': d, 'Weekly': w, 'DFresh': df_, 'WFresh': wf, 'DCross': dc, 'WCross': wc}
sig = pd.DataFrame(sig).T

rets = pd.DataFrame({p: (close.iloc[-1] / close.iloc[-1 - d] - 1) * 100 for p, d in PERIODS}).round(1)
out = uni.set_index('Symbol').join(rets, how='inner').join(sig, how='left').rename_axis('Symbol').reset_index()
out['1W'] = out['1W']  # keep columns in order
out.to_csv('flow.csv', index=False)
print('saved', len(out), 'stocks; last price date', close.index[-1].date())
