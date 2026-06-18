"""
NDX-100 EV-Ranked Scanner — compute core.
Refactored from the one-shot compute.py into importable functions the server
calls on a schedule. Data source: yfinance (keyless, server-side; bypasses the
browser CORS block on Yahoo). No API keys anywhere in this deployment.

Public API:
    UNIVERSE                       -> list of NDX-100 tickers (constituent set)
    default_start()                -> 'YYYY-MM-DD' lookback start (~2y)
    fetch_ohlc(tickers, start)     -> {ticker: DataFrame[Open,High,Low,Close]}
    compute_scan(data)             -> {'as_of', 'windows': {N: [rows...]}}
    build_chart_ohlc(data, scan)   -> {ticker: [[date,o,h,l,c], ...]} (Top-3 union + QQQ)
    build_payload(data, scan)      -> {'as_of', 'windows', 'ohlc'}  (served at /api/scanner)
"""
import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Universe (NDX-100 constituent set, with display names). Same set the scanner
# has been built on. GOOGL/GOOG dual-class both included.
# ----------------------------------------------------------------------------
NAMES = {
'ADBE':'Adobe','AMD':'AMD','ABNB':'Airbnb','ALNY':'Alnylam','GOOGL':'Alphabet A','GOOG':'Alphabet C',
'AMZN':'Amazon','AEP':'American Electric Power','AMGN':'Amgen','ADI':'Analog Devices','AAPL':'Apple',
'AMAT':'Applied Materials','APP':'AppLovin','ARM':'Arm Holdings','ASML':'ASML','ADSK':'Autodesk','ADP':'ADP',
'AXON':'Axon Enterprise','BKR':'Baker Hughes','BKNG':'Booking','AVGO':'Broadcom','CDNS':'Cadence','CHTR':'Charter',
'CTAS':'Cintas','CSCO':'Cisco','CCEP':'Coca-Cola EP','CTSH':'Cognizant','CMCSA':'Comcast','CEG':'Constellation Energy',
'CPRT':'Copart','COST':'Costco','CRWD':'CrowdStrike','CSX':'CSX','DDOG':'Datadog','DXCM':'DexCom','FANG':'Diamondback',
'DASH':'DoorDash','EA':'Electronic Arts','EXC':'Exelon','FAST':'Fastenal','FER':'Ferrovial','FTNT':'Fortinet',
'GEHC':'GE HealthCare','GILD':'Gilead','HON':'Honeywell','IDXX':'Idexx','INSM':'Insmed','INTC':'Intel','INTU':'Intuit',
'ISRG':'Intuitive Surgical','KDP':'Keurig Dr Pepper','KLAC':'KLA','KHC':'Kraft Heinz','LRCX':'Lam Research','LIN':'Linde',
'LITE':'Lumentum','MAR':'Marriott','MRVL':'Marvell','MELI':'MercadoLibre','META':'Meta','MCHP':'Microchip','MU':'Micron',
'MSFT':'Microsoft','MSTR':'MicroStrategy','MDLZ':'Mondelez','MPWR':'Monolithic Power','MNST':'Monster','NFLX':'Netflix',
'NVDA':'Nvidia','NXPI':'NXP','ORLY':"O'Reilly",'ODFL':'Old Dominion','PCAR':'Paccar','PLTR':'Palantir','PANW':'Palo Alto',
'PAYX':'Paychex','PYPL':'PayPal','PDD':'PDD','PEP':'PepsiCo','QCOM':'Qualcomm','REGN':'Regeneron','ROP':'Roper',
'ROST':'Ross Stores','SNDK':'Sandisk','STX':'Seagate','SHOP':'Shopify','SBUX':'Starbucks','SNPS':'Synopsys',
'TMUS':'T-Mobile','TTWO':'Take-Two','TSLA':'Tesla','TXN':'Texas Instruments','TRI':'Thomson Reuters','VRSK':'Verisk',
'VRTX':'Vertex','WMT':'Walmart','WBD':'Warner Bros Discovery','WDC':'Western Digital','WDAY':'Workday','XEL':'Xcel Energy','ZS':'Zscaler'
}
UNIVERSE = list(NAMES.keys())

WINDOWS = [30, 60, 90, 120, 150, 180]
TRADING = 252
CHART_BARS = 520          # bars kept per ticker for the front-end candles (~2y; covers SMA200)
BENCH = 'QQQ'             # benchmark chart


def default_start():
    """~2 years of history: enough for the 180d window + SMA200 + warmup."""
    return (datetime.utcnow() - timedelta(days=760)).strftime('%Y-%m-%d')


# ----------------------------------------------------------------------------
# Fetch (yfinance). Returns {ticker: DataFrame[Open,High,Low,Close]} ascending.
# ----------------------------------------------------------------------------
def fetch_ohlc(tickers, start=None):
    import yfinance as yf
    start = start or default_start()
    tickers = list(dict.fromkeys(tickers))      # dedupe, keep order
    df = yf.download(tickers, start=start, interval='1d', auto_adjust=True,
                     progress=False, threads=True, group_by='ticker')
    out = {}
    for tk in tickers:
        try:
            sub = df[tk] if isinstance(df.columns, pd.MultiIndex) else df
        except Exception:
            continue
        sub = sub.dropna(subset=['Open', 'High', 'Low', 'Close'])
        if len(sub) == 0:
            continue
        out[tk] = sub[['Open', 'High', 'Low', 'Close']].copy()
    return out


# ----------------------------------------------------------------------------
# Metric helpers (verbatim from compute.py)
# ----------------------------------------------------------------------------
def runs_cumret(r):
    ups, downs = [], []
    i, n = 0, len(r)
    while i < n:
        if r[i] > 0:
            j = i
            while j < n and r[j] > 0: j += 1
            ups.append(np.prod(1.0 + r[i:j]) - 1.0); i = j
        elif r[i] < 0:
            j = i
            while j < n and r[j] < 0: j += 1
            downs.append(np.prod(1.0 + r[i:j]) - 1.0); i = j
        else:
            i += 1
    return ups, downs


def max_drawdown(closes):
    peak = closes[0]; mdd = 0.0
    for c in closes:
        if c > peak: peak = c
        dd = c / peak - 1.0
        if dd < mdd: mdd = dd
    return mdd


def ols_slope(y):
    n = len(y)
    if n < 3: return 0.0
    x = np.arange(n); xm, ym = x.mean(), y.mean()
    denom = ((x - xm) ** 2).sum()
    if denom == 0: return 0.0
    return float(((x - xm) * (y - ym)).sum() / denom)


# ----------------------------------------------------------------------------
# Scan (verbatim metric logic from compute.py, wrapped as a function)
# ----------------------------------------------------------------------------
def compute_scan(data):
    asof = max(v.index[-1] for v in data.values())
    asof_str = pd.Timestamp(asof).strftime('%Y-%m-%d')
    out = {'as_of': asof_str, 'windows': {}}

    for N in WINDOWS:
        rows = []
        roll_w = max(10, round(N / 4))
        for t, df in data.items():
            if t == BENCH:
                continue  # benchmark is charted, not ranked
            C = df['Close'].values.astype(float)
            O = df['Open'].values.astype(float)
            if len(C) < N + 2:
                continue
            ret_full = C[1:] / C[:-1] - 1.0
            r = ret_full[-N:]
            Cwin = C[-(N + 1):]
            Owin = O[-N:]
            Cprev = C[-(N + 1):-1]
            gaps = (Owin - Cprev) / Cprev
            green_body = np.mean(C[-N:] > Owin)
            p = float(np.mean(r > 0))
            up = r[r > 0]; dn = r[r < 0]
            avg_up = float(up.mean()) if up.size else 0.0
            avg_dn = float(-dn.mean()) if dn.size else 0.0
            ev = float(np.mean(r))
            period_ret = float(C[-1] / C[-(N + 1)] - 1.0)
            mdd = float(max_drawdown(Cwin))
            ups, downs = runs_cumret(r)
            up_swing = float(np.mean(ups)) if ups else 0.0
            pullback = float(np.mean(np.abs(downs))) if downs else 0.0
            avg_gap = float(np.mean(gaps))
            sd = float(np.std(r, ddof=1)) if r.size > 1 else 0.0
            sharpe = float(ev / sd * np.sqrt(TRADING)) if sd > 0 else 0.0
            neg = np.minimum(r, 0.0)
            dd_dev = float(np.sqrt(np.mean(neg ** 2)))
            sortino = float(ev / dd_dev * np.sqrt(TRADING)) if dd_dev > 0 else (sharpe if sd > 0 else 0.0)
            roll = [float(np.mean(r[i - roll_w + 1:i + 1])) for i in range(roll_w - 1, N)]
            roll = np.array(roll)
            ev_slope_bps = ols_slope(roll) * 10000.0
            spark = roll * 10000.0
            if len(spark) > 40:
                idx = np.linspace(0, len(spark) - 1, 40).round().astype(int)
                spark = spark[idx]
            spark = [round(float(x), 2) for x in spark]
            rows.append({
                't': t, 'name': NAMES.get(t, t),
                'last': round(float(C[-1]), 2),
                'ret': round(period_ret * 100, 2),
                'win': round(p * 100, 1),
                'green': round(float(green_body) * 100, 1),
                'aup': round(avg_up * 100, 3),
                'adn': round(avg_dn * 100, 3),
                'uswing': round(up_swing * 100, 2),
                'pull': round(pullback * 100, 2),
                'gap': round(avg_gap * 100, 3),
                'mdd': round(mdd * 100, 2),
                'ev': round(ev * 100, 4),
                'evbps': round(ev * 10000, 1),
                'evslope': round(ev_slope_bps, 3),
                'sharpe': round(sharpe, 2),
                'sortino': round(sortino, 2),
                'spark': spark,
            })
        rows.sort(key=lambda x: x['ev'], reverse=True)
        for i, row in enumerate(rows):
            row['rank'] = i + 1
        out['windows'][str(N)] = rows
    return out


# ----------------------------------------------------------------------------
# Chart OHLC subset: union of each window's EV Top-3, plus QQQ benchmark.
# ----------------------------------------------------------------------------
def build_chart_ohlc(data, scan, n_top=3):
    need = set([BENCH])
    for N in WINDOWS:
        rows = scan['windows'].get(str(N), [])
        for row in rows[:n_top]:
            need.add(row['t'])
    ohlc = {}
    for tk in need:
        df = data.get(tk)
        if df is None or len(df) == 0:
            continue
        sub = df.tail(CHART_BARS)
        bars = []
        for idx, row in sub.iterrows():
            bars.append([pd.Timestamp(idx).strftime('%Y-%m-%d'),
                         round(float(row['Open']), 2), round(float(row['High']), 2),
                         round(float(row['Low']), 2), round(float(row['Close']), 2)])
        ohlc[tk] = bars
    return ohlc


def build_payload(data, scan):
    return {'as_of': scan['as_of'], 'windows': scan['windows'],
            'ohlc': build_chart_ohlc(data, scan)}


# ============================================================================
# Build entrypoint (merged from build_data.py) — writes docs/scanner.json
# Run by the GitHub Actions workflow. Also usable locally: `python build.py`.
# ============================================================================
import json

def main():
    uni = UNIVERSE
    lim = os.environ.get('UNIVERSE_LIMIT')
    if lim:
        uni = uni[:int(lim)]
    start = os.environ.get('HIST_START') or default_start()

    data = fetch_ohlc(uni + [BENCH], start=start)
    if not data:
        raise SystemExit('fetch_ohlc returned no data; aborting (keeps last good deploy)')

    scan = compute_scan(data)
    payload = build_payload(data, scan)

    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, 'docs', 'scanner.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump(payload, f, separators=(',', ':'))

    kb = round(os.path.getsize(out) / 1024, 1)
    print('wrote', out, '(%s KB)' % kb)
    print('as_of', payload['as_of'], '| universe', len(payload['windows']['90']),
          '| chart tickers', sorted(payload['ohlc'].keys()))

if __name__ == '__main__':
    main()
