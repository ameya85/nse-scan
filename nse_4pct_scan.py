#!/usr/bin/env python3
"""
NSE 4% Momentum Burst Scanner (Stockbee-style) with HTML dashboard output.

Core scan (Pradeep Bonde):
    c/c1 >= 1.04  and  v > v1  and  v >= 100000

Quality filters layered on top (each scored, best setups float up):
    1. Close near high of day        (range position >= 0.65)
    2. Prior day negative or narrow  (prior close < prior open, or NR7)
    3. Orderly base of 5-40 days     (pre-breakout range <= BASE_MAX_RANGE_PCT)
    4. Young trend                   (not extended: close within EXT_MAX of 20d low)

India adaptations (configurable below):
    - Series EQ only
    - Minimum close price (skips penny junk that games the 100k share rule)
    - Minimum turnover in rupees crores

Data source: NSE daily bhavcopy archives (EOD, free, no login).
    Primary : UDiFF BhavCopy zip
    Fallback: sec_bhavdata_full csv
Files are cached in ./nse_cache so repeat runs only fetch the newest day.

Usage:
    python nse_4pct_scan.py                    # last 120 calendar days, dashboard.html
    python nse_4pct_scan.py --days 150 --out my_scan.html
    python nse_4pct_scan.py --synthetic        # offline demo with generated data

Run after market close (bhavcopy publishes ~6-7 pm IST).
"""

import argparse
import io
import json
import os
import random
import sys
import zipfile
from datetime import date, timedelta

import pandas as pd
import requests

# ----------------------------- configuration -----------------------------

CACHE_DIR = "nse_cache"

# Core Stockbee rule
MIN_GAIN = 1.04          # c/c1
MIN_VOLUME = 100_000     # shares

# India-specific noise filters
MIN_CLOSE = 30.0         # rupees, skip penny counters
MIN_TURNOVER_CR = 3.0    # rupees crores traded today

# Quality filters
RANGE_POS_MIN = 0.65         # close position within day range for "near high"
BASE_LOOKBACK_MIN = 5        # days
BASE_LOOKBACK_MAX = 40       # days
BASE_MAX_RANGE_PCT = 0.15    # (maxH - minL) / minL over base window
EXT_MAX = 1.15               # prior close / 20d min close, above this = extended
NR_LOOKBACK = 7              # NR7 check on prior day

# EMA crossover strategy (separate section on the dashboard)
EMA_FAST = 20
EMA_SLOW = 100
EMA_MIN_BARS = 110           # trading days needed before the cross counts

SPARK_DAYS = 30              # closes shown in sparkline

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/125.0 Safari/537.36")

# ----------------------------- data download -----------------------------


def _session():
    s = requests.Session()
    s.headers.update({"User-Agent": UA, "Accept": "*/*",
                      "Referer": "https://www.nseindia.com/"})
    return s


def fetch_day(sess, d: date) -> pd.DataFrame | None:
    """Return one day's EQ-series OHLCV for all symbols, or None (holiday)."""
    tag = d.strftime("%Y%m%d")
    cache = os.path.join(CACHE_DIR, f"bhav_{tag}.parquet")
    miss = os.path.join(CACHE_DIR, f"bhav_{tag}.miss")
    if os.path.exists(cache):
        return pd.read_parquet(cache)
    if os.path.exists(miss):
        return None

    df = None
    # Primary: UDiFF bhavcopy zip
    url1 = ("https://nsearchives.nseindia.com/content/cm/"
            f"BhavCopy_NSE_CM_0_0_0_{tag}_F_0000.csv.zip")
    try:
        r = sess.get(url1, timeout=30)
        if r.status_code == 200 and r.content[:2] == b"PK":
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                raw = pd.read_csv(z.open(z.namelist()[0]))
            raw = raw[raw["SctySrs"].astype(str).str.strip() == "EQ"]
            df = pd.DataFrame({
                "symbol": raw["TckrSymb"].astype(str).str.strip(),
                "open": raw["OpnPric"], "high": raw["HghPric"],
                "low": raw["LwPric"], "close": raw["ClsPric"],
                "volume": raw["TtlTradgVol"],
                "turnover": raw["TtlTrfVal"],  # rupees
            })
    except Exception:
        df = None

    # Fallback: sec_bhavdata_full
    if df is None:
        url2 = ("https://nsearchives.nseindia.com/products/content/"
                f"sec_bhavdata_full_{d.strftime('%d%m%Y')}.csv")
        try:
            r = sess.get(url2, timeout=30)
            if r.status_code == 200 and b"SYMBOL" in r.content[:200]:
                raw = pd.read_csv(io.BytesIO(r.content))
                raw.columns = [c.strip() for c in raw.columns]
                raw = raw[raw["SERIES"].astype(str).str.strip() == "EQ"]
                df = pd.DataFrame({
                    "symbol": raw["SYMBOL"].astype(str).str.strip(),
                    "open": pd.to_numeric(raw["OPEN_PRICE"], errors="coerce"),
                    "high": pd.to_numeric(raw["HIGH_PRICE"], errors="coerce"),
                    "low": pd.to_numeric(raw["LOW_PRICE"], errors="coerce"),
                    "close": pd.to_numeric(raw["CLOSE_PRICE"], errors="coerce"),
                    "volume": pd.to_numeric(raw["TTL_TRD_QNTY"], errors="coerce"),
                    "turnover": pd.to_numeric(raw["TURNOVER_LACS"],
                                              errors="coerce") * 100_000,
                })
        except Exception:
            df = None

    if df is None or df.empty:
        open(miss, "w").close()
        return None

    df["date"] = pd.Timestamp(d)
    df = df.dropna(subset=["open", "high", "low", "close", "volume"])
    df.to_parquet(cache)
    return df


def load_panel(days: int) -> pd.DataFrame:
    os.makedirs(CACHE_DIR, exist_ok=True)
    sess = _session()
    # Warm cookies once; archives usually accept a plain UA but this helps.
    try:
        sess.get("https://www.nseindia.com/", timeout=15)
    except Exception:
        pass

    frames = []
    today = date.today()
    fetched = 0
    for i in range(days):
        d = today - timedelta(days=i)
        if d.weekday() >= 5:
            continue
        part = fetch_day(sess, d)
        if part is not None:
            frames.append(part)
            fetched += 1
            if fetched % 10 == 0:
                print(f"  {fetched} trading days loaded (up to {d})")
    if not frames:
        sys.exit("No bhavcopy data retrieved. Check network or run --synthetic.")
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(["symbol", "date"]).reset_index(drop=True)
    print(f"Panel: {panel['symbol'].nunique()} symbols, "
          f"{panel['date'].nunique()} trading days")
    return panel


# ----------------------------- synthetic mode -----------------------------


def synthetic_panel(n_symbols=60, n_days=130, seed=7) -> pd.DataFrame:
    """Offline demo data. A subset of symbols gets a planted breakout today."""
    rng = random.Random(seed)
    rows = []
    start = date.today() - timedelta(days=int(n_days * 1.5))
    dates = []
    d = start
    while len(dates) < n_days:
        if d.weekday() < 5:
            dates.append(pd.Timestamp(d))
        d += timedelta(days=1)

    for k in range(n_symbols):
        sym = f"DEMO{k:03d}"
        price = rng.uniform(40, 900)
        base_vol = rng.randint(150_000, 4_000_000)
        breakout = k < 14  # first 14 symbols break out on the last day
        tight_from = n_days - rng.randint(8, 30)
        for i, dt in enumerate(dates):
            last = i == n_days - 1
            if breakout and i >= tight_from and not last:
                drift = rng.uniform(-0.006, 0.006)      # tight base
            else:
                drift = rng.gauss(0.0005, 0.014)
            if breakout and last:
                drift = rng.uniform(0.045, 0.11)        # the burst
            o = price * (1 + rng.uniform(-0.004, 0.004))
            c = price * (1 + drift)
            hi = max(o, c) * (1 + rng.uniform(0.0, 0.012))
            lo = min(o, c) * (1 - rng.uniform(0.0, 0.012))
            if breakout and last:
                c = hi * (1 - rng.uniform(0.0, 0.006))  # close near high
            v = int(base_vol * rng.uniform(0.6, 1.4))
            if breakout and last:
                v = int(base_vol * rng.uniform(2.2, 4.5))
            rows.append(dict(symbol=sym, date=dt, open=round(o, 2),
                             high=round(hi, 2), low=round(lo, 2),
                             close=round(c, 2), volume=v,
                             turnover=c * v))
            price = c
    return pd.DataFrame(rows)


# ----------------------------- scan logic -----------------------------


def run_scan(panel: pd.DataFrame) -> tuple[list[dict], str]:
    latest = panel["date"].max()
    results = []

    for sym, g in panel.groupby("symbol", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        if len(g) < 12 or g.iloc[-1]["date"] != latest:
            continue
        t = g.iloc[-1]      # today
        p = g.iloc[-2]      # prior day
        c, c1 = float(t["close"]), float(p["close"])
        v, v1 = float(t["volume"]), float(p["volume"])
        if c1 <= 0 or v1 <= 0:
            continue

        # ---- core Stockbee rule ----
        gain = c / c1
        if not (gain >= MIN_GAIN and v > v1 and v >= MIN_VOLUME):
            continue
        # ---- India noise filters ----
        turnover_cr = float(t.get("turnover", c * v)) / 1e7
        if c < MIN_CLOSE or turnover_cr < MIN_TURNOVER_CR:
            continue

        hi, lo = float(t["high"]), float(t["low"])
        rng_ = hi - lo
        range_pos = (c - lo) / rng_ if rng_ > 0 else 1.0

        # filter 1: close near high
        f_near_high = range_pos >= RANGE_POS_MIN

        # filter 2: prior day negative or narrowest range of last NR_LOOKBACK
        prior_neg = float(p["close"]) < float(p["open"])
        look = g.iloc[-(NR_LOOKBACK + 1):-1]
        ranges = (look["high"] - look["low"]).values
        prior_nr = len(ranges) >= NR_LOOKBACK and ranges[-1] == ranges.min()
        f_prior_setup = prior_neg or prior_nr

        # filter 3: orderly base before breakout (best window in 5-40 days)
        pre = g.iloc[:-1]
        f_base, base_len, base_range = False, None, None
        max_look = min(BASE_LOOKBACK_MAX, len(pre))
        for w in range(max_look, BASE_LOOKBACK_MIN - 1, -1):
            win = pre.iloc[-w:]
            span = (win["high"].max() - win["low"].min()) / win["low"].min()
            if span <= BASE_MAX_RANGE_PCT:
                f_base, base_len, base_range = True, w, span
                break

        # filter 4: young trend, prior close not extended over 20d min close
        m20 = pre["close"].iloc[-20:].min() if len(pre) >= 20 else pre["close"].min()
        extension = c1 / m20 if m20 > 0 else 99
        f_young = extension <= EXT_MAX

        score = int(sum([bool(f_near_high), bool(f_prior_setup),
                         bool(f_base), bool(f_young)]))

        spark = [round(float(x), 2) for x in g["close"].iloc[-SPARK_DAYS:]]
        risk = c - lo
        results.append(dict(
            symbol=sym,
            close=round(c, 2),
            pct=round((gain - 1) * 100, 2),
            vol=int(v),
            vol_ratio=round(v / v1, 2),
            turnover_cr=round(turnover_cr, 1),
            range_pos=int(round(range_pos * 100)),
            near_high=bool(f_near_high),
            prior_setup=bool(f_prior_setup),
            prior_neg=bool(prior_neg),
            prior_nr=bool(prior_nr),
            base=bool(f_base),
            base_len=int(base_len) if base_len else None,
            base_range=round(float(base_range) * 100, 1) if base_range else None,
            young=bool(f_young),
            extension=round((extension - 1) * 100, 1),
            score=score,
            # trade plan (entry reference = today's close)
            stop=round(lo, 2),
            risk_pct=round(risk / c * 100, 2) if c else 0,
            be_trigger=round(c * 1.045, 2),
            partial_8=round(c * 1.08, 2),
            target_20=round(c * 1.20, 2),
            spark=spark,
        ))

    results.sort(key=lambda r: (-r["score"], -r["vol_ratio"], -r["turnover_cr"]))
    return results, latest.strftime("%d %b %Y")


def run_ema_cross(panel: pd.DataFrame) -> list[dict]:
    """EMA 20 crossed above EMA 100 today (daily closes). Separate strategy."""
    latest = panel["date"].max()
    out = []
    for sym, g in panel.groupby("symbol", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        if len(g) < EMA_MIN_BARS or g.iloc[-1]["date"] != latest:
            continue
        t = g.iloc[-1]
        c = float(t["close"])
        v = float(t["volume"])
        turnover_cr = float(t.get("turnover", c * v)) / 1e7
        if c < MIN_CLOSE or turnover_cr < MIN_TURNOVER_CR:
            continue

        closes = g["close"].astype(float)
        e_fast = closes.ewm(span=EMA_FAST, adjust=False).mean()
        e_slow = closes.ewm(span=EMA_SLOW, adjust=False).mean()
        # the cross: below or equal yesterday, above today
        if not (e_fast.iloc[-2] <= e_slow.iloc[-2] and
                e_fast.iloc[-1] > e_slow.iloc[-1]):
            continue

        vol20 = g["volume"].iloc[-21:-1].mean()
        ret20 = (c / float(closes.iloc[-21]) - 1) * 100 if len(g) >= 21 else 0
        out.append(dict(
            symbol=sym,
            close=round(c, 2),
            pct=round((c / float(g.iloc[-2]["close"]) - 1) * 100, 2),
            ema_fast=round(float(e_fast.iloc[-1]), 2),
            ema_slow=round(float(e_slow.iloc[-1]), 2),
            gap_pct=round((float(e_fast.iloc[-1]) / float(e_slow.iloc[-1]) - 1)
                          * 100, 2),
            vol_x=round(v / vol20, 2) if vol20 > 0 else 0,
            turnover_cr=round(turnover_cr, 1),
            ret20=round(ret20, 1),
            spark=[round(float(x), 2) for x in closes.iloc[-SPARK_DAYS:]],
        ))
    out.sort(key=lambda r: (-r["vol_x"], -r["turnover_cr"]))
    return out


# ----------------------------- dashboard -----------------------------

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>4% Burst | NSE __SCAN_DATE__</title>
<style>
  :root{
    --ink:#10151d; --panel:#171e29; --panel2:#1d2634; --line:#2a3547;
    --text:#e8e4da; --dim:#8b95a5; --faint:#5a6474;
    --saffron:#f2a93b; --saffron-dim:#a97727;
    --up:#4cc38a; --down:#e5534b; --band:#2f3b50;
  }
  *{margin:0;padding:0;box-sizing:border-box}
  html{scroll-behavior:smooth}
  @media (prefers-reduced-motion: reduce){ html{scroll-behavior:auto} *{transition:none!important} }
  body{background:var(--ink);color:var(--text);
       font-family:"Segoe UI",system-ui,-apple-system,sans-serif;
       font-variant-numeric:tabular-nums; padding-bottom:60px}
  .mono{font-family:"Cascadia Code","JetBrains Mono",Consolas,monospace}

  header{border-bottom:1px solid var(--line); padding:26px 5vw 20px;
         display:flex; flex-wrap:wrap; gap:18px; align-items:baseline;
         justify-content:space-between}
  h1{font-size:26px; letter-spacing:.5px; font-weight:600}
  h1 b{color:var(--saffron); font-weight:700}
  .scan-def{color:var(--dim); font-size:13px}
  .scan-def code{color:var(--saffron); background:var(--panel);
                 padding:2px 7px; border:1px solid var(--line)}
  .meta{color:var(--faint); font-size:12px; text-align:right}
  .meta .n{color:var(--text); font-size:20px; display:block}

  main{padding:0 5vw}
  h2{font-size:13px; text-transform:uppercase; letter-spacing:2.5px;
     color:var(--dim); margin:34px 0 14px; display:flex; align-items:center; gap:12px}
  h2::after{content:""; flex:1; height:1px; background:var(--line)}
  h2 .count{color:var(--saffron)}

  .grid{display:grid; grid-template-columns:repeat(auto-fill,minmax(340px,1fr)); gap:14px}
  .card{background:var(--panel); border:1px solid var(--line); padding:16px 16px 14px}
  .card.aplus{border-color:var(--saffron-dim)}
  .card-top{display:flex; justify-content:space-between; align-items:baseline}
  .sym{font-size:19px; font-weight:700; letter-spacing:.5px}
  .pct{font-size:19px; color:var(--up); font-weight:700}
  .stats{display:flex; gap:16px; margin-top:6px; font-size:12px; color:var(--dim)}
  .stats b{color:var(--text); font-weight:600}
  .badges{display:flex; flex-wrap:wrap; gap:6px; margin-top:10px}
  .badge{font-size:10.5px; letter-spacing:.8px; text-transform:uppercase;
         padding:3px 8px; border:1px solid var(--line); color:var(--faint)}
  .badge.on{border-color:var(--saffron-dim); color:var(--saffron)}
  svg.spark{display:block; width:100%; height:44px; margin-top:12px}

  /* signature: the trade ladder */
  .ladder{margin-top:14px; padding-top:12px; border-top:1px dashed var(--line)}
  .rail{position:relative; height:6px; background:var(--band); margin:26px 4px 30px}
  .rail .fill{position:absolute; left:0; top:0; bottom:0; background:
        linear-gradient(90deg, var(--down), var(--saffron) 30%, var(--up))}
  .tick{position:absolute; top:-7px; width:2px; height:20px; background:var(--text)}
  .tick .lab{position:absolute; white-space:nowrap; font-size:10px;
             letter-spacing:.5px; color:var(--dim); left:50%; transform:translateX(-50%)}
  .tick .lab.abv{bottom:24px} .tick .lab.blw{top:24px}
  .tick .prc{display:block; color:var(--text); font-weight:600; font-size:11px}
  .tick.stop{background:var(--down)} .tick.entry{background:var(--saffron)}
  .tick.t8{background:var(--up)}
  .risk-line{font-size:11.5px; color:var(--faint); margin-top:4px}
  .risk-line b{color:var(--down)}

  table{width:100%; border-collapse:collapse; font-size:13px}
  th{color:var(--faint); font-size:10.5px; text-transform:uppercase;
     letter-spacing:1.2px; text-align:right; padding:8px 10px;
     border-bottom:1px solid var(--line)}
  th:first-child, td:first-child{text-align:left}
  td{padding:8px 10px; text-align:right; border-bottom:1px solid var(--panel2)}
  td.sy{font-weight:700} td.up{color:var(--up)}
  tr:hover td{background:var(--panel2)}
  .dot{color:var(--saffron)} .nodot{color:var(--band)}

  .rules{margin-top:44px; background:var(--panel); border:1px solid var(--line);
         padding:20px 22px}
  .rules h3{font-size:12px; letter-spacing:2px; text-transform:uppercase;
            color:var(--saffron); margin-bottom:12px}
  .rules ol{margin-left:18px; color:var(--dim); font-size:13px; line-height:1.9}
  .rules b{color:var(--text)}
  footer{margin-top:26px; padding:0 5vw; color:var(--faint); font-size:11px;
         line-height:1.7}
  .empty{color:var(--faint); padding:30px 0; font-size:14px}
</style>
</head>
<body>
<header>
  <div>
    <h1>4% Momentum Burst <b>· NSE</b></h1>
    <div class="scan-def mono">c/c1 ≥ 1.04 · v &gt; v1 · v ≥ 100k · EQ · ₹__MIN_TURNOVER__ cr+ traded</div>
  </div>
  <div class="meta">
    <span class="n">__SCAN_DATE__</span>
    <span id="hdr-counts"></span>
  </div>
</header>
<main>
  <h2>A+ setups <span class="count" id="aplus-count"></span></h2>
  <div class="grid" id="aplus-grid"></div>
  <div class="empty" id="aplus-empty" hidden>No candidate passed all four quality filters today. Check the full list below.</div>

  <h2>All core-scan hits <span class="count" id="all-count"></span></h2>
  <table id="all-table">
    <thead><tr>
      <th>Symbol</th><th>Close</th><th>%Chg</th><th>Vol ×</th><th>₹ cr</th>
      <th>Range pos</th><th>Near high</th><th>Prior NR/−</th><th>Base</th><th>Young</th><th>Score</th>
    </tr></thead>
    <tbody></tbody>
  </table>

  <h2>EMA 20 × 100 crossover <span class="count" id="ema-count"></span></h2>
  <div class="scan-def mono" style="margin:-4px 0 12px">
    Daily EMA(close, 20) crossed above Daily EMA(close, 100) today ·
    trend entry, not a burst — manage on the slow EMA, not the burst playbook below
  </div>
  <table id="ema-table">
    <thead><tr>
      <th>Symbol</th><th>Close</th><th>%Chg</th><th>EMA 20</th><th>EMA 100</th>
      <th>Gap</th><th>Vol vs 20d</th><th>₹ cr</th><th>20d ret</th>
    </tr></thead>
    <tbody></tbody>
  </table>
  <div class="empty" id="ema-empty" hidden>No stock crossed EMA 20 above EMA 100 today.</div>

  <div class="rules">
    <h3>Trade management playbook</h3>
    <ol>
      <li><b>Stop:</b> low of the entry day. Exit the same day if the gain fades below entry.</li>
      <li><b>Breakeven:</b> once up 4-5%, move the stop to breakeven.</li>
      <li><b>No follow-through by day 3:</b> exit.</li>
      <li><b>Up 8%+:</b> sell about half, trail the stop under each day's high.</li>
      <li><b>Sell into strength.</b> Time-exit most positions on days 3-5. Target band 8-40%.</li>
    </ol>
  </div>
</main>
<footer>
  Data: NSE end-of-day bhavcopy. Prices are the scan day's close; next-day entries will differ.
  Personal research tool, not investment advice.
</footer>

<script>
const DATA = __DATA_JSON__;
const EMA_DATA = __EMA_JSON__;

function fmt(n){ return n.toLocaleString('en-IN'); }

function spark(vals){
  const w=320,h=44,pad=2;
  const mn=Math.min(...vals), mx=Math.max(...vals), sp=(mx-mn)||1;
  const pts=vals.map((v,i)=>[
    pad+i*(w-2*pad)/(vals.length-1),
    h-pad-(v-mn)/sp*(h-2*pad)
  ]);
  const line=pts.map(p=>p[0].toFixed(1)+','+p[1].toFixed(1)).join(' ');
  const last=pts[pts.length-1];
  return `<svg class="spark" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none">
    <polyline points="${line}" fill="none" stroke="#5a6474" stroke-width="1.3"/>
    <circle cx="${last[0].toFixed(1)}" cy="${last[1].toFixed(1)}" r="3" fill="#f2a93b"/>
  </svg>`;
}

function ladder(r){
  // rail spans stop .. +20% target
  const lo=r.stop, hi=r.target_20, span=hi-lo;
  const x=v=>((v-lo)/span*100).toFixed(1);
  const tick=(v,cls,lab,pos)=>`
    <div class="tick ${cls}" style="left:${x(v)}%">
      <span class="lab ${pos}">${lab}<span class="prc">${v.toFixed(2)}</span></span>
    </div>`;
  return `<div class="ladder">
    <div class="rail">
      <div class="fill" style="width:${x(r.close)}%"></div>
      ${tick(r.stop,'stop','STOP · day low','blw')}
      ${tick(r.close,'entry','ENTRY REF','abv')}
      ${tick(r.be_trigger,'','BE @+4.5%','blw')}
      ${tick(r.partial_8,'t8','½ OFF @+8%','abv')}
    </div>
    <div class="risk-line">Risk to stop <b>${r.risk_pct.toFixed(1)}%</b> ·
      day 3 no follow-through → out · time-exit day 3-5</div>
  </div>`;
}

function badge(on, txt){ return `<span class="badge ${on?'on':''}">${txt}</span>`; }

function card(r){
  return `<div class="card ${r.score===4?'aplus':''}">
    <div class="card-top"><span class="sym">${r.symbol}</span>
      <span class="pct">+${r.pct.toFixed(1)}%</span></div>
    <div class="stats">
      <span>Close <b>${r.close.toFixed(2)}</b></span>
      <span>Vol <b>${r.vol_ratio.toFixed(1)}×</b></span>
      <span><b>₹${r.turnover_cr}</b> cr</span>
      <span>Range <b>${r.range_pos}%</b></span>
    </div>
    <div class="badges">
      ${badge(r.near_high,'near high')}
      ${badge(r.prior_setup, r.prior_nr?'prior NR7':'prior −ve')}
      ${badge(r.base, r.base?'base '+r.base_len+'d / '+r.base_range+'%':'no base')}
      ${badge(r.young,'young · +'+r.extension+'% ext')}
    </div>
    ${spark(r.spark)}
    ${ladder(r)}
  </div>`;
}

const aplus = DATA.filter(r=>r.score>=4);
const shown = aplus.length? aplus : DATA.filter(r=>r.score===3);
document.getElementById('aplus-grid').innerHTML = shown.map(card).join('');
document.getElementById('aplus-count').textContent = shown.length;
if(!shown.length) document.getElementById('aplus-empty').hidden=false;
if(!aplus.length && shown.length)
  document.querySelector('h2 .count').textContent = shown.length+' (best score 3/4)';

const tb=document.querySelector('#all-table tbody');
const dot=b=>`<span class="${b?'dot':'nodot'}">●</span>`;
tb.innerHTML = DATA.map(r=>`<tr>
  <td class="sy">${r.symbol}</td><td>${r.close.toFixed(2)}</td>
  <td class="up">+${r.pct.toFixed(1)}%</td><td>${r.vol_ratio.toFixed(1)}×</td>
  <td>${r.turnover_cr}</td><td>${r.range_pos}%</td>
  <td>${dot(r.near_high)}</td><td>${dot(r.prior_setup)}</td>
  <td>${dot(r.base)}</td><td>${dot(r.young)}</td>
  <td><b>${r.score}</b>/4</td></tr>`).join('');
document.getElementById('all-count').textContent = DATA.length;
document.getElementById('hdr-counts').textContent =
  DATA.length+' hits · '+aplus.length+' A+ · '+EMA_DATA.length+' EMA cross';

const etb=document.querySelector('#ema-table tbody');
etb.innerHTML = EMA_DATA.map(r=>`<tr>
  <td class="sy">${r.symbol}</td><td>${r.close.toFixed(2)}</td>
  <td class="${r.pct>=0?'up':''}">${r.pct>=0?'+':''}${r.pct.toFixed(1)}%</td>
  <td>${r.ema_fast.toFixed(2)}</td><td>${r.ema_slow.toFixed(2)}</td>
  <td>+${r.gap_pct.toFixed(2)}%</td><td>${r.vol_x.toFixed(1)}×</td>
  <td>${r.turnover_cr}</td><td>${r.ret20>=0?'+':''}${r.ret20.toFixed(1)}%</td>
</tr>`).join('');
document.getElementById('ema-count').textContent = EMA_DATA.length;
if(!EMA_DATA.length){
  document.getElementById('ema-table').hidden=true;
  document.getElementById('ema-empty').hidden=false;
}
</script>
</body>
</html>
"""


def write_dashboard(results, ema_results, scan_date, out_path):
    html = (HTML_TEMPLATE
            .replace("__SCAN_DATE__", scan_date)
            .replace("__MIN_TURNOVER__", str(int(MIN_TURNOVER_CR)))
            .replace("__DATA_JSON__", json.dumps(results))
            .replace("__EMA_JSON__", json.dumps(ema_results)))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Dashboard written: {out_path} "
          f"({len(results)} hits, {sum(r['score'] == 4 for r in results)} A+, "
          f"{len(ema_results)} EMA crosses)")


# ----------------------------- main -----------------------------


def main():
    ap = argparse.ArgumentParser(description="NSE 4% momentum burst scanner")
    ap.add_argument("--days", type=int, default=250,
                    help="calendar days of history to load (default 250; "
                         "the EMA 100 needs the longer window)")
    ap.add_argument("--out", default="dashboard.html")
    ap.add_argument("--synthetic", action="store_true",
                    help="offline demo with generated data")
    args = ap.parse_args()

    if args.synthetic:
        print("Synthetic mode: generating demo panel")
        panel = synthetic_panel()
    else:
        panel = load_panel(args.days)

    results, scan_date = run_scan(panel)
    ema_results = run_ema_cross(panel)
    write_dashboard(results, ema_results, scan_date, args.out)


if __name__ == "__main__":
    main()
