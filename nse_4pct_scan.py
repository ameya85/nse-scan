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


def build_hist(panel: pd.DataFrame, max_symbols=1200, bars=160) -> dict:
    """Aligned close history for liquid symbols, embedded in the page so the
    EMA scan runs in the browser with editable periods and timeframe."""
    latest = panel["date"].max()
    px = (panel.pivot_table(index="date", columns="symbol",
                            values="close", aggfunc="last")
          .sort_index().ffill().tail(bars))
    today = panel[panel["date"] == latest].set_index("symbol")
    keep = []
    for sym in px.columns:
        if sym not in today.index:
            continue
        t = today.loc[sym]
        c = float(t["close"])
        turn_cr = float(t.get("turnover", c * float(t["volume"]))) / 1e7
        if c < MIN_CLOSE or turn_cr < MIN_TURNOVER_CR:
            continue
        col = px[sym]
        if col.isna().any() or len(col) < EMA_MIN_BARS:
            continue
        keep.append((sym, turn_cr))
    keep.sort(key=lambda x: -x[1])
    keep = keep[:max_symbols]
    return {
        "dates": [d.strftime("%Y-%m-%d") for d in px.index],
        "symbols": {s: {"c": [round(float(x), 2) for x in px[s]],
                        "t": round(tc, 1)} for s, tc in keep},
    }


# ----------------------------- dashboard -----------------------------

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NSE Scans | __SCAN_DATE__</title>
<style>
  :root{
    --bg:#f6f7f9; --card:#ffffff; --line:#e5e8ee; --soft:#f0f2f6;
    --ink:#1c2433; --dim:#5b6575; --faint:#9aa3b2;
    --accent:#b26a00; --accent-fill:#fdf3e3;
    --up:#0c8a5a; --up-fill:#e7f6ef; --down:#cc4437; --down-fill:#fbeceb;
  }
  *{margin:0;padding:0;box-sizing:border-box}
  body{background:var(--bg); color:var(--ink);
       font-family:"Segoe UI",system-ui,-apple-system,Roboto,sans-serif;
       font-variant-numeric:tabular-nums; padding-bottom:60px}
  .mono{font-family:"Cascadia Code",Consolas,Menlo,monospace}

  header{padding:22px 4vw 0; max-width:1280px; margin:0 auto}
  .head-row{display:flex; flex-wrap:wrap; gap:10px 18px; align-items:baseline;
            justify-content:space-between}
  h1{font-size:21px; font-weight:700; letter-spacing:.2px}
  h1 span{color:var(--accent)}
  .date{color:var(--dim); font-size:13px; font-weight:600}

  .tabs{display:flex; gap:6px; margin:16px 0 0; background:var(--soft);
        border:1px solid var(--line); border-radius:999px; padding:4px;
        width:max-content; max-width:100%}
  .tab{border:0; background:transparent; color:var(--dim); font:inherit;
       font-size:13.5px; font-weight:600; padding:8px 18px; border-radius:999px;
       cursor:pointer; white-space:nowrap}
  .tab.active{background:var(--card); color:var(--ink);
              box-shadow:0 1px 3px rgba(20,28,45,.12)}
  .tab .n{color:var(--accent); margin-left:6px}

  main{padding:18px 4vw 0; max-width:1280px; margin:0 auto}
  .panel{display:none} .panel.active{display:block}
  .subline{color:var(--dim); font-size:12.5px; margin:2px 0 16px}
  .subline code{background:var(--soft); border:1px solid var(--line);
                border-radius:6px; padding:2px 7px; color:var(--ink)}

  h2{font-size:12px; text-transform:uppercase; letter-spacing:2px;
     color:var(--faint); margin:26px 0 12px}
  h2 .count{color:var(--accent)}

  .grid{display:grid; grid-template-columns:repeat(auto-fill,minmax(330px,1fr)); gap:14px}
  .card{background:var(--card); border:1px solid var(--line); border-radius:14px;
        padding:16px; box-shadow:0 1px 2px rgba(20,28,45,.05)}
  .card-top{display:flex; justify-content:space-between; align-items:baseline; gap:8px}
  .sym{font-size:17px; font-weight:700; letter-spacing:.3px; overflow-wrap:anywhere}
  .pct{font-size:16px; color:var(--up); font-weight:700; white-space:nowrap}
  .stats{display:flex; flex-wrap:wrap; gap:4px 14px; margin-top:6px;
         font-size:12px; color:var(--dim)}
  .stats b{color:var(--ink); font-weight:600}
  .badges{display:flex; flex-wrap:wrap; gap:6px; margin-top:10px}
  .badge{font-size:10.5px; letter-spacing:.6px; text-transform:uppercase;
         padding:4px 9px; border-radius:999px; background:var(--soft);
         color:var(--faint); font-weight:600}
  .badge.on{background:var(--accent-fill); color:var(--accent)}
  svg.spark{display:block; width:100%; height:40px; margin-top:12px}

  .plan{margin-top:12px; border-top:1px solid var(--line); padding-top:12px;
        display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:8px}
  .plan .cell{background:var(--soft); border-radius:10px; padding:8px 6px;
              text-align:center; min-width:0}
  .plan .lab{font-size:9.5px; letter-spacing:.6px; text-transform:uppercase;
             color:var(--faint); font-weight:700; margin-bottom:3px}
  .plan .val{font-size:13px; font-weight:700; overflow-wrap:anywhere}
  .plan .cell.stop{background:var(--down-fill)} .plan .cell.stop .val{color:var(--down)}
  .plan .cell.sell{background:var(--up-fill)} .plan .cell.sell .val{color:var(--up)}
  .plan-note{grid-column:1/-1; font-size:11.5px; color:var(--faint); text-align:left;
             padding-top:2px}
  @media (max-width:420px){ .plan{grid-template-columns:repeat(2,minmax(0,1fr))} }

  .tablewrap{overflow-x:auto; background:var(--card); border:1px solid var(--line);
             border-radius:14px; box-shadow:0 1px 2px rgba(20,28,45,.05)}
  table{width:100%; border-collapse:collapse; font-size:13px; min-width:640px}
  th{color:var(--faint); font-size:10.5px; text-transform:uppercase;
     letter-spacing:1px; text-align:right; padding:10px 12px;
     border-bottom:1px solid var(--line); background:var(--soft)}
  th:first-child, td:first-child{text-align:left}
  td{padding:9px 12px; text-align:right; border-bottom:1px solid var(--soft)}
  tr:last-child td{border-bottom:0}
  td.sy{font-weight:700} td.up{color:var(--up); font-weight:600}
  td.dn{color:var(--down); font-weight:600}
  .dot{color:var(--up)} .nodot{color:var(--line)}

  .controls{display:flex; flex-wrap:wrap; gap:10px 16px; align-items:flex-end;
            background:var(--card); border:1px solid var(--line); border-radius:14px;
            padding:14px 16px; margin-bottom:18px;
            box-shadow:0 1px 2px rgba(20,28,45,.05)}
  .ctl{display:flex; flex-direction:column; gap:4px}
  .ctl label{font-size:10px; letter-spacing:.6px; text-transform:uppercase;
             color:var(--faint); font-weight:700}
  .ctl input,.ctl select{font:inherit; font-size:13.5px; color:var(--ink);
        background:var(--soft); border:1px solid var(--line); border-radius:8px;
        padding:7px 10px; width:86px}
  .ctl select{width:110px}
  .btn{font:inherit; font-size:13px; font-weight:600; border:0; cursor:pointer;
       border-radius:8px; padding:8px 18px; background:var(--accent); color:#fff}
  .ctl-note{font-size:11.5px; color:var(--faint); flex-basis:100%}

  .score{display:inline-block; min-width:40px; text-align:center; font-weight:700;
         border-radius:999px; padding:3px 10px; font-size:12px}
  .score.hi{background:var(--up-fill); color:var(--up)}
  .score.md{background:var(--accent-fill); color:var(--accent)}
  .score.lo{background:var(--soft); color:var(--faint)}
  .lnk{background:none; border:0; color:var(--accent); font:inherit; font-size:11.5px;
       cursor:pointer; text-decoration:underline; padding:0}

  .rules{margin-top:26px; background:var(--card); border:1px solid var(--line);
         border-radius:14px; padding:18px 20px; box-shadow:0 1px 2px rgba(20,28,45,.05)}
  .rules h3{font-size:11.5px; letter-spacing:1.6px; text-transform:uppercase;
            color:var(--accent); margin-bottom:10px}
  .rules ol{margin-left:18px; color:var(--dim); font-size:13px; line-height:1.85}
  .rules b{color:var(--ink)}
  footer{margin-top:24px; padding:0 4vw; max-width:1280px; margin-inline:auto;
         color:var(--faint); font-size:11px; line-height:1.7}
  .empty{color:var(--faint); padding:26px 0; font-size:14px}
</style>
</head>
<body>
<header>
  <div class="head-row">
    <h1>NSE Scans <span>· __SCAN_DATE__</span></h1>
    <span class="date" id="hdr-counts"></span>
  </div>
  <div class="tabs" role="tablist">
    <button class="tab active" data-panel="burst" role="tab">4% Burst<span class="n" id="tab-burst-n"></span></button>
    <button class="tab" data-panel="ema" role="tab">EMA 20/100 Cross<span class="n" id="tab-ema-n"></span></button>
    <button class="tab" data-panel="grad" role="tab">4h → Daily<span class="n" id="tab-grad-n"></span></button>
  </div>
</header>
<main>

<section class="panel active" id="panel-burst">
  <p class="subline">Core rule <code class="mono">c/c1 ≥ 1.04 · v &gt; v1 · v ≥ 100k</code>
     · EQ series · ₹__MIN_TURNOVER__ cr+ traded · scored on 4 quality filters</p>

  <h2>A+ setups <span class="count" id="aplus-count"></span></h2>
  <div class="grid" id="aplus-grid"></div>
  <div class="empty" id="aplus-empty" hidden>No candidate passed all four quality filters today. The full list is below.</div>

  <h2>All core-scan hits <span class="count" id="all-count"></span></h2>
  <div class="tablewrap"><table id="all-table">
    <thead><tr>
      <th>Symbol</th><th>Close</th><th>%Chg</th><th>Vol ×</th><th>₹ cr</th>
      <th>Range pos</th><th>Near high</th><th>Prior NR/−</th><th>Base</th><th>Young</th><th>Score</th>
    </tr></thead>
    <tbody></tbody>
  </table></div>

  <div class="rules">
    <h3>Burst playbook</h3>
    <ol>
      <li><b>Stop:</b> low of the entry day. Exit the same day if the gain fades below entry.</li>
      <li><b>Breakeven:</b> once up 4-5%, move the stop to breakeven.</li>
      <li><b>No follow-through by day 3:</b> exit.</li>
      <li><b>Up 8%+:</b> sell about half, trail the stop under each day's high.</li>
      <li><b>Sell into strength.</b> Time-exit most positions on days 3-5. Target band 8-40%.</li>
    </ol>
  </div>
</section>

<section class="panel" id="panel-ema">
  <p class="subline"><code class="mono">EMA(close, fast)</code> crossed above
     <code class="mono">EMA(close, slow)</code> on the latest bar · same price and turnover floors</p>

  <div class="controls">
    <div class="ctl"><label for="ema-fast">Fast EMA</label>
      <input id="ema-fast" type="number" min="2" max="200" value="20"></div>
    <div class="ctl"><label for="ema-slow">Slow EMA</label>
      <input id="ema-slow" type="number" min="3" max="300" value="100"></div>
    <div class="ctl"><label for="ema-tf">Timeframe</label>
      <select id="ema-tf">
        <option value="D" selected>Daily</option>
        <option value="W">Weekly</option>
      </select></div>
    <button class="btn" id="ema-apply">Apply</button>
    <div class="ctl-note" id="ema-note"></div>
  </div>

  <h2>Fresh crossovers <span class="count" id="ema-count"></span></h2>
  <div class="tablewrap"><table id="ema-table">
    <thead><tr>
      <th>Symbol</th><th>Close</th><th>%Chg (bar)</th><th id="th-fast">EMA 20</th>
      <th id="th-slow">EMA 100</th><th>Gap</th><th>₹ cr today</th><th>20-bar ret</th>
    </tr></thead>
    <tbody></tbody>
  </table></div>
  <div class="empty" id="ema-empty" hidden></div>

  <div class="rules">
    <h3>Crossover notes</h3>
    <ol>
      <li>This is a trend entry held for weeks. The burst playbook does not apply here.</li>
      <li>Common management: stop under the most recent swing low or the slow EMA, review weekly.</li>
      <li>Crossovers in a stock already up a lot are late; the 20-bar ret column helps you judge.</li>
      <li>Data is end of day, so the timeframes are Daily and Weekly. A 4-hour view needs
          intraday data, which the free NSE archive does not publish.</li>
    </ol>
  </div>
</section>

<section class="panel" id="panel-grad">
  <p class="subline">Upload your Chartink export of 4h EMA 20/100 crossovers.
     The page scores each stock on how close it is to the same cross on the daily chart.</p>

  <div class="controls">
    <div class="ctl"><label for="grad-file">Chartink CSV</label>
      <input id="grad-file" type="file" accept=".csv,.txt" style="width:auto"></div>
    <div class="ctl" style="flex:1; min-width:200px"><label for="grad-paste">Or paste symbols</label>
      <input id="grad-paste" type="text" style="width:100%"
             placeholder="RELIANCE, TCS, OMAXE"></div>
    <button class="btn" id="grad-apply">Score</button>
    <div class="ctl-note" id="grad-note"></div>
  </div>

  <h2>Daily graduation score <span class="count" id="grad-count"></span></h2>
  <div class="tablewrap"><table id="grad-table">
    <thead><tr>
      <th>Symbol</th><th>Close</th><th>Daily gap</th><th>Closing speed</th>
      <th>Est. days</th><th>Above EMA 100</th><th>20d ret</th><th>Score</th><th>Status</th>
    </tr></thead>
    <tbody></tbody>
  </table></div>
  <div class="empty" id="grad-empty">Upload a CSV or paste symbols, then tap Score.</div>

  <div class="rules">
    <h3>How the score works</h3>
    <ol>
      <li><b>Daily gap</b> is EMA 20 vs EMA 100 on the daily chart. Negative means EMA 20 is
          still below; the smaller the gap, the more points (up to 45).</li>
      <li><b>Closing speed</b> is how fast that gap has narrowed over the last 5 sessions.
          A pace that crosses within ~3 days scores the full 35 points, fading to 0 at 15 days.
          A widening gap scores 0 here.</li>
      <li><b>Price above the daily EMA 100</b> adds 10 points, above the EMA 20 another 10.</li>
      <li>A stock that has already crossed on daily scores <b>100 · Confirmed</b>.
          Score 70+ is a strong candidate, below 40 the 4h signal is early.</li>
      <li>Scoring always uses daily EMA 20/100 to match your Chartink strategy, whatever the
          EMA tab is set to. Stocks marked <b>no data</b> are outside the page's coverage
          (below the ₹__MIN_TURNOVER__ cr turnover floor, or a non-EQ series).</li>
    </ol>
  </div>
</section>

</main>
<footer>
  Data: NSE end-of-day bhavcopy. Prices are the scan day's close; next-day entries will differ.
  Personal research tool, not investment advice.
</footer>

<script>
const DATA = __DATA_JSON__;
const HIST = __HIST_JSON__;

document.querySelectorAll('.tab').forEach(b=>{
  b.addEventListener('click',()=>{
    document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));
    document.querySelectorAll('.panel').forEach(x=>x.classList.remove('active'));
    b.classList.add('active');
    document.getElementById('panel-'+b.dataset.panel).classList.add('active');
  });
});

function spark(vals){
  const w=320,h=40,pad=2;
  const mn=Math.min(...vals), mx=Math.max(...vals), sp=(mx-mn)||1;
  const pts=vals.map((v,i)=>[
    pad+i*(w-2*pad)/(vals.length-1),
    h-pad-(v-mn)/sp*(h-2*pad)
  ]);
  const line=pts.map(p=>p[0].toFixed(1)+','+p[1].toFixed(1)).join(' ');
  const last=pts[pts.length-1];
  return `<svg class="spark" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none">
    <polyline points="${line}" fill="none" stroke="#c4cbd6" stroke-width="1.4"/>
    <circle cx="${last[0].toFixed(1)}" cy="${last[1].toFixed(1)}" r="3" fill="#b26a00"/>
  </svg>`;
}

function plan(r){
  return `<div class="plan">
    <div class="cell stop"><div class="lab">Stop · day low</div>
      <div class="val">${r.stop.toFixed(2)}</div></div>
    <div class="cell"><div class="lab">BE at +4.5%</div>
      <div class="val">${r.be_trigger.toFixed(2)}</div></div>
    <div class="cell sell"><div class="lab">½ off at +8%</div>
      <div class="val">${r.partial_8.toFixed(2)}</div></div>
    <div class="cell"><div class="lab">Risk to stop</div>
      <div class="val">${r.risk_pct.toFixed(1)}%</div></div>
    <div class="plan-note">Entry ref ${r.close.toFixed(2)} (scan-day close) ·
      no new high by day 3 → exit · time-exit day 3-5</div>
  </div>`;
}

function badge(on, txt){ return `<span class="badge ${on?'on':''}">${txt}</span>`; }

function card(r){
  return `<div class="card">
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
    ${plan(r)}
  </div>`;
}

const aplus = DATA.filter(r=>r.score>=4);
const shown = aplus.length? aplus : DATA.filter(r=>r.score===3);
document.getElementById('aplus-grid').innerHTML = shown.map(card).join('');
document.getElementById('aplus-count').textContent =
  aplus.length? shown.length : (shown.length? shown.length+' (best score 3/4)':'');
if(!shown.length) document.getElementById('aplus-empty').hidden=false;

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

// ---- interactive EMA scan (computed in the browser from embedded history) ----

// last bar index of each ISO week, for the weekly timeframe
const weekEnds = (()=>{
  const idx=[]; let prev=null;
  HIST.dates.forEach((d,i)=>{
    const dt=new Date(d+'T00:00:00Z');
    const day=(dt.getUTCDay()+6)%7;                       // Mon=0
    const monday=new Date(dt); monday.setUTCDate(dt.getUTCDate()-day);
    const key=monday.toISOString().slice(0,10);
    if(key!==prev){ idx.push(i); prev=key; } else { idx[idx.length-1]=i; }
  });
  return idx;
})();

function emaArr(c, span){
  const k=2/(span+1); let e=c[0]; const out=[e];
  for(let i=1;i<c.length;i++){ e=c[i]*k+e*(1-k); out.push(e); }
  return out;
}

function runEma(){
  const fast=parseInt(document.getElementById('ema-fast').value,10);
  const slow=parseInt(document.getElementById('ema-slow').value,10);
  const tf=document.getElementById('ema-tf').value;
  const note=document.getElementById('ema-note');
  const empty=document.getElementById('ema-empty');
  const wrap=document.querySelector('#panel-ema .tablewrap');
  note.textContent='';
  if(!(fast>=2 && slow>fast)){
    note.textContent='Fast must be at least 2 and smaller than slow.'; return;
  }
  document.getElementById('th-fast').textContent='EMA '+fast;
  document.getElementById('th-slow').textContent='EMA '+slow;

  const need=slow+5, insuff=[], rows=[];
  for(const [sym,s] of Object.entries(HIST.symbols)){
    const c = tf==='D' ? s.c : weekEnds.map(i=>s.c[i]);
    const n=c.length;
    if(n<need){ insuff.push(sym); continue; }
    const ef=emaArr(c,fast), es=emaArr(c,slow);
    if(!(ef[n-2]<=es[n-2] && ef[n-1]>es[n-1])) continue;
    rows.push({sym, close:c[n-1], pct:(c[n-1]/c[n-2]-1)*100,
      ef:ef[n-1], es:es[n-1], gap:(ef[n-1]/es[n-1]-1)*100, t:s.t,
      ret:n>=21?(c[n-1]/c[n-21]-1)*100:0});
  }
  rows.sort((a,b)=>b.t-a.t);

  const etb=document.querySelector('#ema-table tbody');
  etb.innerHTML = rows.map(r=>`<tr>
    <td class="sy">${r.sym}</td><td>${r.close.toFixed(2)}</td>
    <td class="${r.pct>=0?'up':'dn'}">${r.pct>=0?'+':''}${r.pct.toFixed(1)}%</td>
    <td>${r.ef.toFixed(2)}</td><td>${r.es.toFixed(2)}</td>
    <td>+${r.gap.toFixed(2)}%</td><td>${r.t}</td>
    <td class="${r.ret>=0?'up':'dn'}">${r.ret>=0?'+':''}${r.ret.toFixed(1)}%</td>
  </tr>`).join('');
  document.getElementById('ema-count').textContent=rows.length;
  document.getElementById('tab-ema-n').textContent=rows.length;

  wrap.hidden=!rows.length; empty.hidden=!!rows.length;
  const total=Object.keys(HIST.symbols).length;
  if(!rows.length){
    empty.textContent = insuff.length===total
      ? 'Not enough history for EMA '+slow+' on this timeframe. About '
        +(tf==='D'?HIST.dates.length+' daily':weekEnds.length+' weekly')
        +' bars are available; pick a shorter slow EMA.'
      : 'No stock crossed EMA '+fast+' above EMA '+slow+' on the latest '
        +(tf==='D'?'daily':'weekly')+' bar.';
  }
  if(rows.length && insuff.length)
    note.textContent=insuff.length+' of '+total+' stocks skipped: not enough bars for EMA '+slow+'.';

  document.getElementById('hdr-counts').textContent =
    DATA.length+' burst hits · '+aplus.length+' A+ · '+rows.length+' EMA crosses';
}

document.getElementById('ema-apply').addEventListener('click',runEma);
['ema-fast','ema-slow'].forEach(id=>
  document.getElementById(id).addEventListener('keydown',e=>{
    if(e.key==='Enter') runEma();
  }));
document.getElementById('ema-tf').addEventListener('change',runEma);

// ---- 4h → Daily graduation scoring ----

const GF=20, GS=100;   // fixed daily EMAs, matching the Chartink strategy

const JUNK=/^(SR\.?|NSE|BSE|SYMBOL|LINKS?|PRICE|VOLUME|STOCK|NAME|CHG|DATE|SERIES|OPEN|HIGH|LOW|CLOSE)$/;

function parseSymbols(text){
  const lines=text.split(/\r?\n/).filter(l=>l.trim());
  // A CSV with a Symbol column (Chartink export): take that column only.
  const hi=lines.findIndex(l=>/(^|,)\s*"?symbol"?\s*(,|$)/i.test(l));
  if(hi>=0){
    const cols=lines[hi].split(',').map(c=>c.trim().toLowerCase().replace(/"/g,''));
    const si=cols.indexOf('symbol');
    const out=[], seen=new Set();
    lines.slice(hi+1).forEach(l=>{
      const s=(l.split(',')[si]||'').trim().toUpperCase().replace(/"/g,'');
      if(/^[A-Z0-9&.\-]{1,20}$/.test(s) && /[A-Z]/.test(s) && !seen.has(s)){
        seen.add(s); out.push(s);
      }
    });
    if(out.length) return out;
  }
  // Otherwise treat it as a pasted list of symbols.
  const seen=new Set(), out=[];
  text.split(/[\n,;\t ]+/).forEach(tok=>{
    const s=tok.trim().toUpperCase().replace(/^"|"$/g,'');
    if(/^[A-Z0-9&.\-]{2,20}$/.test(s) && /[A-Z]/.test(s) && !JUNK.test(s)
       && !seen.has(s)){ seen.add(s); out.push(s); }
  });
  return out;
}

function gradeSymbol(sym){
  const s=HIST.symbols[sym];
  if(!s) return {sym, nodata:true};
  const c=s.c, n=c.length;
  const ef=emaArr(c,GF), es=emaArr(c,GS);
  const gap=(ef[n-1]/es[n-1]-1)*100;
  const gap5=(ef[n-6]/es[n-6]-1)*100;
  const rate=(gap-gap5)/5;                       // % per day
  const close=c[n-1];
  const ret20=n>=21?(c[n-1]/c[n-21]-1)*100:0;
  const above100=close>es[n-1], above20=close>ef[n-1];
  let score, status, est=null;
  if(gap>0){ score=100; status='Confirmed'; est=0; }
  else{
    const closeness=Math.max(0, Math.min(1, 1 - (-gap)/5));      // 0 at -5% gap
    let conv=0;
    if(rate>1e-6){ est=-gap/rate; conv=Math.max(0, Math.min(1, (15-est)/12)); }
    score=Math.round(45*closeness + 35*conv + (above100?10:0) + (above20?10:0));
    status= score>=70?'Strong' : score>=40?'Watch' : 'Early';
  }
  return {sym, close, gap, rate, est, above100, ret20, score, status};
}

function renderGrad(syms){
  const rows=syms.map(gradeSymbol);
  const have=rows.filter(r=>!r.nodata).sort((a,b)=>b.score-a.score);
  const miss=rows.filter(r=>r.nodata).map(r=>r.sym);
  const gtb=document.querySelector('#grad-table tbody');
  const cls=s=>s>=70?'hi':s>=40?'md':'lo';
  gtb.innerHTML = have.map(r=>`<tr>
    <td class="sy">${r.sym}</td><td>${r.close.toFixed(2)}</td>
    <td class="${r.gap>0?'up':'dn'}">${r.gap>0?'+':''}${r.gap.toFixed(2)}%</td>
    <td class="${r.rate>0?'up':'dn'}">${r.rate>0?'+':''}${r.rate.toFixed(3)}%/d</td>
    <td>${r.est===0?'crossed':(r.est?'~'+Math.ceil(r.est)+'d':'—')}</td>
    <td><span class="${r.above100?'dot':'nodot'}">●</span></td>
    <td class="${r.ret20>=0?'up':'dn'}">${r.ret20>=0?'+':''}${r.ret20.toFixed(1)}%</td>
    <td><span class="score ${cls(r.score)}">${r.score}</span></td>
    <td>${r.status}</td></tr>`).join('')
    + miss.map(s=>`<tr><td class="sy">${s}</td>
      <td colspan="8" style="text-align:left;color:var(--faint)">no data on this page</td></tr>`).join('');
  document.getElementById('grad-count').textContent=have.length;
  document.getElementById('tab-grad-n').textContent=
    have.filter(r=>r.score>=70).length+'/'+have.length;
  document.querySelector('#panel-grad .tablewrap').hidden=!rows.length;
  document.getElementById('grad-empty').hidden=!!rows.length;
  const note=document.getElementById('grad-note');
  note.innerHTML='';
  if(rows.length){
    const parts=[];
    if(miss.length) parts.push(miss.length+' symbol'+(miss.length>1?'s':'')+' without data');
    parts.push('list saved on this device, re-scored on every page update');
    note.innerHTML=parts.join(' · ')+' · <button class="lnk" id="grad-clear">clear saved list</button>';
    const cb=document.getElementById('grad-clear');
    if(cb) cb.addEventListener('click',()=>{
      try{localStorage.removeItem('gradSyms');}catch(e){}
      gtb.innerHTML=''; note.textContent='';
      document.querySelector('#panel-grad .tablewrap').hidden=true;
      document.getElementById('grad-empty').hidden=false;
      document.getElementById('grad-count').textContent='';
      document.getElementById('tab-grad-n').textContent='';
    });
  }
}

function gradApply(){
  const f=document.getElementById('grad-file').files[0];
  const paste=document.getElementById('grad-paste').value;
  const go=text=>{
    const syms=parseSymbols(text);
    if(!syms.length){
      document.getElementById('grad-note').textContent='No symbols found in that input.';
      return;
    }
    try{localStorage.setItem('gradSyms', syms.join(','));}catch(e){}
    renderGrad(syms);
  };
  if(f){ const rd=new FileReader(); rd.onload=()=>go(rd.result); rd.readAsText(f); }
  else if(paste.trim()){ go(paste); }
  else document.getElementById('grad-note').textContent='Choose a CSV or paste symbols first.';
}
document.getElementById('grad-apply').addEventListener('click',gradApply);
document.getElementById('grad-file').addEventListener('change',gradApply);

try{
  const saved=localStorage.getItem('gradSyms');
  if(saved) renderGrad(saved.split(','));
}catch(e){}

document.getElementById('tab-burst-n').textContent = DATA.length;
runEma();
</script>
</body>
</html>
"""


def write_dashboard(results, hist, scan_date, out_path):
    html = (HTML_TEMPLATE
            .replace("__SCAN_DATE__", scan_date)
            .replace("__MIN_TURNOVER__", str(int(MIN_TURNOVER_CR)))
            .replace("__DATA_JSON__", json.dumps(results))
            .replace("__HIST_JSON__", json.dumps(hist)))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Dashboard written: {out_path} "
          f"({len(results)} hits, {sum(r['score'] == 4 for r in results)} A+, "
          f"{len(hist['symbols'])} symbols embedded for the EMA tab)")


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
    hist = build_hist(panel)
    write_dashboard(results, hist, scan_date, args.out)


if __name__ == "__main__":
    main()
