from __future__ import annotations

import asyncio
import math
import time
from pathlib import Path
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import norm
import argparse
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from data_ import (
    discover_names, DeribitStream,
    EXPIRY, DEPTH, T1_SEC, T2_SEC,
)

# to Run this script, you need to give the following arguments:
# 1. expiry code e.g. 23MAY25
# 2. total runtime seconds (T1)
# 3. interval between snapshots (T2)
# 4. additional strike prices to include (optional)
# Example usage:
# python calculating_own_mark_price.py 23MAY25 3600 5 10000 20000

pd.options.display.float_format = "{:.4f}".format


# Constants
SPREAD_IV_MAX = 1.5    # exclude quotes with ask‑IV – bid‑IV > 1.5 vols (~90th‑pctile weekday spread)
SIZE_MIN      = 5      # require ≥ 5 contracts on both sides—covers 98 % of ATM/weeklies but still filters 1‑lot feelers
POLY_DEGREE   = 2      # quadratic in ln(K/S): captures skew + basic smile with minimum over‑fit
LIQ_K         = 10   # λ = depth / (depth + 50·spread) ⇒ 1‑tick/50‑lot book gets 50 % weight on micro‑price
PLOT_EVERY    = 10     # generate smile plot every ~50 s when t2=5 s, keeps disk usage modest

MAX_MONEYNESS = 2.5    # only extend custom strikes to ±250 % of spot; beyond that vols explode and liquidity is zero



SNAP_CSV_DIR = Path("snapshot_csv")
SNAP_CSV_DIR.mkdir(exist_ok=True)

OUTDIR = Path("smile_plots")
OUTDIR.mkdir(exist_ok=True)


# Helper functions
# -----------------------------------------------------------------------------

def parse_instr(name: str) -> Tuple[str, float, str]:
    under, _date, strike, opt_type = name.split("-")
    return under, float(strike), opt_type.upper()
#Black-Scholes formula 
def black_price(S: float, K: float, T: float, sigma: float,
                is_call: bool, r: float = 0.0) -> float:

    if T <= 0 or sigma <= 0:
        return max(0.0, S - K) if is_call else max(0.0, K - S)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    N = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
    disc = math.exp(-r * T)
    return (S * N(d1) - K * disc * N(d2)) if is_call else (K * disc * N(-d2) - S * N(-d1))
#Interpolating the implied volatility surface
def fit_iv_surface(df_tick: pd.DataFrame) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """Return per‑instrument IV map & per‑underlying smile‑coeff map."""
    iv_map: Dict[str, float] = {}
    coeffs_map: Dict[str, np.ndarray] = {}
    if df_tick.empty:
        return iv_map, coeffs_map

    # keep only liquid quotes
    df_liq = df_tick[(df_tick.ask_iv > 0) & (df_tick.bid_iv > 0)].copy()
    df_liq['iv_spread'] = df_liq.ask_iv - df_liq.bid_iv
    df_liq = df_liq[(df_liq.iv_spread <= SPREAD_IV_MAX)
                    & (df_liq.ask_sz >= SIZE_MIN)
                    & (df_liq.bid_sz >= SIZE_MIN)]
    if df_liq.empty:
        return iv_map, coeffs_map

    for under, grp in df_liq.groupby('underlying'):
        S = grp.index_price.iloc[0]
        x = np.log(grp.strike / S)
        y = 0.5 * (grp.bid_iv + grp.ask_iv)
        w = 1.0 / np.square(grp.iv_spread)
        if len(x) < POLY_DEGREE + 1:
            continue
        coeffs = np.polyfit(x, y, POLY_DEGREE, w=w)
        coeffs_map[under] = coeffs
        poly = np.poly1d(coeffs)
        df_under = df_tick[df_tick.underlying == under]
        iv_all = poly(np.log(df_under.strike / S)).clip(0.05, 3.0)
        iv_map.update({ins: iv for ins, iv in zip(df_under.instrument, iv_all)})
    return iv_map, coeffs_map
# Calculate micro price tilting towards thiner sides 
def micro_price(rows: pd.DataFrame) -> float:
    bids = rows[rows.side == 'bids']
    asks = rows[rows.side == 'asks']
    if bids.empty or asks.empty:
        return np.nan
    best_bid = bids.loc[bids.price.idxmax()]
    best_ask = asks.loc[asks.price.idxmin()]
    return (best_ask.price * best_bid.size + best_bid.price * best_ask.size) / (
        best_bid.size + best_ask.size
    )
def plain_mid(rows: pd.DataFrame) -> float:
    bids = rows[rows.side == "bids"]
    asks = rows[rows.side == "asks"]
    if bids.empty or asks.empty:
        return np.nan
    return 0.5 * (bids.price.max() + asks.price.min())

def vwap_mid(rows: pd.DataFrame, depth_levels: int = 3) -> float:
    bids = rows[rows.side == "bids"].nlargest(depth_levels, "price")
    asks = rows[rows.side == "asks"].nsmallest(depth_levels, "price")
    if bids.empty or asks.empty:
        return np.nan
    vwap_bid = (bids["price"] * bids["size"]).sum() / bids["size"].sum()
    vwap_ask = (asks["price"] * asks["size"]).sum() / asks["size"].sum()
    return (vwap_bid * asks["size"].sum() + vwap_ask * bids["size"].sum()) / (
           bids["size"].sum() + asks["size"].sum())

# Calculate liquidity weight
def liquidity_weight(spread: float, depth: float) -> float:
    return depth / (depth + LIQ_K * spread) if spread > 0 else 1.0
# Plotting the implied volatility smile with respect to Ln(K/S)
def plot_smile(df_tick: pd.DataFrame, coeffs_map: Dict[str, np.ndarray], snap: int) -> None:
    for under, grp in df_tick.groupby('underlying'):
        coeffs = coeffs_map.get(under)
        if coeffs is None:
            continue
        S = grp.index_price.iloc[0]
        x = np.log(grp.strike / S)
        y_mid = 0.5 * (grp.bid_iv + grp.ask_iv)
        poly = np.poly1d(coeffs)
        xx = np.linspace(x.min() - 0.5, x.max() + 0.5, 300)
        yy = poly(xx)
        fig, ax = plt.subplots()
        ax.scatter(x, y_mid, label='market mid', marker='o')
        # color='#21C7B2' is the colour of the Deribit logo
        ax.plot(xx, yy, label='fitted', linewidth=2,color='#21C7B2')
        ax.set_title(f"{under} smile – snapshot {snap}")
        ax.set_xlabel("ln(K/S)")
        ax.set_ylabel("implied vol")
        ax.grid(True, linestyle=':')
        ax.legend()
        fname = OUTDIR / f"{under}_snap{snap:03d}.png"
        fig.savefig(fname, dpi=120, bbox_inches='tight')
        plt.close(fig)

# Custom‑strikes
# -----------------------------------------------------------------------------

def append_custom_strikes(
    df: pd.DataFrame,
    coeffs_map: Dict[str, np.ndarray],
    custom_strikes: List[float],
) -> pd.DataFrame:
    """Add hypothetical quotes for requested strikes.
     IV is capped by the highest observed ask IV for that underlying.
    Theoretical coin price is clipped to the interval [0,1].
    """
    if not custom_strikes:
        return df

    expiry_ts = time.mktime(time.strptime(EXPIRY, "%d%b%y"))
    now = time.time()
    T = max(expiry_ts - now, 0.0) / 31557600.0
    spot_map = df.groupby('underlying')['index_price'].first().to_dict()
    if not spot_map:
        return df



    rows: List[dict] = []
    existing = df.groupby('underlying')['strike'].unique().to_dict()

    for K in custom_strikes:
        
        under = min(spot_map.keys(), key=lambda u: abs(K - spot_map[u]))
        S = spot_map[under]

        if K / S > MAX_MONEYNESS or S / K > MAX_MONEYNESS:
            continue
        if K in existing.get(under, []):
            continue

        coeffs = coeffs_map.get(under)
        if coeffs is None:
            continue

        # IV cap
        ask_iv_max = df[(df.underlying == under) & df.ask_iv.notna()].ask_iv.max()
        iv_cap = float(ask_iv_max) if not np.isnan(ask_iv_max) else 2.0

        poly = np.poly1d(coeffs)
        raw_iv = float(poly(math.log(K / S)))
        iv = np.clip(raw_iv, 0.05, iv_cap)

        for opt_type in ['C', 'P']:
            usd_pv = black_price(S, K, T, iv, opt_type == 'C')
            theo_coin = np.clip(usd_pv / S if S > 0 else np.nan, 0, 1.0)
            instr = f"{under}-{EXPIRY}-{int(K)}-{opt_type}"
            rows.append({
                'instrument': instr,
                'underlying': under,
                'strike': float(K),
                'type': opt_type,
                'index_price': S,
                'ts': int(now * 1000),
                'best_bid': np.nan,
                'best_ask': np.nan,
                'bid_iv': np.nan,
                'ask_iv': np.nan,
                'bid_sz': np.nan,
                'ask_sz': np.nan,
                'micro': np.nan,
                'mark_px': np.nan,
                'my_mark_px': theo_coin,
                'is_custom': True
            })

        existing.setdefault(under, np.array([]))
        existing[under] = np.append(existing[under], K)

    if rows:
        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    return df

# Main function
# -----------------------------------------------------------------------------

async def main(args) -> None:
    names = discover_names(args.expiry)
    if not names:
        raise RuntimeError(f"No live options for {args.expiry}")

    stream = DeribitStream()
    await stream.connect()
    await stream.subscribe(names, DEPTH)

    prev_iv_map: Dict[str, float] = {}
    prev_coeffs_map: Dict[str, np.ndarray] = {}

    end_time = time.time() + args.t1
    snap = 0
    print(f"Running until {time.strftime('%H:%M:%S', time.localtime(end_time))}, "
          f"snapshots every {args.t2}s")
    avg_plain_series = []
    avg_micro_series = []
    avg_vwap_series  = []       #
    avg_our_series   = [] 
    while time.time() < end_time:
        ...
        await asyncio.sleep(args.t2)
        snap += 1

        tick = stream.ticker_df()
        book = stream.book_df()
        if tick.empty or book.empty:
            continue


        tick[['underlying', 'strike', 'type']] = tick.instrument.apply(
            lambda s: pd.Series(parse_instr(s)))
        book[['underlying', 'strike', 'type']] = book.instrument.apply(
            lambda s: pd.Series(parse_instr(s)))


        mprice = (
            book.groupby('instrument', group_keys=False)
                .apply(micro_price, include_groups=False)  
                .rename('micro')
        )
        plain   = (book.groupby('instrument', group_keys=False)
             .apply(plain_mid, include_groups=False)
             .rename('plain_mid'))

        vwap3   = (book.groupby('instrument', group_keys=False)
                    .apply(vwap_mid, include_groups=False)
                    .rename('vwap3'))

        benchmarks = pd.concat([mprice, plain, vwap3], axis=1)   # index = instrument
        tick = tick.join(benchmarks, on="instrument")
               


        iv_map, coeffs_map = fit_iv_surface(tick)
        if not iv_map:
            iv_map, coeffs_map = prev_iv_map, prev_coeffs_map
        else:
            prev_iv_map, prev_coeffs_map = iv_map, coeffs_map

        
        expiry_ts = time.mktime(time.strptime(EXPIRY, "%d%b%y"))
        def blend_mark(r):
            micro = r.micro
            spread = r.best_ask - r.best_bid
            depth = r.bid_sz + r.ask_sz
            theo_iv = iv_map.get(r.instrument)
            if theo_iv is None or np.isnan(theo_iv):
                theo_c = micro
            else:
                T_loc = max(expiry_ts - r.ts / 1000.0, 0.0) / 31557600.0
                usd = black_price(r.index_price, r.strike, T_loc, theo_iv, r.type == 'C')
                theo_c = usd / r.index_price
            if np.isnan(micro):
                micro = theo_c
            lam = liquidity_weight(spread, depth)
            blended = lam * micro + (1 - lam) * theo_c
            # clip to [0,1] interval
            return np.clip(blended, 0, 1.0)

        tick['my_mark_px'] = tick.apply(blend_mark, axis=1)
        tick['is_custom'] = False  

        # custom strikes
        tick = append_custom_strikes(tick, coeffs_map, args.custom_strikes)
        tick.sort_values('strike', inplace=True)


        # Snapshot CSV output
        # ------------------------------------------------------------------
        df_out = (tick[['instrument', 'strike',
                        'plain_mid', 'micro', 'vwap3',
                        'my_mark_px', 'mark_px', 'is_custom']]
                    .rename(columns={'my_mark_px': 'our_mark_px',
                                     'mark_px':   'deribit_mark_px'}))

        # --- diff columns ----------------------------------------------------
        for col in ['plain_mid', 'micro', 'vwap3', 'our_mark_px']:
            df_out[f'diff_{col}'] = df_out[col] - df_out['deribit_mark_px']

        # numeric diff (NaN if Deribit price missing)
        df_out['diff'] = df_out['our_mark_px'] - df_out['deribit_mark_px']

        # label rows with missing Deribit (doesn't exist)
        df_out['deribit_mark_px'] = df_out['deribit_mark_px'].fillna('doesnt exist')

        # split by flag
        df_std  = df_out[~df_out['is_custom']].drop(columns='is_custom').round(4).copy()
        df_cust = df_out[df_out['is_custom'] ].drop(columns='is_custom').round(4).copy()
        avg_plain = df_std['diff_plain_mid'].mean()
        avg_micro = df_std['diff_micro'].mean()
        avg_vwap3 = df_std['diff_vwap3'].mean()
        avg_our   = df_std['diff_our_mark_px'].mean()

        avg_plain_series.append(avg_plain)
        avg_micro_series.append(avg_micro)
        avg_vwap_series.append(avg_vwap3)
        avg_our_series.append(avg_our) 
        print(f"Snapshot {snap}: "
              f"plain={avg_plain:.5f} | micro={avg_micro:.5f} | vwap3={avg_vwap3:.5f}| our={avg_our:.5f}")         # store it
        csv_std  = SNAP_CSV_DIR / f"snapshot_{snap:03d}_normal.csv"
        csv_cust = SNAP_CSV_DIR / f"snapshot_{snap:03d}_custom.csv"

        df_std.to_csv(csv_std,  index=False, float_format="%.4f")
        df_cust.to_csv(csv_cust, index=False, float_format="%.4f")

        cols = ['instrument', 'best_bid', 'best_ask', 'my_mark_px', 'mark_px']
        preview = tick[cols].head(10).round(4)
        print(f"\nSnapshot {snap} @ {time.strftime('%H:%M:%S')}")
        print(preview.to_string(index=False))

        # plots every N snapshots
        if snap % PLOT_EVERY == 0:
            plot_smile(tick, coeffs_map, snap)
        if avg_plain_series:
            fig, axs = plt.subplots(2, 2, figsize=(10, 6))
            for ax, data, title in zip(axs.ravel(),
                    [avg_plain_series, avg_micro_series, avg_vwap_series, avg_our_series],
                    ["plain mid", "micro", "vwap‑3", "our mark"]):
                ax.plot(data, marker='o')
                ax.set_title(title); ax.grid(True)
                ax.set_xlabel("snapshot #"); ax.set_ylabel("diff")
            fig.tight_layout()
            fig.savefig("avg_diff_grid.png", dpi=120)
    await stream.ws.close()
    print("Finished. Smile plots in", OUTDIR)
    print("CSV snapshots in", SNAP_CSV_DIR)

# -----------------------------------------------------------------------------
# Simple unit test
# -----------------------------------------------------------------------------

def test_custom_logic():
    """Lightweight unit test for custom‑strike logic."""
    data = [
        {
            'instrument': 'TST-XX-90-C', 'underlying': 'TST', 'strike': 90, 'type': 'C',
            'index_price': 100, 'ts': 0, 'best_bid': 1, 'best_ask': 2,
            'bid_iv': 0.20, 'ask_iv': 0.25, 'bid_sz': 50, 'ask_sz': 50,
        },
        {
            'instrument': 'TST-XX-100-C', 'underlying': 'TST', 'strike': 100, 'type': 'C',
            'index_price': 100, 'ts': 0, 'best_bid': 3, 'best_ask': 4,
            'bid_iv': 0.22, 'ask_iv': 0.27, 'bid_sz': 50, 'ask_sz': 50,
        },
    ]
    df = pd.DataFrame(data)
    iv_map, coeffs_map = fit_iv_surface(df)
    custom = [95]
    df2 = append_custom_strikes(df, coeffs_map, custom)
    print(df2.round(4))




# Entry point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('expiry', type=str, help='Expiry code e.g. 23MAY25')
    parser.add_argument('t1',     type=int, help='Total runtime seconds (T1)')
    parser.add_argument('t2',     type=int, help='Interval between snapshots (T2)')
    parser.add_argument('custom_strikes', type=float, nargs='*',
                        help='Additional strike prices to include')
    parser.add_argument('--test', action='store_true', help='Run simple test instead of live')
    args = parser.parse_args()

    if args.test:
        test_custom_logic()
    else:
        asyncio.run(main(args))
