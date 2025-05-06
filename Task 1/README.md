# coconut-price-detection

## Determine the mark price for every listed strike on Deribit (given a single expiry) and synthesise fair marks for non‑existent strikes.

I have generated a fair mark‑price for every option by taking the best of two worlds and letting the data tell us how much weight to give each one:

Micro‑price (market signal)
A size‑weighted mid‑price that leans toward the heavier side of the order book — the quickest read of where traders really want to deal.
Theoretical price (model signal)
A no‑arbitrage Black‑Scholes value using an implied‑volatility surface we fit to the liquid quotes (tight spreads, decent size).
The surface is quadratic in ln(K/S); simple, smooth and robust.
We blend them with a liquidity switch

<p align="center">
  <strong>mark</strong> = &lambda;<sub>liq</sub> &middot; micro  
  &nbsp;&nbsp;+ (1 − &lambda;<sub>liq</sub>) &middot; theory  
  <br>
  &lambda;<sub>liq</sub> = depth / (depth + L<sub>k</sub> &middot; spread)
</p>

 depth = total number of contracts at the top of the book (bid_size + ask_size)
If the book is deep & tight → λ ≈ 1 → follow the market.
If the quote is thin or wide (far OTM wings) → λ → 0 → fall back to theory.
Because the IV surface is continuous, we can price any user‑requested strike on the fly and keep the sheet arbitrage‑free.

A live dashboard plots the tracking error of this blended mark against three norm models (plain mid, micro, VWAP‑3). Our mark is not always the closest to Deribit’s official mark. As in the instruction it has been noted that the methodology is more important. By sliding L_k you can instantly pivot from “pure market” to “pure model” without touching the rest of the code.

---

## How to Run the Application

1. Clone the repository: `git clone https://github.com/amirraufi/deri8its_assign.git'
2. Navigate into the project directory: `cd deri8its_assign`
3. Navigate into task 2: 'cd "Task 1"'
5. Create and activate a virtual environment:
   - On Unix/macOS: `python3 -m venv .venv source .venv/bin/activate `
   - On Windows: `python -m venv .venv\Scripts\activate`
6. Install dependencies: `pip install -r requirements.txt`
7. Run the application: `python3 calculating_own_mark_price.py "Expiry" "run time" "interval times" "strikes" '
8. example: python3 calculating_own_mark_price.py 23MAY25 60 5 \
> >   1500 1600 1700 1800 1900 2000 2100 2200 2400 2600 \
> >   88000 90000 92000 94000 96000 98000 100000 102000 104000 106000 \
> >   500 3000 50000 200000
9. to get the csv files type: 'code snapshot_csvs'
10. to get the vol smiles type: 'code smile_plots'
11. to see the performance chart of average differences type: 'avg_diff_grid.png'



---
2. Data Flow

Below is the end-to-end pipeline, referencing the helper functions in calculating_own_mark_price.py:

WebSocket Snapshot
Every T₂ seconds we pull two tables from DeribitStream:
tick: best-bid/ask from Deribit + Deribit’s mark price
book: full L1 depth (price & size on both sides)
Parse Instruments
parse_instr()  →  (underlying, strike, type)
Compute Reference Prices
micro_price()  →  imbalance-weighted mid
plain_mid()    →  simple (bid + ask)/2
vwap_mid()     →  VWAP over top 3 levels
Group book by instrument, apply each function → three series: micro, plain_mid, vwap3.
Join Benchmarks
benchmarks = concat([micro, plain_mid, vwap3], axis=1)
tick = tick.join(benchmarks, on="instrument")
Fit IV Surface
fit_iv_surface(df_tick)
Filters only “liquid” quotes:
• ask_iv > 0, bid_iv > 0
• iv_spread = ask_iv – bid_iv ≤ SPREAD_IV_MAX
• ask_sz, bid_sz ≥ SIZE_MIN
Fits a quadratic in ln(K/S) weighted by 1/iv_spread².
Outputs
iv_map: per‐instrument smoothed IV
coeffs_map: per‐underlying smile coefficients
Compute Theoretical Price
black_price(S, K, T, σ, is_call)
– Undiscounted Black–Scholes in coin terms
– theoretical_coin_price = black_price(...) / S
Blend Market & Theory
liquidity_weight(spread, depth)
blend_mark(row)  →  my_mark_px
depth = bid_sz + ask_sz
spread = best_ask – best_bid
Append Custom Strikes
append_custom_strikes(tick, coeffs_map, custom_strikes)
For each K*, finds nearest S, enforces K*/S ≤ MAX_MONEYNESS
Evaluates smile polynomial at ln(K*/S), clips IV to observed max
Computes BS price → coin → flagged is_custom=True
Output Artifacts
CSV snapshots
snapshot_csv/snapshot_XXX_normal.csv  
snapshot_csv/snapshot_XXX_custom.csv
Volatility smiles
smile_plots/{UNDERLYING}_snapXXX.png
Performance chart
avg_diff_grid.png
Compares average differences (mean(our_mark_px – Deribit_mark_px))
for plain_mid, micro_price, vwap3, and our blended mark.
This workflow ensures that our mark prices gracefully transition from pure market data to theory as liquidity conditions warrant, while providing clear visual and tabular outputs for validation and comparison.


## API Choice and Justification



**Why this API?**

- **Accuracy:** Since the shop priced coconuts based on *previous-day settlements*, it was important to use the same final values rather than transient mid-market prices.
- **Historical Depth:** The `get_deliveries` endpoint allows fetching multiple pages of past data, enabling the project to search across a wide date range.
- **Instrument Coverage:** The API supports multiple currencies and indices (BTC, ETH, SOL, PAXG, XRP, ADA), aligning with the symbols shown on the payment terminal.
- **Programmatic Access:** Well-documented and consistent structure made it straightforward to automate the retrieval and comparison process.

For full documentation: https://docs.deribit.com/#public-get_deliveries


---

## Logic and Design Decisions

### Summary

- The script collects **historical delivery prices** for 6 Deribit index pairs: `btc_usdc`, `eth_usdc`, `sol_usdc`, `paxg_usdc`, `xrp_usdc`, and `ada_usdc`.
- Based on visible amounts in the image, we compute **target price ratios** (e.g., `btc/eth`, `sol/paxg`, `xrp/ada`).
- We fetch historical prices from **both testnet and mainnet**, compute the ratios for each date, and identify which dates match the known ratios within a **tolerance level**.
- From this match, we extract the ETH-to-USD conversion rate and use it to calculate the **USD price of a king coconut**, known to cost `0.0013371 ETH`.

### Efficiency Features

- **Concurrent Fetching with Semaphores:** We fetch data in parallel using `asyncio` and `asyncio.Semaphore` to limit concurrency to 5 WebSocket connections, reducing I/O wait while avoiding Deribit rate limits.
- **Polars for DataFrame Processing:** We use Polars instead of Pandas for its significantly faster performance in handling large datasets and expressive syntax for filtering and computing ratios.

---

## Configuration and Assumptions

| Parameter           | Value        | Reasoning |
|---------------------|--------------|-----------|
| MAX_REQUEST_COUNT   | 100          | Max allowed by Deribit API per request |
| TOTAL_RECORDS       | 1000         | Balance between historical depth and API throttling risk |
| TOLERANCE           | 1e-3         | Small enough to capture only close ratio matches but still flexible for rounding and float error |
| CONCURRENCY_LIMIT   | 5            | Limits the number of concurrent requests to avoid Deribit rate errors |

---

## Key Challenges

Finding the level 2 data to find the size of the asks and bids for the vwap model, where as it was a challenge for me to get that data from the API. The differences in the prices of all the models was very good and close and inline with deribits for close to money strike prices but for further very ITM or very OTM it was a bit far from deribits mark price. As the requirement of the assignment I have not used the mark prices of deribit and had to come up with challenges regarding ways to make the vol smile and how to come up with my own mark prices. However, first I have used mid prices and came up with mark prices and used brent solver to get IV of the options byt then I realised as the theory is more important I came up with a blend solution that will fix the problem of the tails that have very wide bid and ask with a more theoretical rich approach were it is dynamic and you can change the weight allocated to the standard mid price and the builded IV model

---


---


---

## References

- Deribit Public API Documentation: https://docs.deribit.com
- Polars Python API: https://pola-rs.github.io/polars/py-polars/html/reference/index.html
- Asyncio – Python Docs: https://docs.python.org/3/library/asyncio.html
- WebSockets for Python: https://websockets.readthedocs.io/

