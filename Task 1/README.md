# coconut-price-detection

## Determine the mark price for every listed strike on Deribit (given a single expiry) and synthesise fair marks for non‑existent strikes.

We generate a fair mark‑price for every option by taking the best of two worlds and letting the data tell us how much weight to give each one:

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

 
If the book is deep & tight → λ ≈ 1 → follow the market.
If the quote is thin or wide (far OTM wings) → λ → 0 → fall back to theory.
Because the IV surface is continuous, we can price any user‑requested strike on the fly and keep the sheet arbitrage‑free.

A live dashboard plots the tracking error of this blended mark against three quick‑and‑dirty yardsticks (plain mid, micro, VWAP‑3). Our mark is not always the closest to Deribit’s official mark, but that was never the brief; the goal was to demonstrate a defensible, tunable methodology. By sliding L_k you can instantly pivot from “pure market” to “pure model” without touching the rest of the code.

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
2. Data flow

Real‑time feed – WebSocket subscription to all instruments for the chosen expiry.
Snapshot every T2 seconds – build two DataFrames:
tick (best quotes, Deribit mark)
book (full depth used to compute micro‑price).
Micro‑price – size‑weighted mid of the top L1 quotes
(P_ask*Q_bid +P_bid*Q_ask)/(Q_bid + Q_ask)


​	
 
​	
 
– captures order‑book imbalance.
Fit IV surface – for each underlying, polynomial in 
log
⁡
(
K
/
S
)
log(K/S) using only “liquid” quotes (tight spread, ≥ SIZE_MIN on both sides).
Theoretical coin price – Black–Scholes (undiscounted) with that IV.
Liquidity blend –
mark
=
λ
 
micro
+
(
1
−
λ
)
 
theoretical
mark=λmicro+(1−λ)theoretical
with
λ
=
depth
depth
+
L
k
⋅
spread
λ= 
depth+L 
k
​	
 ⋅spread
depth
​	
 
→ flows to theory when depth is thin / spread wide.
Custom strikes – same IV surface + Black–Scholes, flagged is_custom=True.

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

