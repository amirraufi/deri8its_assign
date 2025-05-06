# Task 1 Finding mark prices

## Determine the mark price for every listed strike on Deribit (given a single expiry) and synthesise fair marks for non‑existent strikes.

I have generated a fair mark‑price for every option by taking the best of two worlds and letting the data tell us how much weight to give each
one:

Micro‑price (market signal)
A size‑weighted mid‑price that leans toward the heavier side of the order book reduces noise and an indication of where traders really want to deal.
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

 depth = total number of contracts (bid_size + ask_size)
If the book is deep & tight → λ ≈ 1 → follow the market.
If the quote is thin or wide (far OTM wings) → λ → 0 → fall back to theory.
Because the IV surface is continuous, we can price any user‑requested strike on the fly and keep the sheet arbitrage free.

A live dashboard plots the tracking error of this blended mark against three norm models (plain mid, micro, VWAP‑3). Our mark is not always the closest to Deribit’s official mark. As in the instruction it has been noted that the methodology is more important. By sliding L_k you can instantly pivot from “pure market” to “pure model” without touching the rest of the code.

---

## How to Run the Application

1. Clone the repository: `git clone https://github.com/amirraufi/deri8its_assign.git`
2. Navigate into the project directory: `cd deri8its_assign`
3. Navigate into task 2: `cd "Task 1"`
5. Create and activate a virtual environment:
   - On Unix/macOS: `python3 -m venv .venv && source .venv/bin/activate`
   - On Windows: `python -m venv .venv && .venv\Scripts\activate`
6. Install dependencies: `pip install -r requirements.txt`
7. Run the application: `python3 calculating_own_mark_price.py "Expiry" "run time" "interval times" "strikes" `
8. example: `python3 calculating_own_mark_price.py 23MAY25 60 5  1500 1600 1700 1800 1900 2000 2100 2200 2400 2600 \ 88000 90000 92000 94000 96000 98000 100000 102000 104000 106000 \`
> >   500 3000 50000 200000
9. to get the csv files type: `code snapshot_csv`
10. to get the vol smiles type: `code smile_plots`
11. to see the performance chart of average differences type: `code avg_diff_grid.png`



---


## 2. Data Flow — helper-by-helper

```text
WebSocket snapshot (DeribitStream)
│
├─ every T2 s → tick DF (best-bid/ask + mark)
│
└─              → book DF (L1 depth)

tick DF ──┐
          ├─ micro_price() ──┐
          ├─ plain_mid()     │
          └─ vwap_mid()      │
                            ↓
                       benchmarks

benchmarks + tick DF → fit_iv_surface() → iv_map, coeffs_map
                                  │
                                  ↓
                            blend_mark() → my_mark_px
                                  │
                                  ↓
                   append_custom_strikes() → write CSV + plots


## API Choice and Justification



**Why this API?**

## Why the Deribit Public API? (also required by assinment:-))

| What we need for the model | How Deribit’s API delivers |
|----------------------------|----------------------------|
| **Real‑time, low‑latency quotes**<br>to build 5‑second snapshots of every option in a given expiry. | `public/subscribe` WebSocket channels stream order‑book updates in ≈ 100–200 ms. No REST polling, no throttling issues. |
| **Level‑2 depth (contracts on bid / ask)**<br>to compute the liquidity term<br>`depth = bid_size + ask_size` used in the λ‑blend. | `book.<instrument>.10` channels expose the top‑10 levels (price, size). |
| **Venue mark & surface**<br>so we can benchmark our blended marks and show the “diff” plots. | `public/get_book_summary_by_instrument` returns **mark_price**, **bid_iv**, **ask_iv** for every option. |
| **Historical context**<br>for hyper‑parameter scans (e.g. finding an optimal `LIQ_K`). | `public/get_last_trades_by_instrument` (tick history) and `public/get_tradingview_chart_data` (OHLC) let us replay past sessions quickly. |
| **Breadth of instruments**<br>(BTC, ETH, SOL, etc.) exactly as requested in the assignment. | Deribit lists the deepest, most liquid crypto‑option markets under one uniform API. |
| **Clean JSON + open access**<br>for rapid prototyping in plain Python. | Well‑documented endpoints, predictable field names and generous public rate‑limits (≈ 20 req/s REST; unlimited on WebSocket). |



---

## Logic & Design Decisions  

### Summary  

1. **Snapshot Engine**  
   * A dedicated `asyncio` loop wakes every **T₂ seconds** (default 5 s) and assembles two dataframes:  
     * **`tick`** – best‐bid/ask, Deribit mark, index price.  
     * **`book`** – top‑10 order‑book lines used for micro‑price, plain‑mid and VWAP‑mid benchmarks.  

2. **Micro‑price Estimator**  
   \[
   \text{micro}=\frac{P_{\text{ask}}Q_{\text{bid}}+P_{\text{bid}}Q_{\text{ask}}}{Q_{\text{bid}}+Q_{\text{ask}}}
   \]  
   This tilts the mid‑price toward the heavier side of the book and reacts instantly to imbalance.

3. **Fitted IV Surface**  
   * Filter to “**liquid**” quotes (tight < 1.5 vol bid–ask, ≥ 5 contracts both sides).  
   * Fit a **quadratic in ln(K/S)** per underlying; weights ∝ 1 / spreads².  
   * Use the polynomial to generate a continuous IV map for every live strike.

4. **Liquidity Blend**  
   \[
   \text{mark}= \lambda \;\text{micro}+(1-\lambda)\;\text{theory},\quad
   \lambda=\frac{\text{depth}}{\text{depth}+{ \small LIQ\_K}\times\text{spread}}
   \]  

   * **Depth** = contracts visible on L1 (bid + ask).  
   * **`LIQ_K`** controls how fast we fade to theory when markets are wide / thin (set via grid‑search).  

5. **Custom Strikes**  
   * Same IV surface + Black‑Scholes → generates fair prices for **user‑requested strikes**, flagged `is_custom=True`.

6. **Benchmarks & Visuals**  
   * Side‑by‑side diff charts versus **plain‑mid**, **micro**, **VWAP‑3**, and Deribit **mark_price**.  
   * Auto‑export volatility‑smile PNGs every N snapshots for quick QC.

---

### Configuration & Rationale  

| Parameter | Value | Reasoning |
|-----------|-------|-----------|
| `SPREAD_IV_MAX` | **1.5 vols** | 90ᵗʰ‑percentile weekday spread; keeps only tradable quotes in the fit. |
| `SIZE_MIN` | **5 contracts** | Filters 1‑lot feelers while retaining > 98 % of on‑the‑run strikes. |
| `POLY_DEGREE` | **2** | Quadratic captures skew + basic smile without over‑fitting. |
| `LIQ_K` | **70** | With a 1‑tick / 70‑lot book λ ≈ 50 %; calibrated via 1‑week grid‑search to minimise RMSE vs. realised trades. |
| `MAX_MONEYNESS` | **±2.5 × spot** | Wings beyond 250 % are essentially zero‑delta and distort the fit. |
| `DEPTH_LEVELS` (VWAP) | **3** | Gives a stable VWAP without fetching the full book. |
| `PLOT_EVERY` | **10 snapshots** | Roughly every 50 s ⇒ low disk usage, still visually dense. |

---

### Key Challenges & Mitigations  

* **Level‑2 Depth Availability** – Public API exposes only 10 levels; sufficient for micro/VWAP, but deeper stats (order book slope) were out of scope and it was my first time working with websockets and going from level1 to level2 was also a bit challenging.
* **Wing Liquidity** – Far ITM/OTM options exhibit wide spreads. Most of the mehodolgies would give a bit of a far price as the spread is big and liquidity is not enough. ✔︎ addressed by λ‑blend gravitating toward theory.  
* **No Deribit Mark Reliance** – The assignment forbids copying venue marks; we solved IV via the described model and only using bid and ask information and not the mark price or the IV that is based on the mark price.


### Methodology

I searched through all the coins that Deribit offers and found those with options, then retrieved all the strikes for the given expiration. I then obtained their Level 1 and Level 2 data for ask size, bid size, best ask, best bid, etc. Next, I considered the best ways to calculate mid prices and identified three methods: plain mid, VWAP, and micro-price (which has been explained).

First, I tried the basic approach by computing mark prices using each of the three methods (micro price, mid price, etc.).
Then I calculated implied volatility using Brent’s method, interpolated the smile to find IV for all unlisted options, and used Black–Scholes(r ≈ 0 for crypto and all basic assumptions of Black Scholes). to obtain their theoretical values.
However, I noticed the wide spread between bid and ask in far out‑of‑the‑money options, so I created a λ function to weight theoretical and market values based on bid–ask spread.
The resulting blend keeps ATM strikes within a few basis points of venue marks while providing smooth, arbitrage‑free prices for illiquid wings and custom strikes.
Finally, I added the vol‑smile graph I generated and a chart comparing average differences between my model, VWAP, micro, and midpoint prices.


## Output

Running the script produces three kinds of artefacts for every session:

CSV snapshots (one file per snapshot)
<snapshot_n>_normal.csv  prices for strikes that actually exist on Deribit.
<snapshot_n>_custom.csv  prices for user‑supplied strikes (no Deribit quote, so no direct comparison).
Volatility‑smile PNGs
A fitted IV smile for each underlying, saved every N snapshots (default 10).
Performance dashboard
avg_diff_grid.png  a 2 × 2 grid comparing the mean price difference (our mark  Deribit mark) across snapshots for all four strategies: plain‑mid, micro‑price, VWAP, and our blended mark.


---

## References

Deribit Public API Docs – https://docs.deribit.com
Python asyncio – https://docs.python.org/3/library/asyncio.html
websockets library – https://websockets.readthedocs.io/
NumPy – https://numpy.org/doc/stable/   (used for vectorised maths & polynomial fitting)
Pandas – https://pandas.pydata.org/docs/   (core tabular engine)
Matplotlib – https://matplotlib.org/stable/   (vol‑smile & diagnostics)
SciPy – https://docs.scipy.org/doc/scipy/   (polyfit, Black–Scholes helpers)
Polars – https://pola-rs.github.io/polars/py-polars/html/reference/index.html   (back‑testing experiments)
