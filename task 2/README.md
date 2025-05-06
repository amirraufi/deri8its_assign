# coconut-price-detection

## Determine the Historical USD Price of a King Coconut Using Cryptocurrency Clues

This project is a Python-based solution to determine the historical USD price of a king coconut from a photo of a payment terminal. The shop in question accepted multiple cryptocurrencies, and the price was based on the **previous day's Deribit settlement prices**. However, the **exact date** and **whether the prices came from testnet or mainnet** were unknown.

This tool programmatically investigates both Deribit environments, aligns historical cryptocurrency ratios with known coconut pricing ratios, and identifies possible historical dates on which the price could have been calculated.

---

## How to Run the Application

1. Clone the repository: `git clone https://github.com/8ehrad/Crypto-Coconut-Pricing.git`
2. Navigate into the project directory: `cd Crypto-Coconut-Pricing`
3. Create and activate a virtual environment:
   - On Unix/macOS: `python3 -m venv .venv && source .venv/bin/activate`
   - On Windows: `python -m venv .venv && .venv\Scripts\activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Run the application: `python Task2.py`

---

## API Choice and Justification

### API Used: `get_deliveries` – Deribit Public API

This project uses the `get_deliveries` endpoint from Deribit’s public API to retrieve historical **settlement prices** for various instruments. Unlike other endpoints that return live mark or index prices, `get_deliveries` provides **final settlement prices** at expiry, which are authoritative for determining real transaction prices in the context of Deribit derivatives.

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

- **Unknown Environment (Testnet or Mainnet):** I tackled this by analysing both environments in parallel and reporting the results independently or together if they matched.
- **Unknown Date:** By analysing 1000 past settlement prices per index, I ensured a reasonable temporal range without overwhelming the API.
- **API Constraints:** Deribit limits each delivery price request to a maximum of 100 records. Fetching 1000 records required batching 10 requests per index, multiplied across 6 indices. High concurrency caused errors, which were mitigated using semaphore-based throttling.
- **Floating-Point Precision Errors:** Ratios derived from the terminal image are subject to small inaccuracies. A tolerance of `0.001` was chosen empirically to allow slight drift while still avoiding false positives.

---

## Final output

- Date: 2024-11-11 (the photo would have been taken the day after this date)
- ETH/USD Rate: 3141.284
- Coconut Price (0.0013371 ETH): $4.200211

---

## Files in This Repo

| File               | Description                          |
|--------------------|--------------------------------------|
| `Task2.py`         | Main Python script                   |
| `requirements.txt` | Dependency list                      |
| `README.md`        | This documentation                   |

---

## References

- Deribit Public API Documentation: https://docs.deribit.com
- Polars Python API: https://pola-rs.github.io/polars/py-polars/html/reference/index.html
- Asyncio – Python Docs: https://docs.python.org/3/library/asyncio.html
- WebSockets for Python: https://websockets.readthedocs.io/

