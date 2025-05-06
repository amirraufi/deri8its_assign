
This repository is my submission for the **Deribit Junior Quant Risk assignment**.  
It contains two self‑contained mini‑projects (“tasks”) that demonstrate live
data ingestion, custom pricing logic, volatility‑surface fitting and
result visualisation.

## Repository Structure
├─ Task 1/ # Mark‑price blender & custom‑strike generator (main focus)
│ ├─ calculating_own_mark_price.py ← live script
│ ├─ snapshot_csv/ ← auto‑generated CSVs
│ ├─ smile_plots/ ← auto‑generated IV‑smile PNGs
│ ├─ avg_diff_grid.png ← model‑vs‑Deribit comparison chart
│ └─ README.md ← deep dive into methodology
│
├─ Task 2/ # (Optional) Coconut‑pricing puzzle — web‑scraping + ratios
│ ├─ coconut_price.py
│ └─ README.md ← short explanation
│
├─ requirements.txt # Minimal Python deps (async‑websocket, pandas, numpy…)
└─ top‑level README.md (you’re here)


