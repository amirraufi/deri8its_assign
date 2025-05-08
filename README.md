
This repository is my submission for the **Deribit Junior Quant Risk assignment**.  
It contains two self‑contained mini‑projects (“tasks”) that demonstrate live
data ingestion, custom pricing logic, volatility‑surface fitting and
result visualisation.

## Repository Structure

- **Task 1: Mark-Price Blender & Custom-Strike Generator.**  
  - `calculating_own_mark_price.py` — live streaming & mark-price script  
  - `snapshot_csvs/` — auto-generated per-snapshot CSVs  
  - `smile_plots/` — auto-generated IV-smile PNGs  
  - `avg_diff_grid.png` — model-vs-Deribit comparison chart  
  - `README.md` — detailed methodology & usage instructions  

- **Task 2: Coconut-Pricing Puzzle.**  
  - `coconut_price.py` — web-scraping & ratio computation  
  - `README.md` — concise explanation of approach  

- **Top-Level Files**  
  - `README.md` — this overview  
  - `requirements.txt` — project dependencies (async-websocket, pandas, numpy, …)  

