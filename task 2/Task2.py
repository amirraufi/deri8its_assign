import asyncio
import websockets
import json
import polars as pl
from loguru import logger
from typing import Union

# Constants
MAX_REQUEST_COUNT = 100     # Max records per API call
TOTAL_RECORDS = 1000        # Total records to fetch per index
CONCURRENCY_LIMIT = 5       # Limit of concurrent WebSocket connections

# Indexes to retrieve historical delivery prices for
INDEX_NAMES = ['btc_usdc', 'eth_usdc', 'sol_usdc', 'paxg_usdc', 'xrp_usdc', 'ada_usdc']

# Crypto amounts we are targeting
CRYPTOS = {
    'ETH': 0.0013371,
    'BTC': 0.00005181,
    'SOL': 0.020196, 
    'PAXG': 0.001586, 
    'XRP': 7.2942,
    'ADA': 7.3376,  
}

# Compute target price ratios based on the amounts
TARGET_RATIOS = {
    "btc_eth_ratio": CRYPTOS["ETH"] / CRYPTOS["BTC"],
    "sol_paxg_ratio": CRYPTOS["PAXG"] / CRYPTOS["SOL"],
    "xrp_ada_ratio": CRYPTOS["ADA"] / CRYPTOS["XRP"]
}

# Allowable deviation in ratios
TOLERANCE = 1e-3 


async def retrieve_historical_prices(index_name: str, testnet: bool, offset: int, sem: asyncio.Semaphore) -> list[dict[str, Union[str, float]]]:
    """Fetches a batch of historical delivery prices for a given index using Deribit's WebSocket API."""
    api = 'wss://test.deribit.com/ws/api/v2' if testnet else 'wss://www.deribit.com/ws/api/v2'
    msg = {
        "jsonrpc": "2.0",
        "id": 3601,
        "method": "public/get_delivery_prices",
        "params": {
            "index_name": index_name,
            "offset": offset,
            "count": MAX_REQUEST_COUNT
        }
    }

    async with sem:  # Limit concurrent connections
        try:
            async with websockets.connect(api) as websocket:
                await websocket.send(json.dumps(msg))
                response = await websocket.recv()
                return json.loads(response)["result"]["data"]
        except Exception as e:
            logger.error(f"Error fetching {index_name} offset {offset}: {e}")
            return []  # Return empty list on failure to prevent crashing


async def fetch_all_prices(testnet: bool) -> dict[str, list]:
    """Fetches all batches for all index names."""
    sem = asyncio.Semaphore(CONCURRENCY_LIMIT)
    tasks = [
        retrieve_historical_prices(name, testnet, offset, sem)
        for name in INDEX_NAMES
        for offset in range(0, TOTAL_RECORDS, MAX_REQUEST_COUNT)
    ]
    results = await asyncio.gather(*tasks)

    # Organise results by index name
    data_by_index = {name: [] for name in INDEX_NAMES}
    for i, result in enumerate(results):
        index = INDEX_NAMES[i // (TOTAL_RECORDS // MAX_REQUEST_COUNT)]
        data_by_index[index].extend(result)
    return data_by_index


def build_date_price_map(data_by_index: dict[str, list]) -> dict[str, pl.DataFrame]:
    """Creates a Polars DataFrame for each index keyed by date."""
    dfs = {}
    for index, records in data_by_index.items():
        df = pl.DataFrame(records).rename({"delivery_price": index})
        dfs[index] = df
    return dfs


def find_matching_dates(dfs: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Finds dates where the price ratios are within the given tolerance of the target ratios."""
    # Find common dates across all indexes
    common_dates = set(dfs[INDEX_NAMES[0]]["date"])
    for df in dfs.values():
        common_dates &= set(df["date"])
    if not common_dates:
        return []

    # Filter all DataFrames to only include common dates
    for k in dfs:
        dfs[k] = dfs[k].filter(pl.col("date").is_in(common_dates))

    # Merge all price columns into one DataFrame
    df_combined = dfs[INDEX_NAMES[0]]
    for name in INDEX_NAMES[1:]:
        df_combined = df_combined.with_columns(dfs[name][name])

    # Calculate the price ratios
    df_ratios = df_combined.with_columns([
        (pl.col("btc_usdc") / pl.col("eth_usdc")).alias("btc_eth_ratio"),
        (pl.col("sol_usdc") / pl.col("paxg_usdc")).alias("sol_paxg_ratio"),
        (pl.col("xrp_usdc") / pl.col("ada_usdc")).alias("xrp_ada_ratio"),
    ])

    # Filter rows where all ratios are within tolerance
    condition = (
        ((pl.col("btc_eth_ratio") - TARGET_RATIOS["btc_eth_ratio"]).abs() < TOLERANCE) &
        ((pl.col("sol_paxg_ratio") - TARGET_RATIOS["sol_paxg_ratio"]).abs() < TOLERANCE) &
        ((pl.col("xrp_ada_ratio") - TARGET_RATIOS["xrp_ada_ratio"]).abs() < TOLERANCE)
    )

    return df_ratios.filter(condition)


def compute_coconut_prices(matching_df: pl.DataFrame, label: str) -> pl.DataFrame:
    """Compute coconut price in USD from ETH price on matching dates."""
    coconut_price_eth = CRYPTOS["ETH"]
    if matching_df.is_empty():
        logger.warning(f"No matching ratio-valid dates found for {label}.")
        return None
    return matching_df.with_columns(
        (pl.col("eth_usdc") * coconut_price_eth).alias("coconut_price_usd")
    ).select(["date", "eth_usdc", "coconut_price_usd"])


async def main():
    logger.info("Fetching testnet data...")
    testnet_data = await fetch_all_prices(testnet=True)

    logger.info("Fetching production data...")
    production_data = await fetch_all_prices(testnet=False)

    dfs_testnet = build_date_price_map(testnet_data)
    dfs_production = build_date_price_map(production_data)

    matching_testnet = find_matching_dates(dfs_testnet)
    matching_production = find_matching_dates(dfs_production)

    coconut_testnet = compute_coconut_prices(matching_testnet, "testnet")
    coconut_production = compute_coconut_prices(matching_production, "production")

    if coconut_testnet is None and coconut_production is None:
        logger.error("No matching results found in either environment.")
        return

    if coconut_testnet is not None and coconut_production is not None:
        dates_testnet = set(coconut_testnet["date"].to_list())
        dates_production = set(coconut_production["date"].to_list())
        shared_dates = dates_testnet & dates_production
        if shared_dates:
            logger.success("Matching date(s) found in both environments:")
            shared = coconut_production.filter(pl.col("date").is_in(shared_dates))
            logger.info("\n{}", shared)
            return

    if coconut_testnet is not None:
        logger.success("Coconut price (USD) from testnet data:")
        logger.info("\n{}", coconut_testnet)

    if coconut_production is not None:
        logger.success("Coconut price (USD) from production data:")
        logger.info("\n{}", coconut_production)

# Run the script
asyncio.run(main())
