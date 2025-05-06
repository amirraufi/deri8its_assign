import asyncio, json, time, numpy as np, requests, websockets, pandas as pd
from typing import Dict, List, Any

# input parameters
EXPIRY   = "23MAY25"   # expiry code to filter
DEPTH    = 20          # price levels per side to keep
T1_SEC   = 60          # total runing time
T2_SEC   = 5           # interval between snapshots

REST_URL = "https://www.deribit.com/api/v2"
WS_URL   = "wss://www.deribit.com/ws/api/v2"

# finding the coins with option contracts
def discover_names(expiry: str) -> List[str]:
    coins = [c["currency"] for c in
             requests.get(f"{REST_URL}/public/get_currencies",
                          timeout=10).json()["result"]]
    names: List[str] = []
    for coin in coins:
        r = requests.get(f"{REST_URL}/public/get_instruments",
                         params={"currency": coin, "kind": "option",
                                 "expired": "false"},
                         timeout=10).json()["result"]
        names += [i["instrument_name"] for i in r
                  if expiry.upper() in i["instrument_name"]]
    return names

# Deribit WebSocket stream
class DeribitStream:
    def __init__(self) -> None:
        self.ws: websockets.WebSocketClientProtocol | None = None
        self.pending: Dict[int, asyncio.Future] = {}
        self.ticker: Dict[str, dict] = {}
        self.book:   Dict[str, dict] = {}  # {ins:{bids{}, asks{}}}

    # connect
    async def connect(self) -> None:
        self.ws = await websockets.connect(WS_URL,
                                           ping_interval=None,
                                           ping_timeout=None)
        asyncio.create_task(self._reader())

    async def _rpc(self, method: str, params: dict) -> Any:
        rid = int(time.time()*1000) + np.random.randint(99_999)
        fut = asyncio.get_running_loop().create_future()
        self.pending[rid] = fut
        await self.ws.send(json.dumps(
            {"jsonrpc":"2.0","id":rid,"method":method,"params":params}))
        return await fut

    async def subscribe(self, names: List[str], depth: int) -> None:
        chans = ([f"ticker.{n}.100ms"            for n in names] +
                 [f"book.{n}.none.{depth}.100ms" for n in names])
        await self._rpc("public/subscribe", {"channels": chans})
        try:
            await self._rpc("public/set_heartbeat", {"interval": 15})
        except Exception:
            pass

    # helper
    def ticker_df(self) -> pd.DataFrame:
        return (pd.DataFrame.from_dict(self.ticker, orient="index")
                  .reset_index(names="instrument"))
    def book_df(self) -> pd.DataFrame:
        rows = []
        for ins, lad in self.book.items():
            for side in ("bids", "asks"):
                for px, sz in lad[side].items():
                    rows.append({"instrument": ins,
                                 "side": side,
                                 "price": px,
                                 "size": sz})
        return pd.DataFrame(rows)




    async def _reader(self) -> None:
        async for raw in self.ws:
            msg = json.loads(raw)

            if "id" in msg and msg["id"] in self.pending:
                fut = self.pending.pop(msg["id"])
                fut.set_result(msg.get("result", msg.get("error")))
                continue
            # if msg.get("method") == "heartbeat":
            #         if msg["params"].get("type") == "test_request":
            #     # respond immediately – no need to await the result
            #              await self.ws.send(json.dumps({
            #         "jsonrpc": "2.0",
            #         "id":      msg.get("id", 0),   # any int is fine
            #         "method":  "public/test"
            #     }))
            #         continue      
            if msg.get("method") != "subscription":
                continue

            ch   = msg["params"]["channel"]
            data = msg["params"]["data"]
            ins  = data["instrument_name"]

            if ch.startswith("ticker."):     
                self.ticker[ins] = {
                    "ts":       data["timestamp"],
                    "best_bid": data["best_bid_price"],
                    "bid_sz":   data["best_bid_amount"],
                    "best_ask": data["best_ask_price"],
                    "ask_sz":   data["best_ask_amount"],
                    "bid_iv":   data["bid_iv"],
                    "ask_iv":   data["ask_iv"],
                    "mark_px":  data["mark_price"],
                    "mark_iv":  data["mark_iv"],
                    "index_price": data["index_price"], 
                }
            else:                              
                book = self.book.setdefault(ins, {"bids": {}, "asks": {}})
                for side in ("bids", "asks"):
                    for row in data[side]:
                        if len(row) == 3:
                            action, px, sz = row
                        else:
                            px, sz = row
                            action = "snapshot"

                        if action in ("delete", "remove"):
                            book[side].pop(px, None)
                        else:
                            book[side][px] = sz


async def main() -> None:
    names = discover_names(EXPIRY)
    if not names:
        raise RuntimeError(f"No live options for {EXPIRY}")

    stream = DeribitStream()
    await stream.connect()
    await stream.subscribe(names, DEPTH)

    print(f"Streaming {len(names)} contracts — snapshots every {T2_SEC}s")
    end = time.time() + T1_SEC
    snap = 0
    while time.time() < end:
        await asyncio.sleep(T2_SEC)
        snap += 1

        df_book  = stream.book_df()
        df_tick  = stream.ticker_df()
        # merge book and ticker data
        merged = df_book.merge(df_tick, on="instrument", how="left")

        ts = time.strftime("%H:%M:%S")
        print(f"\nSnapshot {snap} @ {ts} "
              f"({len(df_tick)} instruments, {len(df_book)} ladder rows)")
        print(merged.head(50).to_string(index=False))

        # save to CSV
        #merged.to_csv(f"deribit_{EXPIRY}_{snap:03d}.csv",
                     # index=False, mode="a", header=snap==1)

    await stream.ws.close()
    print("\nFinished.")
#name: __main__
if __name__ == "__main__":
    asyncio.run(main())