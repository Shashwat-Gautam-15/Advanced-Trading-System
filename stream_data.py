from core.streamer import OrderBookStream
import asyncio

ob = OrderBookStream()
asyncio.run(ob.stream_orderbook())
