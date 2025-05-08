import websockets
import asyncio
import json
import sqlite3
from datetime import datetime
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import threading
import order_execution  # The compiled C++ module
# Add this with your other imports at the very top
import logging
from datetime import datetime
from binance.client import Client
import time

api_key = 'your_api_key'
api_secret = 'your_api_secret'
client = Client(api_key, api_secret)

def get_klines(symbol, interval, lookback_minutes):
    raw_data = client.get_historical_klines(symbol, interval, f"{lookback_minutes} min ago UTC")
    df = pd.DataFrame(raw_data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 
        'close_time', 'quote_asset_volume', 'number_of_trades', 
        'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df.astype(float)
    return df

# Configure logging (add this right after imports, before class definitions)
logging.basicConfig(
    filename='trading.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class OrderBookStream:
    def __init__(self, symbol='BTCUSDT', depth='5', test_mode = True): #-----------------------Test_mode = flase is for local testing
        self.symbol = symbol
        self.depth = depth
        self.test_mode = test_mode
        self.ws_url = f"wss://stream.binance.com:9443/ws/{symbol.lower()}@depth{depth}@100ms"
        self.db_conn = sqlite3.connect('orderbook.db', timeout =10)
        self._init_db()
        
    def _init_db(self):
        cursor = self.db_conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS orderbook (
                timestamp INTEGER PRIMARY KEY,
                bids TEXT,
                asks TEXT
            )
        ''')
        self.db_conn.commit()
        
    async def stream_orderbook(self):
        """Binance-compliant WebSocket connection with auto-reconnect"""
        reconnect_delay = 1  # Start with 1-second delay
        max_attempts = 5     # Max reconnection attempts
    
        headers = {
            "User-Agent": "MyTradingBot/1.0 (Educational Project)"
        }

        for attempt in range(max_attempts):
            try:
                async with websockets.connect(
                    self.ws_url,
                    ping_interval=30,
                    ping_timeout=10,
                    extra_headers=headers
                ) as ws:
                    logging.info("Connected to Binance WebSocket")
                    while True:
                        try:
                            data = await asyncio.wait_for(ws.recv(), timeout=60)
                            ob_data = json.loads(data)
                            self._store_snapshot(ob_data)
                            await asyncio.sleep(0.1)  # Rate limiting
                        except asyncio.TimeoutError:
                            await ws.ping()  # Keep connection alive
                        except json.JSONDecodeError:
                            logging.warning("Malformed data received")
                            continue

            except websockets.ConnectionClosed as e:
                logging.warning(f"Disconnected (code {e.code}), retrying in {reconnect_delay}s...")
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, 60)  # Exponential backoff
            except Exception as e:
                logging.error(f"Unexpected error: {str(e)}")
                break

        logging.error("Max reconnection attempts reached")  
                    
    def _store_snapshot(self, data):
        cursor = self.db_conn.cursor()
        timestamp = int(datetime.now().timestamp() * 1000)
        cursor.execute('''
            INSERT INTO orderbook (timestamp, bids, asks)
            VALUES (?, ?, ?)
        ''', (timestamp, json.dumps(data['bids']), json.dumps(data['asks'])))
        self.db_conn.commit()
        
    def close(self):
        self.db_conn.close()

class OrderBookProcessor:
    def __init__(self, db_path='orderbook.db'):
        self.db_conn = sqlite3.connect(db_path)
        
    def to_dataframe(self, start_time=None, end_time=None):
        query = "SELECT timestamp, bids, asks FROM orderbook"
        if start_time and end_time:
            query += f" WHERE timestamp BETWEEN {start_time} AND {end_time}"
        df = pd.read_sql(query, self.db_conn)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
        
    def to_parquet(self, filepath, start_time=None, end_time=None):
        df = self.to_dataframe(start_time, end_time)
        table = pa.Table.from_pandas(df)
        pq.write_table(table, filepath)
        
    def process_orderbook(self, df):
        df['bids'] = df['bids'].apply(json.loads)
        df['asks'] = df['asks'].apply(json.loads)
        
        df['best_bid'] = df['bids'].apply(lambda x: float(x[0][0]))
        df['best_ask'] = df['asks'].apply(lambda x: float(x[0][0]))
        df['mid_price'] = (df['best_bid'] + df['best_ask']) / 2
        
        df['bid_volume'] = df['bids'].apply(lambda x: sum(float(b[1]) for b in x))
        df['ask_volume'] = df['asks'].apply(lambda x: sum(float(a[1]) for a in x))
        df['imbalance'] = (df['bid_volume'] - df['ask_volume']) / (df['bid_volume'] + df['ask_volume'])
        
        return df

class SignalGenerator:
    def __init__(self):
        # Use class_weight to handle imbalance and increase max iterations
        self.model = LogisticRegression(class_weight='balanced',max_iter=1000,random_state=42)
        self.trained = False
        
    def prepare_features(self, df, lookback=10):
        X = []
        y = []
        
        for i in range(lookback, len(df)):
            past_imb = df['imbalance'].iloc[i-lookback:i].values
            past_spread = (df['best_ask'] - df['best_bid']).iloc[i-lookback:i].values
            past_mid = df['mid_price'].iloc[i-lookback:i].values
            # New feautures
            price_change = df['mid_price'].iloc[i] - df['mid_price'].iloc[i-1]
            rolling_vol = df['mid_price'].iloc[i-lookback:i].std()

            #target = 1 if df['mid_price'].iloc[i] > df['mid_price'].iloc[i-1] else 0
            target = 1 if price_change > rolling_vol*0.2 else 0 # only predicit if significant move
            
            features = np.concatenate([
                past_imb,
                past_spread,
                np.diff(past_mid),
                [rolling_vol]  # Add volatility
            ])
            
            X.append(features)
            y.append(target)
            
        return np.array(X), np.array(y)
        
    # def train(self, X, y):
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    #     self.model.fit(X_train, y_train)
    #     self.trained = True
        
    #     preds = self.model.predict(X_test)
    #     print(classification_report(y_test, preds))
    def train(self, X, y):
    # Check if we have both classes
        if len(np.unique(y)) < 2:
            raise ValueError("Need samples from both classes to train")

    # Use stratified split to maintain class balance
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.2, 
            stratify=y,
            random_state=42
        )
        self.model.fit(X_train, y_train)
        self.trained = True
    
        # Keep the evaluation------------------------------ISSUE IS HERE
        preds = self.model.predict(X_test)
        print(classification_report(y_test, preds))
        
    def predict_signal(self, features):
        if not self.trained:
            raise ValueError("Model not trained yet")
            
        proba = self.model.predict_proba([features])[0]
        if max(proba) < 0.6 :# only trade if confident
            return "HOLD"
        return "BUY" if np.argmax(proba) == 1 else "SELL"

class ExecutionEngine:
    def __init__(self):
        self.engine = order_execution.OrderExecutionEngine()
        self.execution_thread = None
        
    def start(self):
        self.execution_thread = threading.Thread(target=self.engine.start)
        self.execution_thread.daemon = True
        self.execution_thread.start()
        
    def stop(self):
        self.engine.stop()
        if self.execution_thread:
            self.execution_thread.join()
            
    def execute_order(self, side, price, quantity):
        self.engine.add_order(side, price, quantity)
        logging.info(f"Queued {side} order: {quantity} @ {price:.2f}")
        
    def queue_size(self):
        return self.engine.queue_size()

class Backtester:
    def __init__(self):
        self.positions = []
        self.pnl = []
        self.current_position = None
        
    def run_backtest(self, df, signal_generator, lookback=10):
        for i in range(lookback, len(df)):
            features, _ = signal_generator.prepare_features(df.iloc[:i], lookback)
            if len(features) == 0:
                continue
                
            signal = signal_generator.predict_signal(features[-1])
            self.execute_trade(signal, df.iloc[i])
            
    def execute_trade(self, signal, data):
        if signal == "BUY" and (self.current_position is None or self.current_position['side'] == 'SELL'):
            self.close_position(data)
            self.current_position = {
                'side': 'BUY',
                'entry_price': data['best_ask'],
                'entry_time': data['timestamp'],
                'quantity': 1
            }
        elif signal == "SELL" and (self.current_position is None or self.current_position['side'] == 'BUY'):
            self.close_position(data)
            self.current_position = {
                'side': 'SELL',
                'entry_price': data['best_bid'],
                'entry_time': data['timestamp'],
                'quantity': 1
            }
            
    def close_position(self, data):
        if self.current_position is None:
            return
            
        exit_price = data['best_bid'] if self.current_position['side'] == 'BUY' else data['best_ask']
        pnl = (exit_price - self.current_position['entry_price']) * self.current_position['quantity']
        if self.current_position['side'] == 'SELL':
            pnl = -pnl
            
        trade = {
            'entry_time': self.current_position['entry_time'],
            'exit_time': data['timestamp'],
            'side': self.current_position['side'],
            'entry_price': self.current_position['entry_price'],
            'exit_price':data['best_bid'] if self.current_position['side'] == 'BUY' else data['best_ask'],

            'pnl': pnl
        }
        
        self.positions.append(trade)
        self.pnl.append(pnl)
        self.current_position = None
        
    def get_performance(self):
        if not self.pnl:
            return {}
            
        total_pnl = sum(self.pnl)
        sharpe_ratio = np.mean(self.pnl) / np.std(self.pnl) * np.sqrt(252*24)
        max_drawdown = min(np.minimum.accumulate(self.pnl) - np.maximum.accumulate(self.pnl))
        
        return {
            'total_pnl': total_pnl,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': len([p for p in self.pnl if p > 0]) / len(self.pnl) if len(self.pnl) > 0 else 0
        }

class TradingSystem:
    def __init__(self):
        self.ob_stream = OrderBookStream()
        self.execution_engine = ExecutionEngine()
        self.signal_generator = SignalGenerator()
        self.backtester = Backtester()
        
    def run_live(self):
    #""Run the live trading system with WebSocket connection and order execution
    # Start the execution engine
        self.execution_engine.start()
        logging.info("Execution engine started")

        async def live_trading():
            reconnect_delay = 5  # seconds to wait before reconnecting
            max_reconnect_attempts = 5
            attempt = 0
        
            while attempt < max_reconnect_attempts:
                try:
                    logging.info(f"Connecting to WebSocket at {self.ob_stream.ws_url}")
                    async with websockets.connect(
                        self.ob_stream.ws_url,
                        ping_interval=30,  # Send ping every 30 seconds
                        ping_timeout=10,   # Wait 10 seconds for pong response
                        close_timeout=1    # Wait 1 second when closing
                    ) as ws:
                        attempt = 0  # Reset attempt counter on successful connection
                        logging.info("WebSocket connection established")
                    
                        while True:
                            try:
                            # Receive and process order book updates
                                data = await asyncio.wait_for(ws.recv(), timeout=60)
                                ob_data = json.loads(data)
                            
                            # Extract market data
                                best_bid = float(ob_data['bids'][0][0])
                                best_ask = float(ob_data['asks'][0][0])
                                spread = best_ask - best_bid
                                mid_price = (best_bid + best_ask) / 2
                            
                            # Generate trading signal
                                features = self._prepare_live_features(ob_data)
                                signal = self.signal_generator.predict_signal(features)
                            
                            # Log market data and signal
                                logging.info(
                                    f"Market Update | Bid: {best_bid:.2f} | Ask: {best_ask:.2f} | "
                                    f"Spread: {spread:.4f} | Mid: {mid_price:.2f} | Signal: {signal}")
                            
                            
                            # Execute trades based on signal
                                if signal == "BUY":
                                    quantity = 0.01  # Fixed quantity for demo
                                    self.execution_engine.execute_order('B', best_ask, quantity)
                                    logging.info(f"Executed BUY order: {quantity} @ {best_ask:.2f}")
                                elif signal == "SELL":
                                    quantity = 0.01  # Fixed quantity for demo
                                    self.execution_engine.execute_order('S', best_bid, quantity)
                                    logging.info(f"Executed SELL order: {quantity} @ {best_bid:.2f}")
                            
                            # Small delay to prevent overwhelming the exchange
                                await asyncio.sleep(0.1)
                            
                            except asyncio.TimeoutError:
                                logging.warning("No data received for 60 seconds, sending ping")
                                await ws.ping()
                                continue
                            except json.JSONDecodeError:
                                logging.error("Failed to parse WebSocket message")
                                continue
                            except Exception as e:
                                logging.error(f"Error processing message: {str(e)}")
                                continue
                            
                except websockets.exceptions.ConnectionClosed:
                    attempt += 1
                    logging.warning(
                        f"Connection closed unexpectedly. Reconnecting attempt {attempt}/"
                        f"{max_reconnect_attempts} in {reconnect_delay} seconds...")
                    await asyncio.sleep(reconnect_delay)
                except Exception as e:
                    logging.critical(f"Fatal error in live trading: {str(e)}")
                    raise
        
            logging.error(f"Failed to reconnect after {max_reconnect_attempts} attempts")
            raise ConnectionError("Maximum reconnection attempts reached")

        # Start the live trading loop
        try:
            asyncio.run(live_trading())
        except KeyboardInterrupt:
            logging.info("Received keyboard interrupt, shutting down...")
        except Exception as e:
            logging.critical(f"Critical error in live trading: {str(e)}")
            raise
        finally:
            self.shutdown()
        
    def _prepare_live_features(self, ob_data):
        bids = np.array([[float(b[0]), float(b[1])] for b in ob_data['bids']])
        asks = np.array([[float(a[0]), float(a[1])] for a in ob_data['asks']])
    
        bid_volume = bids[:,1].sum()
        ask_volume = asks[:,1].sum()
        imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
    
        # Add more features
        spread = asks[0,0] - bids[0,0]
        mid_price = (asks[0,0] + bids[0,0]) / 2
    
        return np.array([imbalance, spread, mid_price])
        
    def run_backtest(self, start_time=None, end_time=None):
        symbol = 'BTCUSDT'
        interval = '1m'
        lookback_minutes = 120  # You can increase this based on your indicators

        df = get_klines(symbol, interval, lookback_minutes)

        if df.empty or len(df) < 100:
            raise ValueError(f"Insufficient data : Only {len(df)} records found")

        print(f"Price stats:\n{df['Close'].describe()}")

        X, y = self.signal_generator.prepare_features(df)
        print(f"Class balance: {pd.Series(y).value_counts().to_dict()}")

        self.signal_generator.train(X, y)
        self.backtester.run_backtest(df, self.signal_generator)
        return self.backtester.get_performance()

        
    def shutdown(self):
        self.execution_engine.stop()
        self.ob_stream.close()

if __name__ == "__main__":
    print("Starting Trading System...")
    
    # Initialize system
    ts = TradingSystem()
    
    # First collect some data (run for 1 minute)
    print("Collecting initial data...")
    async def collect_data():
        await ts.ob_stream.stream_orderbook()

    # try:
    #     asyncio.get_event_loop().run_until_complete(asyncio.wait_for(collect_data(), timeout=60))
    # except asyncio.TimeoutError:
    #     pass
    async def main():
        try:
            await asyncio.wait_for(collect_data(), timeout=900)
        except asyncio.TimeoutError:
            logging.infp("Completed 15-minute data collection")
            pass
    asyncio.run(main())
    # Run backtest
    print("Running backtest...")
    performance = ts.run_backtest(
        start_time=int((datetime.now().timestamp() - 86400) * 1000),  # 1 hour ago
        end_time=int(datetime.now().timestamp() * 1000)
    )
    print("Backtest Results:", performance)
    
    # Run live trading
    print("Starting live trading...")
    try:
        ts.run_live()
    except KeyboardInterrupt:
        ts.shutdown()
        print("System shutdown complete")