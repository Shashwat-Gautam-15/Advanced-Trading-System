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

class OrderBookStream:
    def __init__(self, symbol='BTCUSDT', depth='5'):
        self.symbol = symbol
        self.depth = depth
        self.ws_url = f"wss://stream.binance.com:9443/ws/{symbol.lower()}@depth{depth}@100ms"
        self.db_conn = sqlite3.connect('orderbook.db')
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
        async with websockets.connect(self.ws_url) as ws:
            while True:
                try:
                    data = await ws.recv()
                    ob_data = json.loads(data)
                    self._store_snapshot(ob_data)
                except Exception as e:
                    print(f"Error: {e}")
                    continue
                    
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
        self.model = LogisticRegression()
        self.trained = False
        
    def prepare_features(self, df, lookback=10):
        X = []
        y = []
        
        for i in range(lookback, len(df)):
            past_imb = df['imbalance'].iloc[i-lookback:i].values
            past_spread = (df['best_ask'] - df['best_bid']).iloc[i-lookback:i].values
            past_mid = df['mid_price'].iloc[i-lookback:i].values
            
            target = 1 if df['mid_price'].iloc[i] > df['mid_price'].iloc[i-1] else 0
            
            features = np.concatenate([
                past_imb,
                past_spread,
                np.diff(past_mid)
            ])
            
            X.append(features)
            y.append(target)
            
        return np.array(X), np.array(y)
        
    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        self.model.fit(X_train, y_train)
        self.trained = True
        
        preds = self.model.predict(X_test)
        print(classification_report(y_test, preds))
        
    def predict_signal(self, features):
        if not self.trained:
            raise ValueError("Model not trained yet")
            
        pred = self.model.predict([features])[0]
        return "BUY" if pred == 1 else "SELL" if np.random.rand() > 0.5 else "HOLD"

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
            'exit_price': exit_price,
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
        self.execution_engine.start()
        
        async def live_trading():
            async with websockets.connect(self.ob_stream.ws_url) as ws:
                while True:
                    data = await ws.recv()
                    ob_data = json.loads(data)
                    
                    best_bid = float(ob_data['bids'][0][0])
                    best_ask = float(ob_data['asks'][0][0])
                    
                    features = self._prepare_live_features(ob_data)
                    signal = self.signal_generator.predict_signal(features)
                    
                    if signal == "BUY":
                        self.execution_engine.execute_order('B', best_ask, 0.01)
                    elif signal == "SELL":
                        self.execution_engine.execute_order('S', best_bid, 0.01)
                        
        asyncio.get_event_loop().run_until_complete(live_trading())
        
    def _prepare_live_features(self, ob_data):
        bid_volume = sum(float(b[1]) for b in ob_data['bids'])
        ask_volume = sum(float(a[1]) for a in ob_data['asks'])
        imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
        return np.array([imbalance])
        
    def run_backtest(self, start_time, end_time):
        processor = OrderBookProcessor()
        df = processor.to_dataframe(start_time, end_time)
        processed_df = processor.process_orderbook(df)
        
        X, y = self.signal_generator.prepare_features(processed_df)
        self.signal_generator.train(X, y)
        
        self.backtester.run_backtest(processed_df, self.signal_generator)
        return self.backtester.get_performance()
        
    def shutdown(self):
        self.execution_engine.stop()
        self.ob_stream.close()


if __name__ == "__main__":
    print("Starting Trading System...")

    # Initialize system
    ts = TradingSystem()

    async def main():
        print("Collecting initial data...")

        async def collect_data():
            await ts.ob_stream.stream_orderbook()

        try:
            await asyncio.wait_for(collect_data(), timeout=60)
        except asyncio.TimeoutError:
            print("Initial data collection timed out. Proceeding anyway.")

        # Run backtest
        print("Running backtest...")
        performance = ts.run_backtest(
            start_time=int((datetime.now().timestamp() - 3600) * 1000),  # 1 hour ago
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

    asyncio.run(main())
