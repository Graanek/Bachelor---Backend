from datetime import datetime, timedelta
from typing import List, Tuple, Optional
from binance.client import Client
from binance.exceptions import BinanceAPIException

# Importuj funkcje z app.database.operations
from app.database.operations import get_price_history, insert_price_data, clean_old_data
from app.config import DEFAULT_ASSETS # Upewnij się, że importujesz DEFAULT_ASSETS z config.py

def download_binance_data(symbol: str, interval: str, start_time: datetime = None, end_time: datetime = None):
        # Initialize Binance client
        client = Client()
        
        # Set default time range if not provided
        if not start_time:
            start_time = datetime.now() - timedelta(days=30)
        if not end_time:
            end_time = datetime.now()
        
        # Ensure end_time is not in the future
        current_time = datetime.now()
        if end_time > current_time:
            end_time = current_time
        
        # Map interval to Binance interval
        interval_map = {
            '5min': Client.KLINE_INTERVAL_5MINUTE,
            '15min': Client.KLINE_INTERVAL_15MINUTE,
            '1h': Client.KLINE_INTERVAL_1HOUR,
            '1d': Client.KLINE_INTERVAL_1DAY
        }
        
        binance_interval = interval_map.get(interval)
        if not binance_interval:
            raise ValueError(f"Invalid interval: {interval}")
        
        # Convert symbol to Binance format
        binance_symbol = f"{symbol}USDT"
         
        # Download klines (candlestick data)
        klines = client.get_historical_klines(
            symbol=binance_symbol,
            interval=binance_interval,
            start_str=int(start_time.timestamp() * 1000),
            end_str=int(end_time.timestamp() * 1000)
        )
        
        if not klines:
            return []
        
        # Process and return data
        return [(datetime.fromtimestamp(kline[0] / 1000), float(kline[4]), float(kline[5])) for kline in klines]

    
def download_all_data():
    """Download data for all symbols and intervals, but only if data is missing or old. Only new records are downloaded and inserted."""
    symbols = ["BTC", "ETH", "SOL"]
    intervals = ["5min", "15min", "1h", "1d"]
    interval_deltas = {
        "5min": timedelta(minutes=5),
        "15min": timedelta(minutes=15),
        "1h": timedelta(hours=1),
        "1d": timedelta(days=1),
    }
    for symbol in symbols:
        for interval in intervals:
            print(f"\nProcessing {interval} data for {symbol}...")
            clean_old_data(interval)
            try:
                history = get_price_history(symbol, interval, limit=1)
                current_time = datetime.now()
                should_download = False
                latest_timestamp = None
                if not history:
                    print(f"No data found for {symbol} at {interval} interval, downloading...")
                    should_download = True
                else:
                    try:
                        latest_timestamp = datetime.strptime(history[0][0], '%Y-%m-%d %H:%M:%S')
                        print(f"Latest timestamp in DB for {symbol} {interval}: {latest_timestamp}")
                        if interval == "5min" and (current_time - latest_timestamp).total_seconds() > 300:
                            should_download = True
                        elif interval == "15min" and (current_time - latest_timestamp).total_seconds() > 900:
                            should_download = True
                        elif interval == "1h" and (current_time - latest_timestamp).total_seconds() > 3600:
                            should_download = True
                        elif interval == "1d" and (current_time - latest_timestamp).total_seconds() > 86400:
                            should_download = True
                    except (ValueError, IndexError) as e:
                        print(f"Error parsing timestamp: {e}")
                        should_download = True
                if should_download:
                    print(f"Downloading {interval} data for {symbol}...")
                    end_time = current_time
                    if latest_timestamp:
                        start_time = latest_timestamp + interval_deltas[interval]
                        print(f"Setting start_time for Binance download: {start_time}")
                    else:
                        if interval == "5min":
                            start_time = end_time - timedelta(days=1)
                        elif interval == "15min":
                            start_time = end_time - timedelta(days=7)
                        elif interval == "1h":
                            start_time = end_time - timedelta(days=30)
                        else:  # 1d - get all historical data
                            start_time = datetime(2017, 1, 1)
                        print(f"No latest timestamp, using default start_time: {start_time}")
                    data = download_binance_data(symbol, interval, start_time=start_time, end_time=end_time)
                    if data:
                        print(f"Downloaded {len(data)} records for {symbol} at {interval} interval")
                        for timestamp, price, volume in data:
                            if latest_timestamp and timestamp <= latest_timestamp:
                                continue
                            insert_price_data(symbol, price, timestamp, interval, volume)
                        clean_old_data(interval)
                    else:
                        print(f"No data downloaded for {symbol} at {interval} interval")
                else:
                    print(f"Data for {symbol} at {interval} interval is up to date")
            except Exception as e:
                print(f"Error checking/downloading {interval} data for {symbol}: {e}")
