# main.py
import io
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import sqlite3
from tradingview_ta import TA_Handler, Interval
from binance.client import Client
from binance.exceptions import BinanceAPIException

from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Konfiguracja CORS – pozwalamy na połączenia z dowolnego źródła
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Domyślna lista 10 najpopularniejszych kryptowalut (bez stablecoinów)
DEFAULT_ASSETS = ["BTC", "ETH", "XRP", "SOL"]

# Database initialization
def init_db():
    conn = sqlite3.connect('crypto_prices.db')
    cursor = conn.cursor()
    
    # Create symbols table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS symbols (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL UNIQUE,
        name TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Create tables for different time intervals
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS crypto_history_5min (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol_id INTEGER NOT NULL,
        timestamp DATETIME NOT NULL,
        price REAL NOT NULL,
        volume REAL,
        FOREIGN KEY (symbol_id) REFERENCES symbols(id),
        UNIQUE(symbol_id, timestamp)
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS crypto_history_15min (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol_id INTEGER NOT NULL,
        timestamp DATETIME NOT NULL,
        price REAL NOT NULL,
        volume REAL,
        FOREIGN KEY (symbol_id) REFERENCES symbols(id),
        UNIQUE(symbol_id, timestamp)
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS crypto_history_1h (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol_id INTEGER NOT NULL,
        timestamp DATETIME NOT NULL,
        price REAL NOT NULL,
        volume REAL,
        FOREIGN KEY (symbol_id) REFERENCES symbols(id),
        UNIQUE(symbol_id, timestamp)
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS crypto_history_1d (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol_id INTEGER NOT NULL,
        timestamp DATETIME NOT NULL,
        price REAL NOT NULL,
        volume REAL,
        FOREIGN KEY (symbol_id) REFERENCES symbols(id),
        UNIQUE(symbol_id, timestamp)
    )
    ''')
    
    conn.commit()
    conn.close()

def get_or_create_symbol(symbol: str) -> int:
    """Get symbol ID or create new symbol if it doesn't exist"""
    conn = sqlite3.connect('crypto_prices.db')
    cursor = conn.cursor()
    
    try:
        # Try to get existing symbol
        cursor.execute('SELECT id FROM symbols WHERE symbol = ?', (symbol,))
        result = cursor.fetchone()
        
        if result:
            return result[0]
        
        # Create new symbol if it doesn't exist
        cursor.execute('INSERT INTO symbols (symbol) VALUES (?)', (symbol,))
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()

def insert_price_data(symbol: str, price: float, timestamp: datetime, interval: str, volume: float = None):
    """Insert price data into specified interval table"""
    conn = sqlite3.connect('crypto_prices.db')
    cursor = conn.cursor()
    
    try:
        # Get or create symbol ID
        symbol_id = get_or_create_symbol(symbol)
        
        # Map interval to table name
        table_map = {
            '5min': 'crypto_history_5min',
            '15min': 'crypto_history_15min',
            '1h': 'crypto_history_1h',
            '1d': 'crypto_history_1d'
        }
        
        table_name = table_map.get(interval)
        if not table_name:
            raise ValueError(f"Invalid interval: {interval}")
        
        cursor.execute(f'''
        INSERT OR REPLACE INTO {table_name}
        (symbol_id, timestamp, price, volume)
        VALUES (?, ?, ?, ?)
        ''', (symbol_id, timestamp.strftime('%Y-%m-%d %H:%M:%S'), price, volume))
        
        conn.commit()
    except Exception as e:
        print(f"Error inserting data: {e}")
    finally:
        conn.close()

def get_price_history(symbol: str, interval: str, limit: int = None):
    """Get price history from specified interval table"""
    conn = sqlite3.connect('crypto_prices.db')
    cursor = conn.cursor()
    
    try:
        # Get symbol ID
        cursor.execute('SELECT id FROM symbols WHERE symbol = ?', (symbol,))
        result = cursor.fetchone()
        if not result:
            print(f"Symbol {symbol} not found in database")
            return []
        
        symbol_id = result[0]
        
        # Map interval to table name
        table_map = {
            '5min': 'crypto_history_5min',
            '15min': 'crypto_history_15min',
            '1h': 'crypto_history_1h',
            '1d': 'crypto_history_1d'
        }
        
        table_name = table_map.get(interval)
        if not table_name:
            raise ValueError(f"Invalid interval: {interval}")
        
        query = f'''
        SELECT timestamp, price, volume
        FROM {table_name}
        WHERE symbol_id = ?
        ORDER BY timestamp DESC
        '''
        
        if limit:
            query += f' LIMIT {limit}'
        
        cursor.execute(query, (symbol_id,))
        results = cursor.fetchall()
        
        if not results:
            print(f"No data found for {symbol} in {table_name}")
            return []
            
        return results
    except Exception as e:
        print(f"Error getting price history: {e}")
        return []
    finally:
        conn.close()

def cleanup_old_data():
    """Clean up old data from the tables"""
    conn = sqlite3.connect('crypto_prices.db')
    cursor = conn.cursor()
    
    now = datetime.now()
    
    # Clean 5min data older than 24 hours
    cursor.execute('''
    DELETE FROM crypto_history_5min 
    WHERE timestamp < ?
    ''', (now - timedelta(days=1),))
    
    # Clean 15min data older than 7 days
    cursor.execute('''
    DELETE FROM crypto_history_15min 
    WHERE timestamp < ?
    ''', (now - timedelta(days=7),))
    
    # Clean 1h data older than 30 days
    cursor.execute('''
    DELETE FROM crypto_history_1h 
    WHERE timestamp < ?
    ''', (now - timedelta(days=30),))
    
    conn.commit()
    conn.close()

# Initialize database on startup
init_db()

def map_ticker(asset: str) -> str:
    """Mapuje symbol kryptowaluty na ticker Yahoo Finance (np. BTC -> BTC-USD)."""
    return f"{asset}-USD"

@app.get("/portfolio")
async def portfolio(
    start_date: str = Query(..., description="Data początkowa w formacie RRRR-MM-DD"),
    end_date: str = Query(..., description="Data końcowa w formacie RRRR-MM-DD"),
    assets: str = Query(
        None,
        description="Lista aktywów oddzielona przecinkami (np. BTC,ETH,...). Jeśli puste, używamy domyślnej listy 10 kryptowalut."
    )
):
    try:
        # Parse dates
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        current_time = datetime.now()
        
        # Validate dates
        if start > current_time or end > current_time:
            raise HTTPException(
                status_code=400, 
                detail="Future dates are not supported. Please use dates up to today."
            )
        
        if start > end:
            raise HTTPException(
                status_code=400,
                detail="Start date must be before end date"
            )
        
        # Get asset list
        if assets is None or assets.strip() == "":
            asset_list = DEFAULT_ASSETS
        else:
            asset_list = [asset.strip().upper() for asset in assets.split(",") if asset.strip() != ""]
        
        # Get data from database for each asset
        data = {}
        for asset in asset_list:
            # Get daily data for the date range
            history = get_price_history(asset, "1d")
            
            if not history:
                # If no data in database, try to fetch from Binance
                try:
                    # Set end_time to current time to avoid future dates
                    end_time = min(end, current_time)
                    binance_data = download_binance_data(asset, "1d", start_time=start, end_time=end_time)
                    if binance_data:
                        for timestamp, price, volume in binance_data:
                            insert_price_data(asset, price, timestamp, "1d", volume)
                        history = get_price_history(asset, "1d")
                except Exception as e:
                    print(f"Error fetching data for {asset}: {e}")
            
            if history:
                # Convert database data to pandas Series
                dates = [datetime.strptime(entry[0], '%Y-%m-%d %H:%M:%S') for entry in history]
                prices = [entry[1] for entry in history]
                data[asset] = pd.Series(prices, index=dates)
        
        if not data:
            raise HTTPException(
                status_code=404, 
                detail="No data available for the specified date range"
            )
        
        # Create DataFrame from the data
        df = pd.DataFrame(data)
        df = df.sort_index()
        
        # Filter for the requested date range
        mask = (df.index >= start) & (df.index <= end)
        df = df.loc[mask]
        
        if df.empty:
            raise HTTPException(
                status_code=404, 
                detail="No data available for the specified date range"
            )
        
        # Calculate returns
        returns = df.pct_change().dropna()
        if returns.empty:
            raise HTTPException(
                status_code=404, 
                detail="Not enough data points to calculate returns"
            )
        
        # Calculate portfolio statistics
        mean_returns = returns.mean()
        cov_matrix = returns.cov()
        
        # Generate random portfolios
        num_portfolios = 1000
        num_assets = len(asset_list)
        random_weights = np.random.random((num_portfolios, num_assets))
        random_weights = random_weights / random_weights.sum(axis=1, keepdims=True)
        
        # Calculate portfolio returns and volatility
        port_returns = (random_weights @ mean_returns.values) * 252
        port_vols = np.sqrt(np.sum(random_weights * (random_weights @ cov_matrix), axis=1) * 252)
        sharpes = np.divide(port_returns, port_vols, out=np.zeros_like(port_returns), where=port_vols != 0)
        
        # Find optimal portfolio
        best_idx = np.argmax(sharpes)
        best_weights = random_weights[best_idx]
        best_allocation = {asset: round(weight * 100, 2) for asset, weight in zip(asset_list, best_weights)}
        
        # Calculate efficient frontier
        sorted_indices = np.argsort(port_vols)
        sorted_vols = port_vols[sorted_indices]
        sorted_rets = port_returns[sorted_indices]
        efficient_vols = []
        efficient_rets = []
        current_max_ret = -np.inf
        for vol, ret in zip(sorted_vols, sorted_rets):
            if ret > current_max_ret:
                efficient_vols.append(vol)
                efficient_rets.append(ret)
                current_max_ret = ret
        
        # Sample simulations for response
        scatter_sample_size = 2000
        if num_portfolios > scatter_sample_size:
            indices = np.random.choice(num_portfolios, scatter_sample_size, replace=False)
            sim_vols_sample = port_vols[indices]
            sim_rets_sample = port_returns[indices]
            sim_sharpes_sample = sharpes[indices]
        else:
            sim_vols_sample = port_vols
            sim_rets_sample = port_returns
            sim_sharpes_sample = sharpes
        
        response_data = {
            "simulations": {
                "vols": sim_vols_sample.tolist(),
                "rets": sim_rets_sample.tolist(),
                "sharpes": sim_sharpes_sample.tolist()
            },
            "efficient_frontier": {
                "vols": efficient_vols,
                "rets": efficient_rets
            },
            "best_allocation": best_allocation,
            "num_simulations": num_portfolios,
            "assets": asset_list
        }
        
        return JSONResponse(content=response_data)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating portfolio: {str(e)}")

def download_binance_data(symbol: str, interval: str, start_time: datetime = None, end_time: datetime = None):
    """Download historical data from Binance"""
    try:
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
        
        print(f"Fetching data for {binance_symbol} from {start_time} to {end_time}")
        
        # Download klines (candlestick data)
        klines = client.get_historical_klines(
            symbol=binance_symbol,
            interval=binance_interval,
            start_str=int(start_time.timestamp() * 1000),
            end_str=int(end_time.timestamp() * 1000)
        )
        
        if not klines:
            print(f"No data returned from Binance for {binance_symbol}")
            return []
        
        # Process and return data
        data = []
        for kline in klines:
            try:
                timestamp = datetime.fromtimestamp(kline[0] / 1000)
                close_price = float(kline[4])  # Close price
                volume = float(kline[5])       # Volume
                data.append((timestamp, close_price, volume))
            except (ValueError, IndexError) as e:
                print(f"Error processing kline data: {e}")
                continue
        
        print(f"Downloaded {len(data)} records for {symbol} from Binance")
        return data
        
    except BinanceAPIException as e:
        print(f"Binance API Error: {e}")
        return []
    except Exception as e:
        print(f"Error downloading from Binance: {e}")
        return []

def update_database_from_binance(symbol: str, interval: str):
    """Update database with data from Binance"""
    try:
        # Get the latest timestamp from database
        history = get_price_history(symbol, interval, limit=1)
        start_time = None
        
        if history:
            try:
                start_time = datetime.strptime(history[0][0], '%Y-%m-%d %H:%M:%S')
                print(f"Found existing data up to {start_time}")
            except (ValueError, IndexError) as e:
                print(f"Error parsing timestamp: {e}")
        
        # Download data from Binance
        data = download_binance_data(symbol, interval, start_time=start_time)
        
        if not data:
            print(f"No new data downloaded for {symbol}")
            return
        
        print(f"Inserting {len(data)} records into database for {symbol}")
        
        # Insert data into database
        for timestamp, price, volume in data:
            insert_price_data(
                symbol=symbol,
                price=price,
                timestamp=timestamp,
                interval=interval,
                volume=volume
            )
        
        # Clean up old data
        cleanup_old_data()
        
    except Exception as e:
        print(f"Error updating database from Binance: {e}")

def clean_old_data(interval: str):
    """Clean up old data based on the interval"""
    conn = sqlite3.connect('crypto_prices.db')
    cursor = conn.cursor()
    
    current_time = datetime.now()
    
    # Set the cutoff time based on interval
    if interval == "5min":
        cutoff_time = current_time - timedelta(days=1)  # Keep last 24 hours
    elif interval == "15min":
        cutoff_time = current_time - timedelta(days=7)  # Keep last 7 days
    elif interval == "1h":
        cutoff_time = current_time - timedelta(days=30)  # Keep last 30 days
    else:  # 1d - keep all historical data
        return
    
    # Convert cutoff time to string format
    cutoff_time_str = cutoff_time.strftime('%Y-%m-%d %H:%M:%S')
    
    # First, count how many records will be deleted
    cursor.execute(f'''
        SELECT COUNT(*) FROM crypto_history_{interval}
        WHERE timestamp < ?
    ''', (cutoff_time_str,))
    count = cursor.fetchone()[0]
    
    if count > 0:
        print(f"Cleaning up {count} old records from crypto_history_{interval} table")
        
        # Delete old data
        cursor.execute(f'''
            DELETE FROM crypto_history_{interval}
            WHERE timestamp < ?
        ''', (cutoff_time_str,))
        
        conn.commit()
        print(f"Successfully removed {count} old records from crypto_history_{interval} table")
    
    conn.close()

@app.get("/price-data")
async def price_data(
    asset: str = Query(..., description="Symbol aktywa, np. BTC, ETH"),
    timeframe: str = Query(..., description="Przedział czasowy: 1d, 1w, 1m, all")
):
    try:
        # Map timeframe to interval and limit
        interval_map = {
            "1d": ("5min", 288),    # 5min intervals, 288 entries for 24h
            "1w": ("15min", 672),   # 15min intervals for 1 week
            "1m": ("1h", 720),      # 1h intervals for 1 month
            "all": ("1d", None)     # Daily data for all history
        }
        
        if timeframe not in interval_map:
            raise HTTPException(status_code=400, detail="Invalid timeframe")
            
        interval, limit = interval_map[timeframe]
        
        # Clean up old data for this interval
        clean_old_data(interval)
        
        # Always try to get data from database first
        history = get_price_history(asset, interval, limit)
        current_time = datetime.now()
        
        # Check if we need to fetch from API
        should_fetch_from_api = False
        
        if not history:
            should_fetch_from_api = True
        else:
            try:
                latest_timestamp = datetime.strptime(history[0][0], '%Y-%m-%d %H:%M:%S')
                # Check if data is too old based on interval
                if interval == "5min" and (current_time - latest_timestamp).total_seconds() > 300:  # 5 minutes
                    should_fetch_from_api = True
                elif interval == "15min" and (current_time - latest_timestamp).total_seconds() > 900:  # 15 minutes
                    should_fetch_from_api = True
                elif interval == "1h" and (current_time - latest_timestamp).total_seconds() > 3600:  # 1 hour
                    should_fetch_from_api = True
                elif interval == "1d" and (current_time - latest_timestamp).total_seconds() > 86400:  # 1 day
                    should_fetch_from_api = True
            except (ValueError, IndexError) as e:
                print(f"Error parsing timestamp: {e}")
                should_fetch_from_api = True
        
        # If we need to fetch from API, update the database
        if should_fetch_from_api:
            try:
                # Set appropriate time range based on interval
                end_time = current_time
                if interval == "5min":
                    start_time = end_time - timedelta(days=1)
                elif interval == "15min":
                    start_time = end_time - timedelta(days=7)
                elif interval == "1h":
                    start_time = end_time - timedelta(days=30)
                else:  # 1d - get all historical data
                    start_time = datetime(2013, 1, 1)
                
                # Download and store data
                data = download_binance_data(asset, interval, start_time=start_time, end_time=end_time)
                if data:
                    print(f"Downloaded {len(data)} records for {asset} at {interval} interval")
                    for timestamp, price, volume in data:
                        insert_price_data(asset, price, timestamp, interval, volume)
                    
                    # Clean up old data again after inserting new data
                    clean_old_data(interval)
                    
                    # Get updated history from database
                    history = get_price_history(asset, interval, limit)
                else:
                    print(f"No new data downloaded for {asset}")
                    if not history:  # If we have no data at all, raise an error
                        raise HTTPException(status_code=500, detail="No data available")
            except Exception as e:
                print(f"Error fetching from Binance: {e}")
                if not history:  # If we have no data at all, raise an error
                    raise HTTPException(status_code=500, detail=f"Error fetching price data: {str(e)}")
        
        # Format response data
        dates = [entry[0] for entry in history]
        prices = [entry[1] for entry in history]
        volumes = [entry[2] for entry in history if entry[2] is not None]
        
        # For 24h change calculation
        if prices:
            open_price = prices[-1]  # Last price in the history
            current_price = prices[0]  # Latest price
            price_change = ((current_price - open_price) / open_price) * 100
            high = max(prices)
            low = min(prices)
        else:
            current_price = 0
            price_change = 0
            high = 0
            low = 0
        
        # Get technical indicators from the latest data
        if prices:
            # Calculate RSI (simplified version)
            price_changes = [prices[i] - prices[i+1] for i in range(len(prices)-1)]
            gains = [change for change in price_changes if change > 0]
            losses = [-change for change in price_changes if change < 0]
            
            avg_gain = sum(gains) / len(gains) if gains else 0
            avg_loss = sum(losses) / len(losses) if losses else 0
            
            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            
            # Calculate MACD (simplified version)
            ema12 = sum(prices[:12]) / 12 if len(prices) >= 12 else current_price
            ema26 = sum(prices[:26]) / 26 if len(prices) >= 26 else current_price
            macd = ema12 - ema26
            signal = sum(prices[:9]) / 9 if len(prices) >= 9 else current_price
            histogram = macd - signal
            
            macd_data = {
                'MACD': macd,
                'Signal': signal,
                'Histogram': histogram
            }
        else:
            rsi = 50
            macd_data = {
                'MACD': 0,
                'Signal': 0,
                'Histogram': 0
            }
        
        response_data = {
            "dates": dates,
            "prices": prices,
            "volumes": volumes,
            "price_change_24h": price_change,
            "high_24h": high,
            "low_24h": low,
            "technical_indicators": {
                "rsi": rsi,
                "macd": macd_data
            }
        }
        
        return JSONResponse(content=response_data)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching price data: {str(e)}")

# Nowy endpoint generujący wykres jako obraz (matplotlib)
@app.get("/chart")
async def chart(
    symbol: str = Query("BTC", description="Symbol aktywa, np. BTC, ETH"),
    timeframe: str = Query("1d", description="Przedział czasowy: 1d, 1w, 1m, all")
):
    ticker = map_ticker(symbol)
    
    # Ustalanie okresu i interwału w zależności od timeframe
    if timeframe == "1d":
        period = "1d"
        interval = "1m"
    elif timeframe == "1w":
        period = "5d"   # pobieramy dane z ostatnich 5 dni
        interval = "5m"
    elif timeframe == "1m":
        period = "1mo"
        interval = "30m"
    elif timeframe == "1y":
        period = "1y"
        interval = "1d"
    elif timeframe == "all":
        period = "max"
        interval = "1d"
    else:
        raise HTTPException(status_code=400, detail="Niepoprawny timeframe. Użyj: 1d, 1w, 1m lub all.")
    
    # Pobieranie danych
    data = yf.download(ticker, period=period, interval=interval)
    if data.empty:
        raise HTTPException(status_code=404, detail="Brak danych dla wybranego aktywa i przedziału czasowego.")
    if "Adj Close" in data.columns:
        prices = data["Adj Close"]
    elif "Close" in data.columns:
        prices = data["Close"]
    else:
        raise HTTPException(status_code=500, detail="Brak odpowiednich kolumn w danych.")
    
    plt.figure(figsize=(10, 6))
    plt.plot(prices.index, prices, color="red", linewidth=2)
    plt.title(f"{symbol.upper()} Price Chart ({timeframe})")
    plt.xlabel("Data")
    plt.ylabel("Cena (USD)")
    plt.grid(True)
    
    if timeframe == "all":
        plt.yscale("log")
    
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)
    
    return StreamingResponse(buf, media_type="image/png")

def download_all_data():
    """Download data for all symbols and intervals, but only if data is missing or old"""
    symbols = ["BTC", "ETH", "SOL"]
    intervals = ["5min", "15min", "1h", "1d"]
    
    for symbol in symbols:
        for interval in intervals:
            print(f"\nProcessing {interval} data for {symbol}...")
            
            # First, clean up old data for this interval
            clean_old_data(interval)
            
            try:
                # Get the latest timestamp from database
                history = get_price_history(symbol, interval, limit=1)
                current_time = datetime.now()
                should_download = False
                
                if not history:
                    print(f"No data found for {symbol} at {interval} interval, downloading...")
                    should_download = True
                else:
                    try:
                        latest_timestamp = datetime.strptime(history[0][0], '%Y-%m-%d %H:%M:%S')
                        # Check if data is too old based on interval
                        if interval == "5min" and (current_time - latest_timestamp).total_seconds() > 300:  # 5 minutes
                            should_download = True
                        elif interval == "15min" and (current_time - latest_timestamp).total_seconds() > 900:  # 15 minutes
                            should_download = True
                        elif interval == "1h" and (current_time - latest_timestamp).total_seconds() > 3600:  # 1 hour
                            should_download = True
                        elif interval == "1d" and (current_time - latest_timestamp).total_seconds() > 86400:  # 1 day
                            should_download = True
                    except (ValueError, IndexError) as e:
                        print(f"Error parsing timestamp: {e}")
                        should_download = True
                
                if should_download:
                    print(f"Downloading {interval} data for {symbol}...")
                    # Set appropriate time range based on interval
                    end_time = current_time
                    if interval == "5min":
                        start_time = end_time - timedelta(days=1)
                    elif interval == "15min":
                        start_time = end_time - timedelta(days=7)
                    elif interval == "1h":
                        start_time = end_time - timedelta(days=30)
                    else:  # 1d - get all historical data
                        start_time = datetime(2017, 1, 1)
                    
                    # Download and store data
                    data = download_binance_data(symbol, interval, start_time=start_time, end_time=end_time)
                    if data:
                        print(f"Downloaded {len(data)} records for {symbol} at {interval} interval")
                        for timestamp, price, volume in data:
                            insert_price_data(symbol, price, timestamp, interval, volume)
                        
                        # Clean up old data again after inserting new data
                        clean_old_data(interval)
                    else:
                        print(f"No data downloaded for {symbol} at {interval} interval")
                else:
                    print(f"Data for {symbol} at {interval} interval is up to date")
                    
            except Exception as e:
                print(f"Error checking/downloading {interval} data for {symbol}: {e}")

# Initialize database and check/download data if needed
init_db()
download_all_data()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
