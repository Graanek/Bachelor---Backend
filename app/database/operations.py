import sqlite3
from datetime import datetime, timedelta
from typing import List, Tuple, Optional

from app.config import DB

def get_or_create_symbol(symbol: str) -> int:
    """Get symbol ID or create new symbol if it doesn't exist"""
    conn = sqlite3.connect(DB)
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
    conn = sqlite3.connect(DB)
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
    conn = sqlite3.connect(DB)
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


def clean_old_data(interval: str):
    """Clean up old data based on the interval"""
    conn = sqlite3.connect(DB)
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
