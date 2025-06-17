from sqlite3 import Connection
from datetime import datetime




def init_db(conn: Connection):
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
    cursor.execute('''
      CREATE TABLE IF NOT EXISTS global_sentiment (
        entry_date DATE PRIMARY KEY,
        fear_greed REAL,
        vix REAL,
        fetched_at DATETIME DEFAULT CURRENT_TIMESTAMP
      )
    ''')
    # Per-asset: one row per day per symbol
    cursor.execute('''
      CREATE TABLE IF NOT EXISTS asset_ta (
        entry_date DATE,
        asset TEXT,
        rsi REAL,
        macd REAL,
        macd_signal REAL,
        macd_histogram REAL,
        fetched_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (entry_date, asset)
      )
    ''')
    
    conn.commit()
    conn.close()