import sqlite3
from datetime import date
import requests
from io import StringIO
import csv
import fear_and_greed
from tradingview_ta import TA_Handler, Interval
from app.config import DB


def fetch_fear_and_greed() -> float:
    # returns 0–100
    idx = fear_and_greed.get()
    return idx.get_current_value()

def fetch_vix() -> float:
    # Download the CSV from datasets/finance-vix repo
    url = "https://raw.githubusercontent.com/datasets/finance-vix/master/data/vix-daily.csv"
    resp = requests.get(url)
    if resp.status_code != 200:
        raise RuntimeError("Failed to download VIX CSV")
    # parse CSV and pull today's date
    reader = csv.DictReader(StringIO(resp.text))
    today = date.today().isoformat()
    last_value = None
    for row in reader:
        last_value = float(row["CLOSE"])
    if last_value is not None:
        return last_value
    raise RuntimeError("No VIX data found")

def fetch_rsi_macd(asset: str) -> dict:

    handler = TA_Handler(
        symbol=f"{asset}USD",
        screener="crypto",
        exchange="BINANCE",
        interval=Interval.INTERVAL_1_DAY
    )
    ta = handler.get_analysis().indicators
    return {
      "rsi": ta.get("RSI", 0),
      "macd": ta.get("MACD.macd", 0),
      "macd_signal": ta.get("MACD.signal", 0),
      "macd_histogram": ta.get("MACD.histogram", 0)
    }



def get_cached_global(d: str):
    conn = sqlite3.connect(DB)
    row = conn.execute(
      "SELECT fear_greed, vix FROM global_sentiment WHERE entry_date = ?", (d,)
    ).fetchone()
    conn.close()
    if not row:
        return None
    return {"fear_greed": row[0], "vix": row[1]}


def cache_global(d: str, fg: float, vix: float):
    conn = sqlite3.connect(DB)
    conn.execute("""
      INSERT OR REPLACE INTO global_sentiment(entry_date, fear_greed, vix)
      VALUES (?, ?, ?)
    """, (d, fg, vix))
    conn.commit()
    conn.close()


def get_cached_ta(d: str, asset: str):
    conn = sqlite3.connect(DB)
    row = conn.execute("""
      SELECT rsi, macd, macd_signal, macd_histogram
      FROM asset_ta WHERE entry_date = ? AND asset = ?
    """, (d, asset)).fetchone()
    conn.close()
    if not row:
        return None
    return {
      "rsi": row[0],
      "macd": row[1],
      "macd_signal": row[2],
      "macd_histogram": row[3]
    }


def cache_ta(d: str, asset: str, ta: dict):
    conn = sqlite3.connect(DB)
    conn.execute("""
      INSERT OR REPLACE INTO asset_ta
      (entry_date, asset, rsi, macd, macd_signal, macd_histogram)
      VALUES (?, ?, ?, ?, ?, ?)
    """, (d, asset, ta["rsi"], ta["macd"], ta["macd_signal"], ta["macd_histogram"]))
    conn.commit()
    conn.close()


