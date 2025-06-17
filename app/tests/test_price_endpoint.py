import pytest
from fastapi.testclient import TestClient
from datetime import datetime, timedelta
from app.main import app
from app.database.operations import insert_price_data, get_or_create_symbol
from app.config import DB
import sqlite3
from unittest.mock import patch

client = TestClient(app)

@pytest.fixture
def setup_test_data():
    # Create test data in the database
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    
    # Create test tables if they don't exist
    cur.execute('''
        CREATE TABLE IF NOT EXISTS symbols (
            id INTEGER PRIMARY KEY,
            symbol TEXT UNIQUE
        )
    ''')
    
    cur.execute('''
        CREATE TABLE IF NOT EXISTS crypto_history_5min (
            id INTEGER PRIMARY KEY,
            symbol_id INTEGER,
            timestamp TEXT,
            price REAL,
            volume REAL,
            FOREIGN KEY (symbol_id) REFERENCES symbols(id)
        )
    ''')
    
    # Clear existing data
    cur.execute("DELETE FROM crypto_history_5min")
    cur.execute("DELETE FROM symbols")
    
    # Insert test symbol
    cur.execute("INSERT INTO symbols (symbol) VALUES (?)", ("BTC",))
    conn.commit()
    
    # Insert some test price data
    test_timestamp = datetime.now()
    for i in range(10):
        timestamp = test_timestamp - timedelta(minutes=5*i)
        timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
        insert_price_data("BTC", 50000 + i, timestamp_str, "5min", 1000 + i)
    
    conn.close()

@pytest.fixture
def mock_binance_data():
    # Mock data that would be returned by Binance API
    mock_data = [
        (datetime.now() - timedelta(minutes=i), 50000 + i, 1000 + i)
        for i in range(10)
    ]
    return mock_data

def test_price_data_invalid_timeframe():
    response = client.get("/api/price-data?asset=BTC&timeframe=invalid")
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid timeframe"

def test_price_data_missing_parameters():
    response = client.get("/api/price-data")
    assert response.status_code == 422  # FastAPI validation error

@patch('app.services.binance.download_binance_data')
def test_price_data_success(mock_download, setup_test_data, mock_binance_data):
    # Mock the Binance API response
    mock_download.return_value = mock_binance_data
    
    response = client.get("/api/price-data?asset=BTC&timeframe=1d")
    assert response.status_code == 200
    
    data = response.json()
    assert "dates" in data
    assert "prices" in data
    assert "volumes" in data
    assert "price_change_24h" in data
    assert "high_24h" in data
    assert "low_24h" in data
    assert "technical_indicators" in data
    
    # Check technical indicators structure
    tech_indicators = data["technical_indicators"]
    assert "rsi" in tech_indicators
    assert "macd" in tech_indicators
    assert "MACD" in tech_indicators["macd"]
    assert "Signal" in tech_indicators["macd"]
    assert "Histogram" in tech_indicators["macd"]

@patch('app.services.binance.download_binance_data')
def test_price_data_different_timeframes(mock_download, setup_test_data, mock_binance_data):
    # Mock the Binance API response
    mock_download.return_value = mock_binance_data
    
    timeframes = ["1d", "1w", "1m", "all"]
    for timeframe in timeframes:
        response = client.get(f"/api/price-data?asset=BTC&timeframe={timeframe}")
        assert response.status_code == 200
        data = response.json()
        assert len(data["dates"]) > 0
        assert len(data["prices"]) > 0

@patch('app.services.binance.download_binance_data')
def test_price_data_nonexistent_asset(mock_download):
    # Mock the Binance API to raise an error for invalid symbol
    mock_download.side_effect = Exception("Invalid symbol")
    
    response = client.get("/api/price-data?asset=INVALID&timeframe=1d")
    assert response.status_code == 500
    data = response.json()
    assert "detail" in data
    assert "Error fetching price data" in data["detail"]

@patch('app.services.binance.download_binance_data')
def test_price_data_technical_indicators(mock_download, setup_test_data, mock_binance_data):
    # Mock the Binance API response
    mock_download.return_value = mock_binance_data
    
    response = client.get("/api/price-data?asset=BTC&timeframe=1d")
    assert response.status_code == 200
    
    data = response.json()
    tech_indicators = data["technical_indicators"]
    
    # Check RSI is within valid range (0-100)
    assert 0 <= tech_indicators["rsi"] <= 100
    
    # Check MACD components
    macd = tech_indicators["macd"]
    assert isinstance(macd["MACD"], (int, float))
    assert isinstance(macd["Signal"], (int, float))
    assert isinstance(macd["Histogram"], (int, float))