from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import JSONResponse
from datetime import datetime, timedelta
from app.services.binance import download_binance_data
from app.database.operations import get_price_history, insert_price_data, clean_old_data
from app.config import DB
import sqlite3
router = APIRouter()

@router.get("/price-data")
async def price_data(
    asset: str = Query(..., description="Symbol aktywa, np. BTC, ETH"),
    timeframe: str = Query(..., description="Przedział czasowy: 1d, 1w, 1m, all")
):
    # Map timeframe to interval and limit
    interval_map = {
        "1d": ("5min", 288),    # 5min intervals, 288 entries for 24h
        "1w": ("15min", 672),   # 15min intervals for 1 week
        "1m": ("1h", 720),      # 1h intervals for 1 month
        "all": ("1d", None)     # Daily data for all history
    }
    
    if timeframe not in interval_map:
        raise HTTPException(status_code=400, detail="Invalid timeframe")
    
    try:
        interval, limit = interval_map[timeframe]
        
        # Clean up old data for this interval
        clean_old_data(interval)
        
        # Always try to get data from database first
        history = get_price_history(asset, interval, limit)
        print(f"DEBUG: History for {asset} ({interval}): {history[:5]}...") # Sprawdź pierwsze 5 rekordów
        current_time = datetime.now()
        
        # Check if we need to fetch from API
        should_fetch_from_api = False
        
        if not history:
            should_fetch_from_api = True
            print(f"DEBUG: No history for {asset}, fetching from API.")
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
    # 1) figure out the one timestamp we already have, if any
            latest_ts = None
            if history:
                latest_ts = datetime.strptime(history[0][0], '%Y-%m-%d %H:%M:%S')

            # 2) delete that one bar so we can refresh it
            if latest_ts:
                conn = sqlite3.connect(DB)
                cur  = conn.cursor()
                cur.execute(
                    f"DELETE FROM crypto_history_{interval} "
                    "WHERE symbol_id = (SELECT id FROM symbols WHERE symbol = ?) "
                    "  AND timestamp = ?",
                    (asset, latest_ts.strftime('%Y-%m-%d %H:%M:%S'))
                )
                conn.commit()
                conn.close()

            # 3) set start_time = latest_ts (or fallback to your normal window)
            if latest_ts:
                start_time = latest_ts
            else:
                if interval == "5min":
                    start_time = current_time - timedelta(days=1)
                elif interval == "15min":
                    start_time = current_time - timedelta(days=7)
                elif interval == "1h":
                    start_time = current_time - timedelta(days=30)
                else:  # 1d
                    start_time = datetime(2013, 1, 1)

            # 4) fetch only from that point forward
            try:
                new_data = download_binance_data(asset, interval,
                                                start_time=start_time,
                                                end_time=current_time)
                # 5) insert only truly new bars (those > latest_ts)
                count = 0
                for ts, price, vol in new_data:
                    if latest_ts and ts <= latest_ts:
                        continue
                    insert_price_data(asset, price, ts, interval, vol)
                    count += 1
                print(f"Inserted {count} new records for {asset} @ {interval}")

                # clean up anything older than your retention policy
                clean_old_data(interval)

                # finally re-load our history slice
                history = get_price_history(asset, interval, limit)
            except Exception as e:
                print(f"Error fetching incremental data: {e}")
                if not history:
                    raise HTTPException(500, f"Failed to fetch any data: {e}")
        
        # Format response data
        dates = [entry[0] for entry in history]
        prices = [entry[1] for entry in history]
        volumes = [entry[2] for entry in history if entry[2] is not None]
        
        print(f"DEBUG: Prices for {asset}: {prices[:5]}...") # Sprawdź pierwsze 5 cen
        
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
