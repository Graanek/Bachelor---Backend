import numpy as np
import pandas as pd
from typing import Dict
from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import JSONResponse
from datetime import datetime
from typing import Optional
import logging
from app.database.operations import get_price_history, insert_price_data
from app.services.binance import download_binance_data
from app.services.portfolio import calculate_efficient_frontier
from app.config import DEFAULT_ASSETS

router = APIRouter()

@router.get("/portfolio")
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
        
        # Calculate efficient frontier and optimal portfolio
        result = calculate_efficient_frontier(returns)
        
        return JSONResponse(content=result)
        
    except ValueError as e:
        logger.error(f"ValueError in portfolio endpoint: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid date format: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in portfolio endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error calculating portfolio: {str(e)}")