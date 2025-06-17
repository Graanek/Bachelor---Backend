import numpy as np
import pandas as pd
from typing import Dict
   
"""
    Calculate the efficient frontier and optimal portfolio allocation.
    
    Args:
        returns (pd.DataFrame): DataFrame containing asset returns
        num_portfolios (int): Number of random portfolios to generate
        
    Returns:
        dict: Dictionary containing:
            - simulations: dict with vols, rets, and sharpes arrays
            - efficient_frontier: dict with vols and rets arrays
            - best_allocation: dict with optimal asset weights
"""




def calculate_efficient_frontier(returns: pd.DataFrame, num_portfolios: int = 1000) -> dict:

    # Calculate portfolio statistics
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    
    # Generate random portfolios
    num_assets = len(returns.columns)
    random_weights = np.random.random((num_portfolios, num_assets))
    random_weights = random_weights / random_weights.sum(axis=1, keepdims=True)
    
    # Calculate portfolio returns and volatility
    port_returns = (random_weights @ mean_returns.values) * 252  
    port_vols = np.sqrt(np.sum(random_weights * (random_weights @ cov_matrix), axis=1) * 252)  
    sharpes = np.divide(port_returns, port_vols, out=np.zeros_like(port_returns), where=port_vols != 0)
    
    # Find optimal portfolio
    best_idx = np.argmax(sharpes)
    best_weights = random_weights[best_idx]
    best_allocation = {asset: round(weight * 100, 2) for asset, weight in zip(returns.columns, best_weights)}
    
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
    
    return {
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
        "assets": returns.columns.tolist()
    }