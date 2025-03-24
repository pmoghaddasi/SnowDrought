"""
Functions for analyzing progression from snow drought to hydrologic drought.

This module provides tools to calculate conditional probabilities and analyze
the temporal relationship between snow drought and subsequent hydrologic drought.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


def load_drought_data(snow_drought_path, hydro_drought_path, threshold=0.3):
    """
    Load snow and hydrologic drought data and identify drought periods.
    
    Parameters:
    -----------
    snow_drought_path : str or Path
        Path to snow drought percentile data
    hydro_drought_path : str or Path
        Path to hydrologic drought percentile data
    threshold : float, default=0.3
        Percentile threshold for identifying drought conditions
    
    Returns:
    --------
    tuple
        (snow_drought_df, hydro_drought_df) with drought indicators added
    """
    try:
        # Load the data
        snow_drought_df = pd.read_csv(snow_drought_path)
        hydro_drought_df = pd.read_csv(hydro_drought_path)
        
        # Convert 'Date' column to datetime
        snow_drought_df['Date'] = pd.to_datetime(snow_drought_df['Date'], format='%Y%m%d')
        hydro_drought_df['Date'] = pd.to_datetime(hydro_drought_df['Date'], format='%Y%m%d')
        
        # Set 'Date' as the index
        snow_drought_df.set_index('Date', inplace=True)
        hydro_drought_df.set_index('Date', inplace=True)
        
        # Identify drought periods
        snow_drought_df['Snow_Drought'] = snow_drought_df['Percentile'] < threshold
        hydro_drought_df['Hydro_Drought'] = hydro_drought_df['Percentile'] < threshold
        
        return snow_drought_df, hydro_drought_df
    
    except Exception as e:
        print(f"Error loading drought data: {str(e)}")
        return None, None


def identify_drought_events(drought_series):
    """
    Identify continuous drought events from a boolean series.
    
    Parameters:
    -----------
    drought_series : pd.Series
        Boolean series where True indicates drought condition
    
    Returns:
    --------
    list of tuple
        List of (start_date, end_date) tuples for each drought event
    """
    drought_events = []
    in_drought = False
    start_date = None
    
    for date, is_drought in drought_series.items():
        if is_drought and not in_drought:
            in_drought = True
            start_date = date
        elif not is_drought and in_drought:
            in_drought = False
            drought_events.append((start_date, date - pd.Timedelta(days=1)))
    
    # Handle ongoing drought at the end of the series
    if in_drought:
        drought_events.append((start_date, drought_series.index[-1]))
    
    return drought_events


def calculate_drought_statistics(drought_events):
    """
    Calculate statistics for drought events.
    
    Parameters:
    -----------
    drought_events : list of tuple
        List of (start_date, end_date) tuples for drought events
    
    Returns:
    --------
    dict
        Dictionary of drought statistics
    """
    if not drought_events:
        return {
            'count': 0,
            'avg_duration': 0,
            'max_duration': 0,
            'min_duration': 0,
            'total_days': 0
        }
    
    durations = [(end - start).days + 1 for start, end in drought_events]
    
    return {
        'count': len(drought_events),
        'avg_duration': np.mean(durations),
        'max_duration': np.max(durations),
        'min_duration': np.min(durations),
        'total_days': sum(durations)
    }


def calculate_conditional_probability(snow_drought_events, hydro_drought_events, lag_days=90):
    """
    Calculate conditional probability of hydrologic drought following snow drought.
    
    Parameters:
    -----------
    snow_drought_events : list of tuple
        List of (start_date, end_date) tuples for snow drought events
    hydro_drought_events : list of tuple
        List of (start_date, end_date) tuples for hydrologic drought events
    lag_days : int, default=90
        Maximum lag time in days to consider
    
    Returns:
    --------
    dict
        Dictionary with probability, count of events, and total snow droughts
    """
    count_followed_by_hydro = 0
    total_snow_droughts = len(snow_drought_events)
    
    for snow_start, snow_end in snow_drought_events:
        # Define the window for hydrologic drought to follow
        window_start = snow_start
        window_end = snow_start + pd.Timedelta(days=lag_days)
        
        # Check if any hydrologic drought starts within this window
        hydro_in_window = any(
            hydro_start >= window_start and hydro_start <= window_end
            for hydro_start, hydro_end in hydro_drought_events
        )
        
        if hydro_in_window:
            count_followed_by_hydro += 1
    
    # Calculate probability
    probability = count_followed_by_hydro / total_snow_droughts if total_snow_droughts > 0 else np.nan
    
    return {
        'probability': probability,
        'count': count_followed_by_hydro,
        'total': total_snow_droughts
    }


def analyze_multiple_lags(snow_drought_events, hydro_drought_events, lag_days_list=None):
    """
    Calculate conditional probabilities for multiple lag times.
    
    Parameters:
    -----------
    snow_drought_events : list of tuple
        List of (start_date, end_date) tuples for snow drought events
    hydro_drought_events : list of tuple
        List of (start_date, end_date) tuples for hydrologic drought events
    lag_days_list : list, default=None
        List of lag times in days to analyze
    
    Returns:
    --------
    dict
        Dictionary with results for each lag time
    """
    if lag_days_list is None:
        lag_days_list = [7, 14, 30, 60, 90, 120, 180]
    
    results = {}
    
    for lag_days in lag_days_list:
        lag_results = calculate_conditional_probability(
            snow_drought_events, 
            hydro_drought_events, 
            lag_days=lag_days
        )
        
        # Store results for this lag time
        results[f'prob_{lag_days}d'] = lag_results['probability']
        results[f'count_{lag_days}d'] = lag_results['count']
    
    # Store the total number of snow droughts (same for all lag times)
    if snow_drought_events:
        results['total_snow_droughts'] = len(snow_drought_events)
    else:
        results['total_snow_droughts'] = 0
    
    return results


def analyze_basin_drought_progression(basin_id, snow_drought_path, hydro_drought_path, 
                                    threshold=0.3, lag_days_list=None):
    """
    Analyze drought progression for a single basin.
    
    Parameters:
    -----------
    basin_id : str
        Basin identifier
    snow_drought_path : str or Path
        Path to snow drought percentile data
    hydro_drought_path : str or Path
        Path to hydrologic drought percentile data
    threshold : float, default=0.3
        Percentile threshold for identifying drought conditions
    lag_days_list : list, default=None
        List of lag times in days to analyze
    
    Returns:
    --------
    dict
        Dictionary containing analysis results
    """
    # Load drought data
    snow_drought_df, hydro_drought_df = load_drought_data(
        snow_drought_path, hydro_drought_path, threshold
    )
    
    if snow_drought_df is None or hydro_drought_df is None:
        return None
    
    # Identify drought events
    snow_drought_events = identify_drought_events(snow_drought_df['Snow_Drought'])
    hydro_drought_events = identify_drought_events(hydro_drought_df['Hydro_Drought'])
    
    # Calculate drought statistics
    snow_stats = calculate_drought_statistics(snow_drought_events)
    hydro_stats = calculate_drought_statistics(hydro_drought_events)
    
    # Calculate conditional probabilities for multiple lag times
    lag_results = analyze_multiple_lags(
        snow_drought_events, 
        hydro_drought_events, 
        lag_days_list
    )
    
    # Combine all results
    results = {
        'basin_id': basin_id,
        'avg_snow_drought_duration': snow_stats['avg_duration'],
        'avg_hydro_drought_duration': hydro_stats['avg_duration'],
        'snow_drought_count': snow_stats['count'],
        'hydro_drought_count': hydro_stats['count']
    }
    results.update(lag_results)
    
    return results


def analyze_multiple_basins(basin_list, snow_drought_dir, hydro_drought_dir, 
                          threshold=0.3, lag_days_list=None, output_file=None):
    """
    Analyze drought progression for multiple basins.
    
    Parameters:
    -----------
    basin_list : list
        List of basin identifiers
    snow_drought_dir : str or Path
        Directory containing snow drought data files
    hydro_drought_dir : str or Path
        Directory containing hydrologic drought data files
    threshold : float, default=0.3
        Percentile threshold for identifying drought conditions
    lag_days_list : list, default=None
        List of lag times in days to analyze
    output_file : str or Path, default=None
        Path to save results CSV file
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with analysis results for all basins
    """
    results = []
    
    for basin_id in basin_list:
        print(f"Processing basin {basin_id}...")
        
        # Construct file paths
        snow_drought_path = os.path.join(snow_drought_dir, f"{basin_id}_UAZ_daily.csv")
        hydro_drought_path = os.path.join(hydro_drought_dir, f"{basin_id}_daily.csv")
        
        # Check if files exist
        if not os.path.exists(snow_drought_path) or not os.path.exists(hydro_drought_path):
            print(f"  Missing data files for basin {basin_id}, skipping.")
            continue
        
        # Analyze this basin
        basin_results = analyze_basin_drought_progression(
            basin_id, 
            snow_drought_path, 
            hydro_drought_path, 
            threshold, 
            lag_days_list
        )
        
        if basin_results:
            results.append(basin_results)
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results if output file is specified
    if output_file:
        results_df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
    
    return results_df





def analyze_drought_progression_by_threshold(basin_id, snow_drought_path, hydro_drought_path,
                                          thresholds=None, lag_days_list=None):
    """
    Analyze how drought progression probabilities change with different threshold values.
    
    Parameters:
    -----------
    basin_id : str
        Basin identifier
    snow_drought_path : str or Path
        Path to snow drought percentile data
    hydro_drought_path : str or Path
        Path to hydrologic drought percentile data
    thresholds : list, default=None
        List of percentile thresholds to test
    lag_days_list : list, default=None
        List of lag times in days to analyze
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with results for each threshold and lag time
    """
    if thresholds is None:
        thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    
    if lag_days_list is None:
        lag_days_list = [7, 14, 30, 60, 90, 120, 180]
    
    results = []
    
    for threshold in thresholds: