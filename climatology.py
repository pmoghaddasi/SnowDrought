"""
Functions for calculating climatology from daily time series data.

This module provides tools to create climatologies for each day of the year
based on historical data, which can be used for drought identification.
"""

import os
import datetime
import numpy as np
import pandas as pd


def convert_to_date(date_str):
    """
    Convert a date string to a datetime object.
    
    Parameters:
    -----------
    date_str : str
        Date string in 'YYYYMMDD' format
    
    Returns:
    --------
    datetime.date
        Date object
    """
    year = int(date_str[:4])
    month = int(date_str[4:6])
    day = int(date_str[6:8])
    return datetime.date(year, month, day)


def shift_date(date_str, delta):
    """
    Shift a date by a specified number of days.
    
    Parameters:
    -----------
    date_str : str
        Date string in 'YYYYMMDD' format
    delta : int
        Number of days to shift (positive or negative)
    
    Returns:
    --------
    str
        Shifted date in 'YYYYMMDD' format
    """
    date = convert_to_date(date_str)
    new_date = date + datetime.timedelta(days=delta)
    return new_date.strftime('%Y%m%d')


def calculate_daily_climatology(data, value_column, date_column=None, 
                              start_year=1982, end_year=2014, 
                              window_size=5, base_date='19820101'):
    """
    Calculate climatology for each day of the year.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing time series data
    value_column : str
        Column name for the values to calculate climatology
    date_column : str, optional
        Column name for dates, if data is not indexed by date
    start_year, end_year : int, default=1982, 2014
        Year range for climatology calculation
    window_size : int, default=5
        Size of the moving window (in days) for climatology calculation
    base_date : str, default='19820101'
        Base date for calculations in 'YYYYMMDD' format
    
    Returns:
    --------
    dict
        Dictionary with keys as day-of-year (MMDD) and values as numpy arrays of climatology data
    """
    # If data is not indexed by date, set the index
    if date_column is not None:
        df = data.copy()
        df['date'] = pd.to_datetime(df[date_column])
        df.set_index('date', inplace=True)
    else:
        df = data.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Data must have a datetime index or a date column")
    
    climatology_dict = {}
    
    # Calculate climatology for each day of the year
    for d in range(365):
        current_date = shift_date(base_date, d)
        current_mmdd = current_date[4:8]  # Extract month and day
        
        climatology_values = []
        
        # Collect values for this day across years
        for year in range(start_year, end_year + 1):
            date_str = f"{year}{current_mmdd}"
            
            try:
                # Convert to datetime
                date_dt = datetime.datetime.strptime(date_str, '%Y%m%d')
                
                # Calculate window around the date
                for offset in range(-(window_size - 1) // 2, (window_size + 1) // 2):
                    offset_date = date_dt + datetime.timedelta(days=offset)
                    
                    # Check if offset date exists in data
                    if offset_date in df.index:
                        value = df.loc[offset_date, value_column]
                        climatology_values.append(value)
            except ValueError:
                # Skip invalid dates (e.g., February 29 in non-leap years)
                continue
        
        # Store climatology for this day
        climatology_dict[current_mmdd] = np.array(climatology_values)
    
    return climatology_dict


def save_climatology(climatology_dict, output_dir, basin_id):
    """
    Save climatology data to files.
    
    Parameters:
    -----------
    climatology_dict : dict
        Dictionary with climatology data
    output_dir : str
        Directory to save climatology files
    basin_id : str
        Basin identifier
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save each day's climatology to a separate file
    for mmdd, values in climatology_dict.items():
        filename = f"{basin_id}_climatology_{mmdd}.npy"
        file_path = os.path.join(output_dir, filename)
        np.save(file_path, values)


def load_climatology(input_dir, basin_id, mmdd):
    """
    Load climatology data from file.
    
    Parameters:
    -----------
    input_dir : str
        Directory containing climatology files
    basin_id : str
        Basin identifier
    mmdd : str
        Month and day in 'MMDD' format
    
    Returns:
    --------
    np.ndarray
        Array of climatology values for the specified day
    """
    filename = f"{basin_id}_climatology_{mmdd}.npy"
    file_path = os.path.join(input_dir, filename)
    
    if os.path.exists(file_path):
        return np.load(file_path, allow_pickle=True)
    else:
        return np.array([])


def calculate_and_save_climatology(data_path, output_dir, value_column='OBS_RUN', 
                                 date_column=None, window_size=5):
    """
    Calculate and save climatology for a basin.
    
    Parameters:
    -----------
    data_path : str
        Path to CSV file with basin data
    output_dir : str
        Directory to save climatology files
    value_column : str, default='OBS_RUN'
        Column name for the values to calculate climatology
    date_column : str, optional
        Column name for dates, if data is not indexed by date
    window_size : int, default=5
        Size of the moving window for climatology calculation
    
    Returns:
    --------
    dict
        Dictionary with climatology data
    """
    # Extract basin ID from filename
    basename = os.path.basename(data_path)
    basin_id = basename.split('_')[0]
    
    # Read data
    df = pd.read_csv(data_path)
    
    # Prepare date column if needed
    if date_column is None and 'date' in df.columns:
        date_column = 'date'
    
    # Calculate climatology
    climatology_dict = calculate_daily_climatology(
        df, value_column, date_column, window_size=window_size
    )
    
    # Save climatology
    save_climatology(climatology_dict, output_dir, basin_id)
    
    return climatology_dict