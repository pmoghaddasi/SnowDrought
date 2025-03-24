"""
Functions for identifying drought conditions based on percentile thresholds.

This module provides tools to calculate percentile ranks and identify drought
conditions based on climatological thresholds.
"""

import os
import datetime
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from .climatology import shift_date, load_climatology


def calculate_percentile_rank(value, reference_values):
    """
    Calculate the percentile rank of a value within reference values.
    
    Parameters:
    -----------
    value : float
        Value to calculate percentile for
    reference_values : array-like
        Reference distribution
    
    Returns:
    --------
    float
        Percentile rank (0-1)
    """
    # Remove NaN values
    valid_refs = np.array(reference_values)[~np.isnan(reference_values)]
    if np.isnan(value) or len(valid_refs) == 0:
        return np.nan
    
    # Add the value to the reference values
    combined = np.append(valid_refs, value)
    
    # Rank the values
    ranks = rankdata(combined, method='average')
    
    # Extract the rank of the value (last entry) and normalize
    percentile = ranks[-1] / len(combined)
    
    return percentile


def calculate_daily_percentiles(data, climatology_dir, basin_id, value_column='OBS_RUN',
                              date_column=None, start_year=1982, end_year=2014,
                              base_date='19820101'):
    """
    Calculate daily percentile ranks based on climatology.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing time series data
    climatology_dir : str
        Directory containing climatology files
    basin_id : str
        Basin identifier
    value_column : str, default='OBS_RUN'
        Column name for the values to calculate percentiles
    date_column : str, optional
        Column name for dates, if data is not indexed by date
    start_year, end_year : int, default=1982, 2014
        Year range for percentile calculation
    base_date : str, default='19820101'
        Base date for calculations in 'YYYYMMDD' format
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with dates, values, and percentile ranks
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
    
    # Initialize lists for results
    dates = []
    percentiles = []
    values = []
    days_of_year = []
    
    # Calculate percentiles for each day
    for year in range(start_year, end_year + 1):
        for d in range(365):
            # Get the date
            current_date = shift_date(base_date, d)
            current_mmdd = current_date[4:8]  # Month and day
            date_str = f"{year}{current_mmdd}"
            
            try:
                # Convert to datetime
                date_dt = datetime.datetime.strptime(date_str, '%Y%m%d')
                
                # Get value for this date if it exists in data
                if date_dt in df.index:
                    value = df.loc[date_dt, value_column]
                else:
                    value = np.nan
                
                # Load climatology for this day
                climatology_data = load_climatology(climatology_dir, basin_id, current_mmdd)
                
                # Calculate percentile rank
                percentile = calculate_percentile_rank(value, climatology_data)
                
                # Store results
                dates.append(date_str)
                percentiles.append(percentile)
                values.append(value)
                days_of_year.append(d)
                
            except ValueError:
                # Skip invalid dates
                continue
    
    # Create DataFrame with results
    result_df = pd.DataFrame({
        'Date': dates,
        'Percentile': percentiles,
        value_column: values,
        'Day': days_of_year
    })
    
    return result_df


def identify_drought_conditions(percentile_df, threshold=0.3, percentile_column='Percentile'):
    """
    Identify drought conditions based on percentile threshold.
    
    Parameters:
    -----------
    percentile_df : pd.DataFrame
        DataFrame with percentile ranks
    threshold : float, default=0.3
        Percentile threshold for drought (values below this are in drought)
    percentile_column : str, default='Percentile'
        Column name for percentile values
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with additional drought indicator column
    """
    result_df = percentile_df.copy()
    result_df['Drought'] = (result_df[percentile_column] < threshold) & (~result_df[percentile_column].isna())
    return result_df


def identify_drought_events(drought_df, date_column='Date', drought_column='Drought',
                          min_duration=1):
    """
    Identify distinct drought events.
    
    Parameters:
    -----------
    drought_df : pd.DataFrame
        DataFrame with drought indicators
    date_column : str, default='Date'
        Column name for dates
    drought_column : str, default='Drought'
        Column name for drought indicators
    min_duration : int, default=1
        Minimum number of days to consider as a drought event
    
    Returns:
    --------
    list of dict
        List of drought events with start date, end date, and duration
    """
    # Sort by date
    df = drought_df.copy()
    df[date_column] = pd.to_datetime(df[date_column], format='%Y%m%d')
    df = df.sort_values(by=date_column)
    
    # Initialize variables
    events = []
    in_drought = False
    start_date = None
    
    # Iterate through dates
    for i, row in df.iterrows():
        if row[drought_column] and not in_drought:
            # Start of drought
            in_drought = True
            start_date = row[date_column]
        elif not row[drought_column] and in_drought:
            # End of drought
            in_drought = False
            end_date = row[date_column] - datetime.timedelta(days=1)
            duration = (end_date - start_date).days + 1
            
            if duration >= min_duration:
                events.append({
                    'start_date': start_date,
                    'end_date': end_date,
                    'duration': duration
                })
    
    # Handle ongoing drought at the end
    if in_drought:
        end_date = df[date_column].iloc[-1]
        duration = (end_date - start_date).days + 1
        
        if duration >= min_duration:
            events.append({
                'start_date': start_date,
                'end_date': end_date,
                'duration': duration
            })
    
    return events


def calculate_drought_statistics(events):
    """
    Calculate statistics for drought events.
    
    Parameters:
    -----------
    events : list of dict
        List of drought events
    
    Returns:
    --------
    dict
        Dictionary with drought statistics
    """
    if not events:
        return {
            'count': 0,
            'avg_duration': 0,
            'max_duration': 0,
            'min_duration': 0,
            'total_drought_days': 0
        }
    
    durations = [event['duration'] for event in events]
    
    return {
        'count': len(events),
        'avg_duration': np.mean(durations),
        'max_duration': np.max(durations),
        'min_duration': np.min(durations),
        'total_drought_days': sum(durations)
    }


def process_basin_for_drought(data_path, climatology_dir, output_dir, 
                            value_column='OBS_RUN', date_column='date',
                            threshold=0.3):
    """
    Process a basin to calculate drought percentiles and identify drought conditions.
    
    Parameters:
    -----------
    data_path : str
        Path to CSV file with basin data
    climatology_dir : str
        Directory containing climatology files
    output_dir : str
        Directory to save results
    value_column : str, default='OBS_RUN'
        Column name for values
    date_column : str, default='date'
        Column name for dates
    threshold : float, default=0.3
        Percentile threshold for drought
    
    Returns:
    --------
    tuple
        (percentile_df, drought_events, drought_stats)
    """
    # Extract basin ID from filename
    basename = os.path.basename(data_path)
    basin_id = basename.split('_')[0]
    
    # Read data
    df = pd.read_csv(data_path)
    
    # Calculate percentiles
    percentile_df = calculate_daily_percentiles(
        df, climatology_dir, basin_id, value_column, date_column
    )
    
    # Identify drought conditions
    drought_df = identify_drought_conditions(percentile_df, threshold)
    
    # Identify drought events
    drought_events = identify_drought_events(drought_df)
    
    # Calculate drought statistics
    drought_stats = calculate_drought_statistics(drought_events)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results
    output_file = os.path.join(output_dir, f"{basin_id}_drought_daily.csv")
    drought_df.to_csv(output_file, index=False)
    
    return percentile_df, drought_events, drought_stats


def main():
    """Command-line interface for drought identification."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Calculate drought percentiles and identify drought conditions')
    parser.add_argument('--data_path', type=str, required=True, help='Path to basin data CSV file')
    parser.add_argument('--climatology_dir', type=str, required=True, help='Directory for climatology files')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save results')
    parser.add_argument('--value_column', type=str, default='OBS_RUN', help='Column name for values')
    parser.add_argument('--date_column', type=str, default='date', help='Column name for dates')
    parser.add_argument('--threshold', type=float, default=0.3, help='Percentile threshold for drought')
    parser.add_argument('--calculate_climatology', action='store_true', help='Calculate climatology first')
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.climatology_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Calculate climatology if requested
    if args.calculate_climatology:
        print(f"Calculating climatology for {args.data_path}...")
        calculate_and_save_climatology(
            args.data_path, args.climatology_dir, 
            args.value_column, args.date_column
        )
    
    # Process basin for drought
    print(f"Calculating drought percentiles for {args.data_path}...")
    percentile_df, drought_events, drought_stats = process_basin_for_drought(
        args.data_path, args.climatology_dir, args.output_dir,
        args.value_column, args.date_column, args.threshold
    )
    
    # Print summary
    print(f"Drought statistics:")
    for stat, value in drought_stats.items():
        print(f"  {stat}: {value}")


if __name__ == "__main__":
    main()