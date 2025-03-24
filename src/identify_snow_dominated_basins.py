"""
This module contains functions to identify snow-dominated basins based on Snow Water Equivalent (SWE) data.

Snow-dominated basins are defined as those where SWE exceeds a threshold (default 30mm)
for a continuous period (default 3 months) for a specified percentage of years (default 95%).
"""

import os
import pandas as pd
import numpy as np


def calculate_consecutive_days_above_threshold(swe_series, threshold=30):
    """
    Calculate the number of consecutive days SWE remains above a threshold.
    
    Parameters:
    -----------
    swe_series : pd.Series
        Time series of SWE values
    threshold : float, default=30
        SWE threshold in mm
    
    Returns:
    --------
    list
        List of durations (in days) of consecutive periods above threshold
    """
    # Create a mask for values above threshold
    above_threshold = swe_series > threshold
    
    # Identify changes in state (above/below threshold)
    shifts = above_threshold.ne(above_threshold.shift())
    
    # Assign group numbers to consecutive periods
    group_number = shifts.cumsum()
    
    # Calculate duration of each consecutive period
    durations = above_threshold.groupby(group_number).sum()
    
    # Return only durations where values were above threshold
    return [int(d) for i, d in enumerate(durations) if above_threshold.iloc[shifts.index.get_indexer([shifts[shifts].index[i]])[0]]]


def is_year_snow_dominated(year_data, swe_column, threshold=30, min_months=3):
    """
    Determine if a specific year is snow-dominated based on SWE data.
    
    Parameters:
    -----------
    year_data : pd.DataFrame
        Data for a specific year
    swe_column : str
        Column name containing SWE values
    threshold : float, default=30
        SWE threshold in mm
    min_months : float, default=3
        Minimum consecutive months SWE must exceed threshold
    
    Returns:
    --------
    bool
        True if year is snow-dominated, False otherwise
    """
    if year_data.empty or swe_column not in year_data.columns:
        return False
    
    # Get consecutive days above threshold
    consecutive_days = calculate_consecutive_days_above_threshold(
        year_data[swe_column].fillna(0), threshold)
    
    # Convert days to months (approximate)
    consecutive_months = [days / 30.0 for days in consecutive_days]
    
    # Check if any period exceeds minimum months
    return any(months >= min_months for months in consecutive_months)


def is_basin_snow_dominated(basin_data, swe_column='uaz_swe', threshold=30, 
                           min_months=3, year_percent=0.95):
    """
    Determine if a basin is snow-dominated based on SWE criteria.
    
    Parameters:
    -----------
    basin_data : pd.DataFrame
        SWE data for the basin with date column
    swe_column : str, default='uaz_swe'
        Column name containing SWE values
    threshold : float, default=30
        SWE threshold in mm
    min_months : float, default=3
        Minimum consecutive months SWE must exceed threshold
    year_percent : float, default=0.95
        Percentage of years that must meet criteria
    
    Returns:
    --------
    bool
        True if basin is snow-dominated, False otherwise
    dict
        Additional details including number of snow-dominated years
    """
    if 'date' not in basin_data.columns:
        raise ValueError("Basin data must contain a 'date' column")
    
    # Convert date column to datetime if needed
    if not pd.api.types.is_datetime64_dtype(basin_data['date']):
        basin_data['date'] = pd.to_datetime(basin_data['date'])
    
    # Group data by year
    yearly_groups = basin_data.groupby(basin_data['date'].dt.year)
    
    # Check each year
    snow_dominated_years = {}
    for year, group in yearly_groups:
        snow_dominated_years[year] = is_year_snow_dominated(
            group, swe_column, threshold, min_months)
    
    # Calculate percentage of snow-dominated years
    total_years = len(snow_dominated_years)
    snow_dominated_count = sum(snow_dominated_years.values())
    
    if total_years == 0:
        return False, {"snow_dominated_years": 0, "total_years": 0, "percentage": 0}
    
    percentage = snow_dominated_count / total_years
    
    # Check if percentage meets the criterion
    is_dominated = percentage >= year_percent
    
    return is_dominated, {
        "snow_dominated_years": snow_dominated_count,
        "total_years": total_years,
        "percentage": percentage
    }


def identify_snow_dominated_basins(data_dir, output_file=None, swe_column='uaz_swe', 
                                threshold=30, min_months=3, year_percent=0.95):
    """
    Identify snow-dominated basins from a directory of basin data files.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing basin data files
    output_file : str, optional
        Path to output CSV file for results
    swe_column : str, default='uaz_swe'
        Column name containing SWE values
    threshold : float, default=30
        SWE threshold in mm
    min_months : float, default=3
        Minimum consecutive months SWE must exceed threshold
    year_percent : float, default=0.95
        Percentage of years that must meet criteria
    
    Returns:
    --------
    pd.DataFrame
        DataFrame containing basin IDs and snow-dominated status
    """
    results = []
    
    # Process each file in the directory
    for filename in os.listdir(data_dir):
        if filename.endswith('.csv'):
            basin_id = filename.split('.')[0]  # Extract basin ID from filename
            file_path = os.path.join(data_dir, filename)
            
            try:
                # Read basin data
                basin_data = pd.read_csv(file_path)
                
                # Check if basin is snow-dominated
                is_dominated, details = is_basin_snow_dominated(
                    basin_data, swe_column, threshold, min_months, year_percent)
                
                # Store results
                results.append({
                    'basin_id': basin_id,
                    'is_snow_dominated': is_dominated,
                    'snow_dominated_years': details['snow_dominated_years'],
                    'total_years': details['total_years'],
                    'percentage': details['percentage']
                })
                
                print(f"Processed basin {basin_id}: {'Snow-dominated' if is_dominated else 'Not snow-dominated'}")
                
            except Exception as e:
                print(f"Error processing basin {basin_id}: {str(e)}")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results if output file is specified
    if output_file:
        results_df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
    
    return results_df


def main():
    """
    Main function to run the snow-dominated basin identification.
    """
    # Example usage - replace with your paths and parameters
    data_dir = "data/basin_swe"
    output_file = "data/snow_dominated_basins.csv"
    
    # Use default parameters
    results = identify_snow_dominated_basins(
        data_dir=data_dir,
        output_file=output_file,
        swe_column='uaz_swe',  # Change to match your data
        threshold=30,          # SWE threshold in mm
        min_months=3,          # Minimum consecutive months above threshold
        year_percent=0.95      # Percentage of years required
    )
    
    # Print summary
    snow_dominated_count = results['is_snow_dominated'].sum()
    total_basins = len(results)
    print(f"\nIdentified {snow_dominated_count} snow-dominated basins out of {total_basins} total basins.")
    print(f"This represents {snow_dominated_count/total_basins*100:.1f}% of the analyzed basins.")


if __name__ == "__main__":
    main()