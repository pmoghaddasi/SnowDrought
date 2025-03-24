"""
LSTM model for predicting streamflow from SWE data.

This module provides functions to build, train, and evaluate LSTM models for streamflow
prediction using Snow Water Equivalent (SWE) as input.
"""

import os
import time
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


def create_sequences(data, n_past, n_future, feature_cols, target_col):
    """
    Create input sequences for LSTM model.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame with features and target
    n_past : int
        Number of past time steps to use as input
    n_future : int
        Number of future time steps to predict
    feature_cols : list
        List of feature column names
    target_col : str
        Target column name
    
    Returns:
    --------
    tuple
        (X, y) arrays for model training
    """
    X, y = [], []
    for i in range(len(data) - n_past - n_future + 1):
        X.append(data.iloc[i:i + n_past][feature_cols].values)
        y.append(data.iloc[i + n_past:i + n_past + n_future][target_col].values)
    return np.array(X), np.array(y)


def preprocess_data(df, feature_cols, target_col, lag, test_size=0.15, val_size=0.2, random_state=42):
    """
    Preprocess data for LSTM model training.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with features and target
    feature_cols : list
        List of feature column names
    target_col : str
        Target column name
    lag : int
        Number of past time steps to use as input
    test_size : float, default=0.15
        Proportion of data to use for testing
    val_size : float, default=0.2
        Proportion of training data to use for validation
    random_state : int, default=42
        Random seed for reproducibility
    
    Returns:
    --------
    tuple
        Preprocessed data splits and normalization parameters
    """
    # Create sequences
    X_seq, y_seq = create_sequences(df, lag, 1, feature_cols, target_col)
    
    # Train-test split
    split_point = int((1 - test_size) * len(X_seq))
    X_train_val, X_test = X_seq[:split_point], X_seq[split_point:]
    y_train_val, y_test = y_seq[:split_point], y_seq[split_point:]
    
    # Train-validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size, random_state=random_state
    )
    
    # Normalize the data
    mean_train_X = X_train.mean()
    std_train_X = X_train.std()
    mean_train_y = y_train.mean()
    std_train_y = y_train.std()
    
    X_train = (X_train - mean_train_X) / std_train_X
    X_val = (X_val - mean_train_X) / std_train_X
    X_test = (X_test - mean_train_X) / std_train_X
    
    y_train = (y_train - mean_train_y) / std_train_y
    y_val = (y_val - mean_train_y) / std_train_y
    y_test = (y_test - mean_train_y) / std_train_y
    
    norm_params = {
        'mean_X': mean_train_X,
        'std_X': std_train_X,
        'mean_y': mean_train_y,
        'std_y': std_train_y
    }
    
    return X_train, X_val, X_test, y_train, y_val, y_test, norm_params


def build_lstm_model(input_shape, first_lstm_units=50, second_lstm_units=30, 
                    dropout_rate=0.2, learning_rate=0.001):
    """
    Build an LSTM model for time series prediction.
    
    Parameters:
    -----------
    input_shape : tuple
        Shape of input data (timesteps, features)
    first_lstm_units : int, default=50
        Number of units in the first LSTM layer
    second_lstm_units : int, default=30
        Number of units in the second LSTM layer
    dropout_rate : float, default=0.2
        Dropout rate for regularization
    learning_rate : float, default=0.001
        Learning rate for the Adam optimizer
    
    Returns:
    --------
    tf.keras.Model
        Compiled LSTM model
    """
    model = Sequential()
    model.add(LSTM(
        units=first_lstm_units,
        return_sequences=True,
        activation='tanh',
        input_shape=input_shape
    ))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(
        units=second_lstm_units,
        return_sequences=False,
        activation='tanh'
    ))
    model.add(Dense(units=1))
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')
    
    return model


def train_and_evaluate_model(X_train, y_train, X_val, y_val, X_test, y_test, norm_params,
                          first_lstm_units=50, second_lstm_units=30, dropout_rate=0.2,
                          learning_rate=0.001, epochs=40, batch_size=64, verbose=1):
    """
    Train and evaluate an LSTM model.
    
    Parameters:
    -----------
    X_train, y_train : np.ndarray
        Training data
    X_val, y_val : np.ndarray
        Validation data
    X_test, y_test : np.ndarray
        Test data
    norm_params : dict
        Normalization parameters
    first_lstm_units : int, default=50
        Number of units in the first LSTM layer
    second_lstm_units : int, default=30
        Number of units in the second LSTM layer
    dropout_rate : float, default=0.2
        Dropout rate for regularization
    learning_rate : float, default=0.001
        Learning rate for the Adam optimizer
    epochs : int, default=40
        Number of training epochs
    batch_size : int, default=64
        Batch size for training
    verbose : int, default=1
        Verbosity level
    
    Returns:
    --------
    dict
        Model performance metrics
    """
    # Build model
    model = build_lstm_model(
        (X_train.shape[1], X_train.shape[2]),
        first_lstm_units, second_lstm_units,
        dropout_rate, learning_rate
    )
    
    # Define callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=verbose
    )
    
    # Evaluate model
    # Make predictions
    y_pred_train = model.predict(X_train, verbose=0)
    y_pred_val = model.predict(X_val, verbose=0)
    y_pred_test = model.predict(X_test, verbose=0)
    
    # Denormalize
    std_y = norm_params['std_y']
    mean_y = norm_params['mean_y']
    
    y_pred_train_denorm = y_pred_train * std_y + mean_y
    y_pred_val_denorm = y_pred_val * std_y + mean_y
    y_pred_test_denorm = y_pred_test * std_y + mean_y
    
    y_train_denorm = y_train * std_y + mean_y
    y_val_denorm = y_val * std_y + mean_y
    y_test_denorm = y_test * std_y + mean_y
    
    # Calculate metrics
    train_metrics = calculate_metrics(y_train_denorm, y_pred_train_denorm)
    val_metrics = calculate_metrics(y_val_denorm, y_pred_val_denorm)
    test_metrics = calculate_metrics(y_test_denorm, y_pred_test_denorm)
    
    # Combine metrics
    metrics = {
        'train': train_metrics,
        'val': val_metrics,
        'test': test_metrics,
        'history': history.history
    }
    
    return metrics, model


def calculate_metrics(y_true, y_pred):
    """
    Calculate performance metrics.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
    
    Returns:
    --------
    dict
        Performance metrics
    """
    # Extract 1D arrays if needed
    if len(y_true.shape) > 1:
        y_true = y_true.flatten()
    if len(y_pred.shape) > 1:
        y_pred = y_pred.flatten()
    
    # Calculate metrics
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Calculate NSE (Nash-Sutcliffe Efficiency)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    ss_res = np.sum((y_true - y_pred)**2)
    nse = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan
    
    # Calculate KGE (Kling-Gupta Efficiency)
    r = np.corrcoef(y_true, y_pred)[0, 1]
    alpha = np.std(y_pred) / np.std(y_true) if np.std(y_true) != 0 else np.nan
    beta = np.mean(y_pred) / np.mean(y_true) if np.mean(y_true) != 0 else np.nan
    kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2) if not np.isnan(alpha) and not np.isnan(beta) else np.nan
    
    return {
        'MSE': mse,
        'R2': r2,
        'NSE': nse,
        'KGE': kge
    }


def train_model_for_multiple_lags(basin_data, feature_cols, target_col, 
                              hyperparams, lags, output_dir=None, verbose=1):
    """
    Train and evaluate LSTM models for multiple lookback periods.
    
    Parameters:
    -----------
    basin_data : pd.DataFrame
        DataFrame with features and target, including a date index
    feature_cols : list
        List of feature column names
    target_col : str
        Target column name
    hyperparams : dict
        Model hyperparameters
    lags : list
        List of lookback periods to evaluate
    output_dir : str, default=None
        Directory to save results
    verbose : int, default=1
        Verbosity level
    
    Returns:
    --------
    dict
        Model performance for each lag
    """
    results = {}
    
    for lag in lags:
        if verbose > 0:
            print(f"\nTraining model with lag: {lag} days")
        
        # Preprocess data
        X_train, X_val, X_test, y_train, y_val, y_test, norm_params = preprocess_data(
            basin_data, feature_cols, target_col, lag
        )
        
        # Train and evaluate model
        metrics, model = train_and_evaluate_model(
            X_train, y_train, X_val, y_val, X_test, y_test, norm_params,
            first_lstm_units=hyperparams.get('first_lstm_units', 50),
            second_lstm_units=hyperparams.get('second_lstm_units', 30),
            dropout_rate=hyperparams.get('dropout_rate', 0.2),
            learning_rate=hyperparams.get('learning_rate', 0.001),
            verbose=verbose
        )
        
        # Store results
        results[lag] = metrics
        
        if verbose > 0:
            print(f"Lag: {lag} days")
            print(f"  Training   - KGE: {metrics['train']['KGE']:.4f}, NSE: {metrics['train']['NSE']:.4f}")
            print(f"  Validation - KGE: {metrics['val']['KGE']:.4f}, NSE: {metrics['val']['NSE']:.4f}")
            print(f"  Test       - KGE: {metrics['test']['KGE']:.4f}, NSE: {metrics['test']['NSE']:.4f}")
        
        # Save metrics to CSV if output directory is specified
        if output_dir:
            # Create metrics DataFrame
            metrics_df = pd.DataFrame({
                'Metric': ['R2', 'MSE', 'NSE', 'KGE'],
                'Train': [metrics['train']['R2'], metrics['train']['MSE'], 
                         metrics['train']['NSE'], metrics['train']['KGE']],
                'Validation': [metrics['val']['R2'], metrics['val']['MSE'], 
                              metrics['val']['NSE'], metrics['val']['KGE']],
                'Test': [metrics['test']['R2'], metrics['test']['MSE'], 
                        metrics['test']['NSE'], metrics['test']['KGE']],
                'Lag': lag
            })
            
            # Transpose the DataFrame for the desired format
            metrics_df = metrics_df.set_index('Metric').transpose()
            
            # Ensure directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            # Save to CSV
            metrics_file = os.path.join(output_dir, f"metrics_lag_{lag}.csv")
            metrics_df.to_csv(metrics_file, index=True)
    
    return results


def analyze_basin(basin_file, hyperparams_file=None, feature_cols=['uaz_swe'], 
                target_col='OBS_RUN', lags=None, output_dir=None, verbose=1):
    """
    Analyze a basin using LSTM models.
    
    Parameters:
    -----------
    basin_file : str
        Path to CSV file with basin data
    hyperparams_file : str, default=None
        Path to CSV file with hyperparameters (optional)
    feature_cols : list, default=['uaz_swe']
        List of feature column names
    target_col : str, default='OBS_RUN'
        Target column name
    lags : list, default=None
        List of lookback periods to evaluate
    output_dir : str, default=None
        Directory to save results
    verbose : int, default=1
        Verbosity level
    
    Returns:
    --------
    dict
        Model performance for each lag
    """
    # Default lags if not specified
    if lags is None:
        lags = [7, 14, 30, 45, 60, 75, 90, 120, 150, 180, 210, 240, 270, 300, 330, 364]
    
    # Extract basin ID from filename
    basin_id = os.path.basename(basin_file).split('_')[0]
    
    if verbose > 0:
        print(f"Analyzing basin: {basin_id}")
    
    # Load basin data
    df = pd.read_csv(basin_file)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.replace(-999.0, pd.NA, inplace=True)
    df.dropna(inplace=True)
    
    # Ensure all required columns exist
    missing_cols = [col for col in feature_cols + [target_col] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in data: {missing_cols}")
    
    # Load hyperparameters if provided
    if hyperparams_file:
        hyperparams_df = pd.read_csv(hyperparams_file)
        basin_hyperparams = hyperparams_df[hyperparams_df['Data File'] == basin_id]
        
        if basin_hyperparams.empty:
            print(f"No hyperparameters found for basin {basin_id}. Using defaults.")
            hyperparams = {
                'first_lstm_units': 50,
                'second_lstm_units': 30,
                'dropout_rate': 0.2,
                'learning_rate': 0.001
            }
        else:
            hyperparams = {
                'first_lstm_units': int(basin_hyperparams['First LSTM Units'].values[0]),
                'second_lstm_units': int(basin_hyperparams['Second LSTM Units'].values[0]),
                'dropout_rate': float(basin_hyperparams['Dropout Rate'].values[0]),
                'learning_rate': float(basin_hyperparams['Learning Rate'].values[0])
            }
    else:
        # Use default hyperparameters
        hyperparams = {
            'first_lstm_units': 50,
            'second_lstm_units': 30,
            'dropout_rate': 0.2,
            'learning_rate': 0.001
        }
    
    # Create output directory if specified
    if output_dir:
        basin_output_dir = os.path.join(output_dir, basin_id)
        os.makedirs(basin_output_dir, exist_ok=True)
    else:
        basin_output_dir = None
    
    # Train models for multiple lags
    results = train_model_for_multiple_lags(
        df, feature_cols, target_col, hyperparams,
        lags, output_dir=basin_output_dir, verbose=verbose
    )
    
    # Compile results across lags
    compiled_results = compile_lag_results(results, lags)
    
    # Save compiled results if output directory is specified
    if basin_output_dir:
        compiled_file = os.path.join(basin_output_dir, "compiled_metrics.csv")
        compiled_results.to_csv(compiled_file, index=False)
        
        # Create plots
        create_performance_plots(compiled_results, basin_id, basin_output_dir)
    
    return results


def compile_lag_results(results, lags):
    """
    Compile results across multiple lags.
    
    Parameters:
    -----------
    results : dict
        Dictionary of model performance metrics for each lag
    lags : list
        List of lookback periods
    
    Returns:
    --------
    pd.DataFrame
        Compiled results
    """
    compiled_data = []
    
    for lag in lags:
        if lag in results:
            metrics = results[lag]
            compiled_data.append({
                'Lag': lag,
                'Train_R2': metrics['train']['R2'],
                'Train_MSE': metrics['train']['MSE'],
                'Train_NSE': metrics['train']['NSE'],
                'Train_KGE': metrics['train']['KGE'],
                'Validation_R2': metrics['val']['R2'],
                'Validation_MSE': metrics['val']['MSE'],
                'Validation_NSE': metrics['val']['NSE'],
                'Validation_KGE': metrics['val']['KGE'],
                'Test_R2': metrics['test']['R2'],
                'Test_MSE': metrics['test']['MSE'],
                'Test_NSE': metrics['test']['NSE'],
                'Test_KGE': metrics['test']['KGE']
            })
    
    return pd.DataFrame(compiled_data)


def create_performance_plots(results_df, basin_id, output_dir):
    """
    Create performance plots.
    
    Parameters:
    -----------
    results_df : pd.DataFrame
        DataFrame with performance metrics
    basin_id : str
        Basin ID
    output_dir : str
        Directory to save plots
    """
    # Create plot directory
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot KGE values
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['Lag'], results_df['Train_KGE'], 'o-', label='Training')
    plt.plot(results_df['Lag'], results_df['Validation_KGE'], 's-', label='Validation')
    plt.plot(results_df['Lag'], results_df['Test_KGE'], '^-', label='Test')
    plt.xlabel('Lookback Period (days)')
    plt.ylabel('KGE')
    plt.title(f'KGE Values for Different Lookback Periods - Basin {basin_id}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(plots_dir, f'{basin_id}_kge_by_lag.png'), dpi=300, bbox_inches='tight')
    
    # Plot NSE values
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['Lag'], results_df['Train_NSE'], 'o-', label='Training')
    plt.plot(results_df['Lag'], results_df['Validation_NSE'], 's-', label='Validation')
    plt.plot(results_df['Lag'], results_df['Test_NSE'], '^-', label='Test')
    plt.xlabel('Lookback Period (days)')
    plt.ylabel('NSE')
    plt.title(f'NSE Values for Different Lookback Periods - Basin {basin_id}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(plots_dir, f'{basin_id}_nse_by_lag.png'), dpi=300, bbox_inches='tight')
    
    # Plot MSE values
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['Lag'], results_df['Train_MSE'], 'o-', label='Training')
    plt.plot(results_df['Lag'], results_df['Validation_MSE'], 's-', label='Validation')
    plt.plot(results_df['Lag'], results_df['Test_MSE'], '^-', label='Test')
    plt.xlabel('Lookback Period (days)')
    plt.ylabel('MSE')
    plt.title(f'MSE Values for Different Lookback Periods - Basin {basin_id}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(plots_dir, f'{basin_id}_mse_by_lag.png'), dpi=300, bbox_inches='tight')
    
    plt.close('all')


def main():
    """Command-line interface for running LSTM models."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train and evaluate LSTM models for SWE-streamflow prediction')
    parser.add_argument('--basin_file', type=str, required=True, help='Path to basin data CSV file')
    parser.add_argument('--hyperparams_file', type=str, help='Path to hyperparameters CSV file')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--feature_cols', type=str, default='uaz_swe', help='Comma-separated feature columns')
    parser.add_argument('--target_col', type=str, default='OBS_RUN', help='Target column name')
    parser.add_argument('--lags', type=str, help='Comma-separated list of lookback periods')
    
    args = parser.parse_args()
    
    # Parse feature columns and lags
    feature_cols = args.feature_cols.split(',')
    lags = [int(lag) for lag in args.lags.split(',')] if args.lags else None
    
    # Analyze basin
    analyze_basin(
        args.basin_file,
        hyperparams_file=args.hyperparams_file,
        feature_cols=feature_cols,
        target_col=args.target_col,
        lags=lags,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()