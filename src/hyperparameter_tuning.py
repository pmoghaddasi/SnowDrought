"""
Hyperparameter tuning for LSTM models used to predict streamflow from SWE data.

This module provides functions to optimize LSTM hyperparameters for streamflow prediction
using Snow Water Equivalent (SWE) as input. It uses Keras Tuner for hyperparameter search.
"""

import os
import time
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import keras_tuner as kt
from sklearn.model_selection import train_test_split


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
    all_cols = feature_cols + [target_col]
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


def build_lstm_tunable_model(hp, input_shape):
    """
    Build an LSTM model with tunable hyperparameters.
    
    Parameters:
    -----------
    hp : kt.HyperParameters
        Hyperparameter space
    input_shape : tuple
        Shape of input data (timesteps, features)
    
    Returns:
    --------
    tf.keras.Model
        Compiled LSTM model
    """
    model = Sequential()
    model.add(LSTM(
        units=hp.Int('units1', min_value=20, max_value=70, step=10),
        return_sequences=True,
        activation='tanh',
        input_shape=input_shape
    ))
    model.add(Dropout(rate=hp.Float('dropout_rate1', min_value=0.2, max_value=0.4, step=0.1)))
    
    model.add(LSTM(
        units=hp.Int('units2', min_value=10, max_value=50, step=10),
        return_sequences=False,
        activation='tanh'
    ))
    model.add(Dense(units=1))
    
    optimizer = Adam(learning_rate=hp.Choice('learning_rate', values=[0.0001, 0.001, 0.01]))
    model.compile(optimizer=optimizer, loss='mse')
    return model


def tune_hyperparameters(basin_data, feature_cols, target_col, lag, 
                        output_dir=None, max_trials=50, epochs=40, verbose=1):
    """
    Tune hyperparameters for an LSTM model.
    
    Parameters:
    -----------
    basin_data : pd.DataFrame
        DataFrame with features and target, including a date index
    feature_cols : list
        List of feature column names
    target_col : str
        Target column name
    lag : int
        Number of past time steps to use as input
    output_dir : str, default=None
        Directory to save tuning results
    max_trials : int, default=50
        Maximum number of tuning trials
    epochs : int, default=40
        Number of training epochs
    verbose : int, default=1
        Verbosity level
    
    Returns:
    --------
    dict
        Best hyperparameters and model performance
    """
    start_time = time.time()
    
    # Preprocess data
    X_train, X_val, X_test, y_train, y_val, y_test, norm_params = preprocess_data(
        basin_data, feature_cols, target_col, lag
    )
    
    # Define the hyperparameter search
    tuner = kt.RandomSearch(
        lambda hp: build_lstm_tunable_model(hp, (X_train.shape[1], X_train.shape[2])),
        objective='val_loss',
        max_trials=max_trials,
        executions_per_trial=3,
        directory=output_dir or 'tuning_results',
        project_name=f'lstm_tuning_lag_{lag}'
    )
    
    # Define callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )
    
    # Search for the best hyperparameters
    tuner.search(
        X_train, y_train,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=verbose
    )
    
    # Get best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    # Build model with best hyperparameters
    best_model = tuner.hypermodel.build(best_hps)
    
    # Train the model with the best hyperparameters
    history = best_model.fit(
        X_train, y_train,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=verbose
    )
    
    # Evaluate the best model
    test_loss = best_model.evaluate(X_test, y_test, verbose=0)
    
    # Get best hyperparameters
    best_params = {
        'first_lstm_units': best_hps.get('units1'),
        'second_lstm_units': best_hps.get('units2'),
        'dropout_rate': best_hps.get('dropout_rate1'),
        'learning_rate': best_hps.get('learning_rate'),
        'test_loss': test_loss,
        'tuning_time': time.time() - start_time
    }
    
    if verbose > 0:
        print(f"Best hyperparameters: {best_params}")
    
    return best_params


def tune_hyperparameters_for_basin(basin_file, feature_cols=['uaz_swe'], target_col='OBS_RUN', 
                                  lag=45, output_dir=None, max_trials=50):
    """
    Tune hyperparameters for a specific basin.
    
    Parameters:
    -----------
    basin_file : str
        Path to CSV file with basin data
    feature_cols : list, default=['uaz_swe']
        List of feature column names
    target_col : str, default='OBS_RUN'
        Target column name
    lag : int, default=45
        Number of past time steps to use as input
    output_dir : str, default=None
        Directory to save tuning results
    max_trials : int, default=50
        Maximum number of tuning trials
    
    Returns:
    --------
    dict
        Best hyperparameters and model performance
    """
    print(f"Tuning hyperparameters for basin: {basin_file}")
    
    # Load basin data
    df = pd.read_csv(basin_file)
    
    # Preprocess
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.replace(-999.0, pd.NA, inplace=True)
    df.dropna(inplace=True)
    
    # Ensure all required columns exist
    missing_cols = [col for col in feature_cols + [target_col] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in data: {missing_cols}")
    
    # Tune hyperparameters
    basin_id = os.path.basename(basin_file).split('_')[0]
    results = tune_hyperparameters(
        df, feature_cols, target_col, lag, 
        output_dir=output_dir, 
        max_trials=max_trials
    )
    
    # Add basin ID to results
    results['basin_id'] = basin_id
    
    return results


def main():
    """Command-line interface for hyperparameter tuning."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Tune LSTM hyperparameters for SWE-streamflow prediction')
    parser.add_argument('--basin_file', type=str, required=True, help='Path to basin data CSV file')
    parser.add_argument('--output_dir', type=str, default='tuning_results', help='Directory to save results')
    parser.add_argument('--feature_cols', type=str, default='uaz_swe', help='Comma-separated feature columns')
    parser.add_argument('--target_col', type=str, default='OBS_RUN', help='Target column name')
    parser.add_argument('--lag', type=int, default=45, help='Lookback period in days')
    parser.add_argument('--max_trials', type=int, default=50, help='Maximum number of tuning trials')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Parse feature columns
    feature_cols = args.feature_cols.split(',')
    
    # Tune hyperparameters
    results = tune_hyperparameters_for_basin(
        args.basin_file,
        feature_cols=feature_cols,
        target_col=args.target_col,
        lag=args.lag,
        output_dir=args.output_dir,
        max_trials=args.max_trials
    )
    
    # Save results
    output_file = os.path.join(args.output_dir, f"{os.path.basename(args.basin_file).split('.')[0]}_hyperparams.csv")
    pd.DataFrame([results]).to_csv(output_file, index=False)
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()