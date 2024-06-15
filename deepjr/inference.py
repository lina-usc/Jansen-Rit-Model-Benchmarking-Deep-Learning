import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import LSTM, Dense, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras import initializers
import pickle
import xarray as xr
import pandas as pd
import random as python_random
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.preprocessing import MinMaxScaler
import scipy.stats


def reset_random_seeds(seed_value=1234):
    ''' Set random seeds for reproducibility. '''
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    python_random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
    # Configure TensorFlow settings if necessary (rarely needed)
    # tf.config.threading.set_intra_op_parallelism_threads(1)
    # tf.config.threading.set_inter_op_parallelism_threads(1)


def create_bi_lstm_model(input_shape, lstm_units=64, dropout_rate=0.1, 
                         dense_units=8):
    ''' Function to create the LSTM model. '''
    model = Sequential()

    # First layer, needs to return sequences for subsequent layers
    lstm_model = LSTM(lstm_units, return_sequences=False, dropout=dropout_rate,
                      kernel_initializer=initializers.GlorotUniform(seed=4287),
                      bias_initializer=initializers.Constant(0.001))
    bilstm_model = Bidirectional(lstm_model, input_shape=input_shape)
    model.add(bilstm_model)

    # Final output layer
    model.add(Dense(dense_units, activation='linear', 
                    kernel_initializer=initializers.GlorotUniform(seed=4287),
                    bias_initializer=initializers.Constant(0.001)))

    return model


def train_model(model, X_train, y_train, X_val, y_val, epochs,
                batch_size):
    ''' Function to compile and train the model. '''
    early_stop = EarlyStopping( monitor='val_loss',    # Monitor validation loss
                                min_delta=0.01,       # an improvement significant if it's greater than 0.001
                                patience=50,           # Number of epochs to wait after the last improvement
                                verbose=1,             # Print messages when stopping
                                mode='min',            # Stop training when the quantity monitored has stopped decreasing
                                restore_best_weights=True) # Restore model weights from the epoch with the best value of the monitored quantity.)
    
    model.compile(optimizer='adam', loss=snr_inv_loss_db, metrics=['mse'])
    history = model.fit(X_train, y_train, epochs=epochs,
                        batch_size=batch_size, callbacks=[early_stop],
                        validation_data=(X_val, y_val))
    return history


def snr_inv_loss_db(y_true, y_pred):
    # Calculate the signal power (mean squared value of the true signal)
    signal_power = tf.reduce_mean(tf.square(y_true))

    # Calculate the noise power (mean squared error)
    noise_power = tf.reduce_mean(tf.square(y_true - y_pred))

    # Calculate the inverse SNR
    snr_inv = noise_power / signal_power  # Inverse SNR
    
    # Convert inverse SNR to decibels
    snr_inv_db = 10 * tf.math.log(snr_inv) / tf.math.log(10.0) 

    return snr_inv_db


def estimate_parameters(parameter, noise, input_path, pred_path,
                        results_file_path, estimated_parameters,
                        nb_simulations=1000,
                        epochs=150, batch_size=32,
                        liftby=10):

    # Load the dataset
    noise_str = str(noise).replace(".", "_")
    fname = f'xarr_noise_{parameter}_{nb_simulations}_{noise_str}.nc'
    dataset = xr.open_dataset(input_path / fname)

    X = dataset["evoked"].transpose("sim_no", "time", "ch_names")
    X = X.squeeze().to_numpy()
    y = dataset["parameters"].sel(param=estimated_parameters)

    scaler = MinMaxScaler()

    # Scale the data: Fit and transform the scaler to only the specified parameter
    y_scaled = scaler.fit_transform(y)  

    # Create a DataFrame for the scaled data
    # y_scaled_df = pd.DataFrame(y_scaled)
        
    lift_scale = 10**liftby     # for avoiding gradient descent vanishing problem 


    #testing and training split
    X_train, X_test, y_train, y_test = train_test_split(X*lift_scale,
                                                        y_scaled, test_size=0.10,
                                                        random_state=68)
    
    #for validation split
    X_train, x_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      test_size=0.10,
                                                      random_state=68)
    
    # Create the LSTM model
    model = create_bi_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))

    # Train the model
    train_model(model, X_train, y_train, X_test, y_test, epochs=epochs, 
                batch_size=batch_size)
    
    # Evaluate the model
    loss_mse_results = model.evaluate(x_val, y_val, batch_size=batch_size)
    
    print(f'Results for {parameter}: Loss = {loss_mse_results[0]}, MSE = {loss_mse_results[1]}')
    
    # Generate predictions
    predictions = model.predict(x_val)
    predictions_original_scale = scaler.inverse_transform(predictions)

    y_val_original = scaler.inverse_transform(y_val)

    #save predictions for future debugging
    df_pred = pd.DataFrame()
    df_pred = pd.DataFrame(predictions_original_scale, columns=estimated_parameters)
    df_orig = pd.DataFrame(y_val_original, columns=estimated_parameters)
    # Convert predictions to a DataFrame

    # Save the DataFrame to a CSV file
    df_pred.to_csv(f'{pred_path}predictions_{parameter}.csv', index=False)
    df_orig .to_csv(f'{pred_path}orignal_{parameter}.csv', index=False)

    # Assuming df_pred and df_orig have the same columns for which you want to compute correlations
    column_correlations = {}

    for column in df_pred.columns:  # Loop through each column name
        # Calculate Spearman correlation for each column
        correlation, p_value = scipy.stats.spearmanr(df_pred[column], df_orig[column])
        
        # Prepare and save the results
        result_string = f"Correlation for {column}: r = {correlation:.3f}, p-value = {p_value:.3f}\n"
        column_correlations[column] = result_string

    with open(results_file_path, 'a') as file:    

        # You can now print or write these results to a file
        for column, result in column_correlations.items():
            
            result_string = f"-- For {noise}"

            file.write(result_string)
            file.write(result)    
    
    model.save(f"deepjr_{parameter}_{nb_simulations}_{noise_str}.keras")

    return model
