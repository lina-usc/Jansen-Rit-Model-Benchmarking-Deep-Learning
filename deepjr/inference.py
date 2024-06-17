import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import LSTM, Dense, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras import initializers
import joblib
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.preprocessing import MinMaxScaler
import scipy.stats
import seaborn as sns

from deepjr.simulation import SimResults


class JRInvModel:

    def __init__(self, nb_sims, noise_fact=0, path="./",
                 estim_params=('A_e', 'A_i', 'b_e', 'b_i',
                               'a_1', 'a_2', 'a_3', 'a_4'),
                 C=135, lift_by=10):
        self.estim_params = list(estim_params)
        self.C = C
        self.nb_sims = nb_sims
        self.noise_fact = noise_fact
        if noise_fact:
            if '.' in str(noise_fact):
                noise_factor = str(noise_fact).replace('.', '_')
            self.fname_model = f'deepjr_all_{nb_sims}_{noise_factor}.keras'
            self.fname_scaler = f'scaler_all_{nb_sims}_{noise_factor}.keras'
        else:
            self.fname_model = f'deepjr_all_{nb_sims}.keras'
            self.fname_scaler = f'scaler_all_{nb_sims}.keras'
        self.path = path

        self.model = None
        self.scaler = None
        self.X_test = None
        self.X_train = None
        self.X_val = None
        self.y_test = None
        self.y_train = None
        self.y_val = None
        self.lift_by = lift_by

    def load(self):
        self.model = tf.keras.models.load_model(self.path / self.fname_model)
        self.scaler = joblib.load(self.path / self.fname_scaler)

    def predict(self, X):
        X_scaled = self.__prepare_data__(X=X)[0]
        predicted_params = self.model.predict(X_scaled)
        predicted_params = self.scaler.inverse_transform(predicted_params)
        parameters = pd.DataFrame(predicted_params, columns=self.estim_params)

        if "C" not in self.estim_params:
            parameters["C"] = self.C
        return parameters

    def apply_scalings(self, X, y):
        if y is not None:
            if self.scaler is None:
                self.scaler = MinMaxScaler()
                self.scaler.fit(y)
            y_scaled = self.scaler.transform(y)
        else:
            y_scaled = None

        # X scaling is to help avoid gradient descent vanishing problem
        lift_scale = 10**self.lift_by

        return X*lift_scale, y_scaled

    def prepare_data(self):
        # Load the dataset
        self.sim_results = SimResults(self.nb_sims, self.noise_fact, self.path)
        self.sim_results.load()
        self.sim_results.clean()
        dataset = self.sim_results.dataset

        X = dataset["evoked"].transpose("sim_no", "time", "ch_names")
        X = X.squeeze().to_numpy()
        y = dataset["parameters"].sel(param=self.estim_params)

        X_scaled, y_scaled = self.apply_scalings(X, y)

        # Testing and training split
        split = train_test_split(X_scaled, y_scaled, test_size=0.10,
                                 random_state=68)
        X_train, self.X_test, y_train, self.y_test = split
        # For validation split
        split = train_test_split(X_train, y_train, test_size=0.10,
                                 random_state=68)
        self.X_train, self.X_val, self.y_train, self.y_val = split

    def __prepare_data__(self, X=None, y=None):
        if X is None:
            if self.X_test is None:
                assert RuntimeError("X_test is None. You need to train the "
                                    "model first. If you loaded a model, you"
                                    " need to pass values for the `X` and "
                                    "`y` arguments.")
            return self.X_test, self.y_test
        return self.apply_scalings(X, y)

    def assess_model(self, parameter, X=None, y=None, pred_path=None,
                     results_file_path=None, batch_size=32,
                     save_output=False):

        X_scaled, y_scaled = self.__prepare_data__(X, y)

        if pred_path is None:
            pred_path = self.path / 'predictions'
            pred_path.mkdir(exist_ok=True)

        if results_file_path is None:
            results_file_path = pred_path / "correlation_results.txt"

        if self.X_test is not None:
            # This evaluation cannot be done when the model has been loaded
            # from the disk. This if statement test this indirectly.

            # Evaluate the model
            loss_mse_results = self.model.evaluate(X_scaled, y_scaled,
                                                   batch_size=batch_size)

            print(f'Results for {parameter}: Loss = {loss_mse_results[0]}'
                  f', MSE = {loss_mse_results[1]}')

        # Generate predictions
        predictions = self.model.predict(X_scaled)
        predictions_original_scale = self.scaler.inverse_transform(predictions)

        y_val_original = self.scaler.inverse_transform(y_scaled)

        # Save predictions for future debugging
        df_pred = pd.DataFrame(predictions_original_scale,
                               columns=self.estim_params)
        df_orig = pd.DataFrame(y_val_original, columns=self.estim_params)
        # Convert predictions to a DataFrame

        # Assuming df_pred and df_orig have the same columns for which
        # you want to compute correlations
        column_correlations = {}
        correlations = []
        pvalues = []
        parameters = []
        for column in df_pred.columns:  # Loop through each column name
            # Calculate Spearman correlation for each column
            correlation, p_value = scipy.stats.spearmanr(df_pred[column],
                                                         df_orig[column])

            # Prepare and save the results
            result_string = f"Correlation for {column}: r = {correlation:.3f},"
            result_string += f" p-value = {p_value:.3f}\n"
            column_correlations[column] = result_string
            correlations.append(correlation)
            pvalues.append(p_value)
            parameters.append(column)

        if save_output:
            with open(results_file_path, 'a') as file:
                # You can now print or write these results to a file
                for column, result in column_correlations.items():
                    result_string = f"-- For {self.noise_fact}"
                    file.write(result_string)
                    file.write(result)

            # Save the DataFrame to a CSV file
            df_pred.to_csv(f'{pred_path}predictions_{parameter}.csv',
                           index=False)
            df_orig.to_csv(f'{pred_path}orignal_{parameter}.csv', index=False)

        return pd.DataFrame({"pvalue": pvalues, "correlation": correlations,
                             "parameter": parameters})

    @property
    def full_path_model(self):
        return self.path / self.fname_model

    @property
    def full_path_scaler(self):
        return self.path / self.fname_scaler

    def save(self):
        self.model.save(self.full_path_model)
        joblib.dump(self.scaler, self.full_path_scaler)

    def train_model(self, epochs=150, batch_size=32):
        ''' Function to compile and train the model. '''

        self.prepare_data()

        # Create the LSTM model
        self.model = create_bi_lstm_model(input_shape=self.X_train.shape[1:])

        early_stop = EarlyStopping(
            monitor='val_loss',    # Monitor validation loss
            min_delta=0.01,        # An improvement significant if it is
                                   # greater than 0.01
            patience=50,           # Number of epochs to wait after the
                                   # last improvement
            verbose=1,             # Print messages when stopping
            mode='min',            # Stop training when the quantity monitored
                                   # has stopped decreasing
            restore_best_weights=True  # Restore model weights from the epoch
                                       # with the best value of the monitored
                                       # quantity.)
        )

        self.model.compile(optimizer='adam',
                           #loss="mse",
                           loss=snr_inv_loss_db,
                           metrics=['mse'])
        self.history = self.model.fit(self.X_train, self.y_train,
                                      epochs=epochs, batch_size=batch_size,
                                      callbacks=[early_stop],
                                      validation_data=(self.X_val, self.y_val))

    def plot_test_regressions(self, X=None, y=None):
        pred = self.predict(X)
        pred.index.name = "test_sim_no"
        pred = pred.reset_index(drop=False)
        pred = pred.melt(id_vars=["test_sim_no"], var_name="param",
                         value_name="predicted")

        actual = pd.DataFrame(y, columns=self.estim_params)
        actual.index.name = "test_sim_no"
        actual = actual.reset_index(drop=False)
        actual = actual.melt(id_vars=["test_sim_no"],
                             var_name="param",
                             value_name="actual")

        combined_df = pd.merge(actual, pred)
        g = sns.lmplot(data=combined_df, col="param",
                       x="actual", y="predicted",
                       col_wrap=4, facet_kws=dict(sharey=False, sharex=False))
        for ax in g.axes.ravel():
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_xlim()

            vmin = min(xmin, ymin)
            vmax = max(xmax, ymax)
            ax.plot([vmin, vmax], [vmin, vmax], color="k", 
                    linestyle="dashed", alpha=0.5)


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
