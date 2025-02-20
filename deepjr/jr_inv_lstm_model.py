import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from pathlib import Path
from deepjr.simulation import SimResults
import numpy as np
class JRInvLSTMModel(nn.Module):
    def __init__(self, input_dim, seq_length, num_params, estim_params, hidden_size=64, num_layers=2, dropout=0.1):
        super(JRInvLSTMModel, self).__init__()
        self.input_dim = input_dim
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.estim_params = estim_params  # Store estim_params as an attribute
        
        self.lstm = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(self.hidden_size, num_params)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        final_output = lstm_out[:, -1, :]  # Get output at the last time step
        output = self.fc(final_output)
        return output

    def train_model(self, X_train, y_train, epochs=150, batch_size=32, learning_rate=1e-2):
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        loss_fn = nn.MSELoss()

        for epoch in range(epochs):
            self.train()
            epoch_loss = 0

            for i in range(0, len(X_train), batch_size):
                x_batch = torch.tensor(X_train[i:i+batch_size], dtype=torch.float32)
                y_batch = torch.tensor(y_train[i:i+batch_size], dtype=torch.float32)

                optimizer.zero_grad()
                output = self(x_batch)
                loss = loss_fn(output, y_batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss / len(X_train):.4f}")

        print("Training finished.")

    def evaluate_model(self, X_test, y_test):
        self.eval()
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

        with torch.no_grad():
            predictions = self(X_test_tensor)

        mse_loss = F.mse_loss(predictions, y_test_tensor)
        print(f"Test MSE Loss: {mse_loss.item():.4f}")
        return mse_loss.item()

    def predict(self, X):
        self.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32)

        with torch.no_grad():
            predictions = self(X_tensor)

        predictions_np = predictions.numpy()
        return pd.DataFrame(predictions_np, columns=self.estim_params)
    
    def print_correlations(self, X_test, y_test, estim_params):
        # Get predictions and create DataFrames with appropriate column names
        predictions = self.predict(X_test)
        actual = pd.DataFrame(y_test, columns=estim_params)
        predictions_df = pd.DataFrame(predictions, columns=estim_params)
        
        # Dictionary to store correlation values for each parameter
        correlations = {}
        
        # Compute correlation for each parameter
        for param in estim_params:
            corr_value = actual[param].corr(predictions_df[param])
            correlations[param] = corr_value
            print(f"Pearson correlation for {param}: {corr_value}")
        
        return correlations

    
    

    def plot_test_regressions(self, X_test, y_test, estim_params):
        predictions = self.predict(X_test)  # Get predictions from the model
        actual = pd.DataFrame(y_test, columns=estim_params)  # Convert y_test to DataFrame

        # Create a combined DataFrame for actual vs predicted values
        combined_df = pd.DataFrame({
            'actual': actual.values.flatten(),  # Flatten actual values
            'predicted': predictions.values.flatten(),  # Flatten predictions
            'param': np.tile(estim_params, len(actual))  # Repeat parameter names for plotting
        })

        # Plot using seaborn's lmplot for regression
        g = sns.lmplot(data=combined_df, col="param", x="actual", y="predicted", col_wrap=4, facet_kws=dict(sharey=False, sharex=False))

        # Customize the plots
        for ax in g.axes.ravel():
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()

            vmin = min(xmin, ymin)
            vmax = max(xmax, ymax)
            ax.plot([vmin, vmax], [vmin, vmax], color="k", linestyle="dashed", alpha=0.5)



class JRInvDataLoader:
    def __init__(self, nb_sims, path, estim_params=('A_e', 'A_i', 'b_e', 'b_i', 'a_1', 'a_2', 'a_3', 'a_4', 'C'), noise_fact=0):
        self.estim_params = estim_params
        self.nb_sims = nb_sims
        self.path = Path(path)
        self.noise_fact = noise_fact
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.scaler = None

    def apply_scalings(self, X, y):
        if y is not None:
            if self.scaler is None:
                self.scaler = MinMaxScaler()
                self.scaler.fit(y)
            y_scaled = self.scaler.transform(y)
        else:
            y_scaled = None

        # X scaling to avoid gradient vanishing problem
        lift_scale = 10 ** 10

        return X * lift_scale, y_scaled

    def prepare_data(self):
        # Load the dataset
        self.sim_results = SimResults(self.nb_sims, self.noise_fact, self.path)  # Pass noise_fact here
        self.sim_results.load()
        self.sim_results.clean()
        dataset = self.sim_results.dataset

        # Extract EEG ERP data and parameters
        X = dataset["evoked"].transpose("sim_no", "time", "ch_names").squeeze().to_numpy()

        # Check available parameters in the dataset
        available_params = dataset["parameters"].coords["param"].values

        # Filter estim_params to only include those that exist in the dataset
        valid_estim_params = [param for param in self.estim_params if param in available_params]

        # Select only the valid parameters that exist in the dataset
        y = dataset["parameters"].sel(param=valid_estim_params)

        # Apply scaling
        X_scaled, y_scaled = self.apply_scalings(X, y)

        # Train-test split
        X_train, X_tmp, y_train, y_tmp = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=68)
        X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=68)

        # Save splits for later use
        self.X_train, self.X_val, self.X_test = X_train, X_val, X_test
        self.y_train, self.y_val, self.y_test = y_train, y_val, y_test

    def __prepare_data__(self, X=None, y=None):
        if X is None:
            if self.X_test is None:
                raise RuntimeError("X_test is None. Please prepare data first.")
            return self.X_test, self.y_test
        return self.apply_scalings(X, y)
