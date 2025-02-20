import torch
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from transformers import TimeSeriesTransformerModel, TimeSeriesTransformerConfig
from sklearn.metrics import mean_squared_error
from transformers import Trainer, TrainingArguments
import seaborn as sns
import matplotlib.pyplot as plt

from deepjr.simulation import SimResults

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
        lift_scale = 10 ** 1
        return X * lift_scale, y_scaled

    def prepare_data(self):
        # Load the dataset
        self.sim_results = SimResults(self.nb_sims, self.noise_fact, self.path)
        self.sim_results.load()
        self.sim_results.clean()
        dataset = self.sim_results.dataset

        # Extract EEG ERP data and parameters
        X = dataset["evoked"].transpose("sim_no", "time", "ch_names").squeeze().to_numpy()

        # Check available parameters in the dataset
        available_params = dataset["parameters"].coords["param"].values
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

    def predict(self, model, X):
        X_scaled, _ = self.__prepare_data__(X=X)
        predictions = model.predict(X_scaled)
        predictions_original_scale = self.scaler.inverse_transform(predictions)
        return pd.DataFrame(predictions_original_scale, columns=self.estim_params)


class JRInvTimeSeriesTransformer:
    def __init__(self, nb_sims, path, estim_params=('A_e', 'A_i', 'b_e', 'b_i', 'a_1', 'a_2', 'a_3', 'a_4', 'C'), noise_fact=0):
        self.estim_params = estim_params
        self.nb_sims = nb_sims
        self.noise_fact = noise_fact
        self.path = Path(path)
        self.model = None
        self.data_loader = JRInvDataLoader(nb_sims, path, estim_params, noise_fact)

    def build_model(self, input_dim, seq_length, num_params):
        config = TimeSeriesTransformerConfig(
            prediction_length=seq_length,
            context_length=seq_length,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            dropout=0.1
        )
        self.model = TimeSeriesTransformerModel(config)
        self.regressor = torch.nn.Linear(64, num_params)

    def train_model(self, epochs=150, batch_size=32):
        self.data_loader.prepare_data()

        # Define optimizer and loss function
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        loss_fn = torch.nn.MSELoss()

        # Training loop
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0

            for i in range(0, len(self.data_loader.X_train), batch_size):
                # Get the batch
                x_batch = torch.tensor(self.data_loader.X_train[i:i+batch_size], dtype=torch.float32)
                y_batch = torch.tensor(self.data_loader.y_train[i:i+batch_size], dtype=torch.float32)

                # Zero gradients
                optimizer.zero_grad()

                # Forward pass
                output = self.model(x_batch)
                loss = loss_fn(output, y_batch)

                # Backward pass
                loss.backward()

                # Update weights
                optimizer.step()

                epoch_loss += loss.item()

            # Print progress
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss / len(self.data_loader.X_train):.4f}")

        print("Training finished.")

    def evaluate_model(self, X_test, y_test):
        self.model.eval()

        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

        with torch.no_grad():
            predictions = self.model(X_test_tensor)

        mse_loss = mean_squared_error(predictions.numpy(), y_test_tensor.numpy())
        print(f"Test MSE Loss: {mse_loss:.4f}")

        return mse_loss

    def plot_test_regressions(self, X_test, y_test):
        predictions = self.predict(X_test)
        actual = pd.DataFrame(y_test, columns=self.estim_params)

        combined_df = pd.merge(actual, predictions, left_index=True, right_index=True)
        g = sns.lmplot(data=combined_df, col="param", x="actual", y="predicted", col_wrap=4, facet_kws=dict(sharey=False, sharex=False))

        for ax in g.axes.ravel():
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_xlim()
            vmin = min(xmin, ymin)
            vmax = max(xmax, ymax)
            ax.plot([vmin, vmax], [vmin, vmax], color="k", linestyle="dashed", alpha=0.5)

"""


# Main code flow (to be used in a different notebook or script)
if __name__ == "__main__":
    path = "./deepjr_training_data"
    estim_params = ('A_e', 'A_i', 'b_e', 'b_i', 'a_1', 'a_2', 'a_3', 'a_4', 'C')
    nb_sims = 1000
    noise_fact = 0.5

    # Initialize the JRInvTimeSeriesTransformer and data loader
    transformer = JRInvTimeSeriesTransformer(nb_sims=nb_sims, path=path, estim_params=estim_params, noise_fact=noise_fact)

    # Prepare data and build the model
    transformer.data_loader.prepare_data()
    transformer.build_model(input_dim=transformer.data_loader.X_train.shape[1], 
                            seq_length=transformer.data_loader.X_train.shape[1], 
                            num_params=len(estim_params))

    # Train the model
    transformer.train_model(epochs=10, batch_size=32)

    # Evaluate the model
    transformer.evaluate_model(transformer.data_loader.X_test, transformer.data_loader.y_test)

    # Plot predictions vs actual values
    transformer.plot_test_regressions(transformer.data_loader.X_test, transformer.data_loader.y_test)
"""