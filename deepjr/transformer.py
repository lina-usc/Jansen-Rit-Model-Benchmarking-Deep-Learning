import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from pathlib import Path
from deepjr.simulation import SimResults

#############################
# EEGTransformer Model for Regression
#############################

class EEGTransformer(nn.Module):
    def __init__(self, num_channels, num_timepoints, output_dim, estim_params,
                 embed_dim, num_heads, hidden_dim=128,
                 hidden_ffn_dim=256, intermediate_dim=512, ffn_output_dim=128, dropout=0.1):
        """
        EEGTransformer for regression with attention pooling and active feature gating.
        
        Parameters:
        - num_channels: number of EEG channels (input features)
        - num_timepoints: number of time steps per sample
        - output_dim: number of regression targets (should equal len(estim_params))
        - estim_params: list/tuple of parameter names for outputs
        - embed_dim: embedding dimension (must be divisible by num_heads)
        - num_heads: number of attention heads
        - hidden_dim: (not used directly; kept for compatibility)
        - hidden_ffn_dim, intermediate_dim, ffn_output_dim: dimensions for the feed-forward network
        - dropout: dropout probability
        """
        super(EEGTransformer, self).__init__()
        self.estim_params = list(estim_params)
        self.num_channels = num_channels
        self.num_timepoints = num_timepoints
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        # Project input from num_channels to embed_dim
        self.input_proj = nn.Linear(num_channels, embed_dim)
        
        # Active Feature Gate: a learnable gating vector to select "active" features.
        # It will have shape (embed_dim) and is passed through a sigmoid.
        self.active_feature_gate = nn.Parameter(torch.ones(embed_dim))

        # Create positional encoding. We want the encoding to have shape (num_timepoints, embed_dim)
        self.positional_encoding = torch.zeros(embed_dim, num_timepoints)
        for j in range(embed_dim):
            for k in range(num_timepoints):
                if j % 2 == 0:
                    self.positional_encoding[j, k] = torch.sin(torch.tensor(k, dtype=torch.float32) / (10000 ** (j / embed_dim)))
                else:
                    self.positional_encoding[j, k] = torch.cos(torch.tensor(k, dtype=torch.float32) / (10000 ** ((j - 1) / embed_dim)))
        # Transpose so positional encoding shape becomes (num_timepoints, embed_dim)
        self.positional_encoding = self.positional_encoding.transpose(0, 1)

        # Multi-Head Self-Attention using embed_dim
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)

        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, ffn_output_dim)
        )

        # Layer Normalizations
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Final output layer for regression
        self.output_layer = nn.Linear(ffn_output_dim, output_dim)

    def forward(self, X):
        """
        Forward pass.
        X: Tensor of shape (batch_size, num_channels, num_timepoints)
        """
        batch_size = X.size(0)
        # Project input: (batch_size, num_timepoints, embed_dim)
        X_proj = self.input_proj(X.transpose(1, 2))
        
        # Active Feature Gating: learn which embedding dimensions are "active"
        gate = torch.sigmoid(self.active_feature_gate)  # shape: (embed_dim)
        # Multiply projected features by the gating vector (broadcasted over batch and time)
        X_proj = X_proj * gate.unsqueeze(0).unsqueeze(0)
        
        # Add positional encoding: shape (batch_size, num_timepoints, embed_dim)
        X_proj = X_proj + self.positional_encoding.to(X.device).unsqueeze(0)
        
        # Permute for transformer: (num_timepoints, batch_size, embed_dim)
        X_proj = X_proj.permute(1, 0, 2)
        
        # Multi-Head Self-Attention
        attn_output, _ = self.multihead_attn(X_proj, X_proj, X_proj)
        attn_output = self.norm1(attn_output)
        
        # Feed-Forward Network with residual connection
        ff_output = self.ffn(attn_output)
        ff_output = self.norm2(ff_output + attn_output)
        
        # Attention Pooling: Aggregate over time dimension (mean pooling here)
        pooled = ff_output.mean(dim=0)  # shape: (batch_size, embed_dim)
        
        # Final regression output
        output = self.output_layer(pooled)  # shape: (batch_size, output_dim)
        return output


    #############################
    # Training, Evaluation, Prediction, Plotting, and Correlation Methods
    #############################

    def train_model(self, X_train, y_train, X_val, y_val, epochs=150, batch_size=32, learning_rate=1e-3):
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        loss_fn = nn.MSELoss()
        num_samples = X_train.shape[0]
        
        for epoch in range(epochs):
            self.train()
            epoch_loss = 0.0
            permutation = torch.randperm(num_samples)
            for i in range(0, num_samples, batch_size):
                indices = permutation[i:i+batch_size]
                x_batch = torch.tensor(X_train[indices], dtype=torch.float32)
                y_batch = torch.tensor(y_train[indices], dtype=torch.float32)
                optimizer.zero_grad()
                outputs = self(x_batch)
                loss = loss_fn(outputs, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / (num_samples / batch_size)
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
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
        return pd.DataFrame(predictions.numpy(), columns=self.estim_params)
    
    def plot_test_regressions(self, X_test, y_test):
        predictions_df = self.predict(X_test)
        actual_df = pd.DataFrame(y_test, columns=self.estim_params)
        for param in self.estim_params:
            plt.figure()
            sns.regplot(x=actual_df[param], y=predictions_df[param])
            plt.xlabel("Actual " + param)
            plt.ylabel("Predicted " + param)
            plt.title(f"Actual vs Predicted for {param}")
            plt.plot([actual_df[param].min(), actual_df[param].max()],
                     [actual_df[param].min(), actual_df[param].max()],
                     'r--')
            plt.show()
    
    def print_correlations(self, X_test, y_test):
        predictions_df = self.predict(X_test)
        actual_df = pd.DataFrame(y_test, columns=self.estim_params)
        corr_dict = {}
        for param in self.estim_params:
            corr = actual_df[param].corr(predictions_df[param])
            corr_dict[param] = corr
            print(f"Correlation for {param}: {corr:.4f}")
        return corr_dict

#############################
# Data Loader for EEG (Preprocessing Only)
#############################

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
        lift_scale = 10 ** 2 #putting it 1 because it is not effecting results. 
        return X * lift_scale, y_scaled

    def prepare_data(self):
        self.sim_results = SimResults(self.nb_sims, self.noise_fact, self.path)
        self.sim_results.load()
        self.sim_results.clean()
        dataset = self.sim_results.dataset
        X = dataset["evoked"].transpose("sim_no", "time", "ch_names").squeeze().to_numpy()
        available_params = dataset["parameters"].coords["param"].values
        valid_estim_params = [param for param in self.estim_params if param in available_params]
        y = dataset["parameters"].sel(param=valid_estim_params)
        X_scaled, y_scaled = self.apply_scalings(X, y)
        X_train, X_tmp, y_train, y_tmp = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=68)
        X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=68)
        self.X_train, self.X_val, self.X_test = X_train, X_val, X_test
        self.y_train, self.y_val, self.y_test = y_train, y_val, y_test

    def __prepare_data__(self, X=None, y=None):
        if X is None:
            if self.X_test is None:
                raise RuntimeError("X_test is None. Please prepare data first.")
            return self.X_test, self.y_test
        return self.apply_scalings(X, y)

#############################
# Main Code Flow
#############################
"""

if __name__ == "__main__":
    path = "./deepjr_training_data"
    estim_params = ('A_e', 'A_i', 'b_e', 'b_i', 'a_1', 'a_2', 'a_3', 'a_4', 'C')
    nb_sims = 1000
    noise_fact = 0.5

    # Initialize the Data Loader
    data_loader = JRInvDataLoader(nb_sims=nb_sims, path=path, estim_params=estim_params, noise_fact=noise_fact)
    data_loader.prepare_data()

    # Get dimensions: assume X_train shape is (num_samples, num_channels, num_timepoints)
    num_channels = data_loader.X_train.shape[1]
    num_timepoints = data_loader.X_train.shape[2]

    # Initialize the EEGTransformer model (for regression)
    model = EEGTransformer(num_channels=num_channels, num_timepoints=num_timepoints,
                             output_dim=len(estim_params), estim_params=estim_params,
                             embed_dim=128, num_heads=4,
                             hidden_dim=128, hidden_ffn_dim=256,
                             intermediate_dim=512, ffn_output_dim=128, dropout=0.1)

    # Train the model
    model.train_model(data_loader.X_train, data_loader.y_train, data_loader.X_val, data_loader.y_val, epochs=10, batch_size=32)

    # Evaluate the model
    model.evaluate_model(data_loader.X_test, data_loader.y_test)

    # Plot predictions vs. actual values
    model.plot_test_regressions(data_loader.X_test, data_loader.y_test)

    # Print correlations between predictions and actual values
    model.print_correlations(data_loader.X_test, data_loader.y_test)

"""