
"""
File: eeg_graph_transformer.py

This file defines a graph-based EEG regression model (EEGGraphTransformer) and a data loader (JRInvDataLoader)
that loads EEG event-related potentials. The EEGGraphTransformer integrates a simple temporal projection
with a graph transformer layer (using PyTorch Geometric’s TransformerConv) to capture inter-channel (spatial)
dependencies. A helper function to create a simple fully-connected graph (without self-loops) is also provided.
"""

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
from pathlib import Path
from deepjr.simulation import SimResults

# Import necessary modules from PyTorch Geometric.
# Ensure you have installed torch-geometric and its dependencies:
# pip install torch-geometric
from torch_geometric.nn import TransformerConv

######################################
# Helper Function: Create Graph Connectivity
######################################

def create_edge_index(num_channels):
    """
    Create a fully connected graph (without self-loops) for the given number of channels.
    
    Parameters:
    - num_channels: Number of nodes (EEG channels)
    
    Returns:
    - edge_index: Tensor of shape (2, num_edges) in COO format.
    """
    edge_index = []
    for i in range(num_channels):
        for j in range(num_channels):
            if i != j:
                edge_index.append([i, j])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # shape: (2, num_edges)
    return edge_index

######################################
# EEGGraphTransformer Model for Regression
######################################

class EEGGraphTransformer(nn.Module):
    def __init__(self, num_channels, num_timepoints, output_dim, estim_params,
                 embed_dim, num_heads, graph_embed_dim, dropout=0.1):
        """
        Combines temporal and graph-based spatial processing for EEG regression.
        
        Parameters:
        - num_channels: number of EEG channels (nodes)
        - num_timepoints: number of time steps per sample
        - output_dim: number of regression targets
        - estim_params: list/tuple of parameter names for outputs
        - embed_dim: embedding dimension for temporal projection (each channel's time series)
        - num_heads: number of attention heads for the graph transformer layer
        - graph_embed_dim: output embedding dimension from the graph transformer per head
        - dropout: dropout probability
        """
        super(EEGGraphTransformer, self).__init__()
        self.estim_params = list(estim_params)
        self.num_channels = num_channels
        self.num_timepoints = num_timepoints

        # Temporal processing: project each channel's time series (treating timepoints as features)
        self.input_proj = nn.Linear(num_timepoints, embed_dim)
        
        # Graph Transformer: process spatial (channel) interactions.
        self.graph_conv = TransformerConv(in_channels=embed_dim,
                                          out_channels=graph_embed_dim,
                                          heads=num_heads,
                                          dropout=dropout)
        
        # Readout layer for regression: combines the multi-head outputs.
        self.readout = nn.Linear(graph_embed_dim * num_heads, output_dim)
        
    def forward(self, X, edge_index):
        """
        Forward pass.
        
        Parameters:
        - X: Tensor of shape (batch_size, num_channels, num_timepoints)
        - edge_index: Graph connectivity in COO format, shape (2, num_edges)
        
        Returns:
        - output: Tensor of shape (batch_size, output_dim)
        """
        batch_size = X.size(0)
        # Project each channel's time series: resulting shape (batch_size, num_channels, embed_dim)
        X_proj = self.input_proj(X)
        
        outputs = []
        # Process each example in the batch individually.
        for i in range(batch_size):
            node_features = X_proj[i]  # shape: (num_channels, embed_dim)
            # Apply graph transformer layer; output shape: (num_channels, graph_embed_dim * num_heads)
            node_out = self.graph_conv(node_features, edge_index)
            # Aggregate node features (mean pooling) to get a global representation.
            graph_rep = node_out.mean(dim=0)  # shape: (graph_embed_dim * num_heads,)
            outputs.append(graph_rep)
        
        outputs = torch.stack(outputs, dim=0)  # shape: (batch_size, graph_embed_dim * num_heads)
        output = self.readout(outputs)  # shape: (batch_size, output_dim)
        return output

######################################
# Data Loader: JRInvDataLoader for EEG Data
######################################

class JRInvDataLoader:
    def __init__(self, nb_sims, path, estim_params=('A_e', 'A_i', 'b_e', 'b_i', 'a_1', 'a_2', 'a_3', 'a_4', 'C'), noise_fact=0):
        """
        Data loader for EEG simulation data.
        
        Parameters:
        - nb_sims: number of simulations to load
        - path: directory path containing simulation data
        - estim_params: tuple of target parameter names
        - noise_fact: noise factor for simulation data
        """
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
        lift_scale = 1e2  # Adjust lift scale if needed
        return X * lift_scale, y_scaled

    def prepare_data(self):
        """
        Loads and preprocesses the simulation data.
        Splits the data into training, validation, and test sets.
        """
        self.sim_results = SimResults(self.nb_sims, self.noise_fact, self.path)
        self.sim_results.load()
        self.sim_results.clean()
        dataset = self.sim_results.dataset
        # Assumes dataset["evoked"] has dimensions: ("sim_no", "time", "ch_names")
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
        """
        Helper to apply scaling if data is directly provided.
        """
        if X is None:
            if self.X_test is None:
                raise RuntimeError("X_test is None. Please prepare data first.")
            return self.X_test, self.y_test
        return self.apply_scalings(X, y)

######################################
# Main Code Flow (Example Usage)
######################################

"""

if __name__ == "__main__":
    # Define parameters and paths.
    path = "./deepjr_training_data"  # Adjust the path to your training data.
    estim_params = ('A_e', 'A_i', 'b_e', 'b_i', 'a_1', 'a_2', 'a_3', 'a_4', 'C')
    nb_sims = 1000
    noise_fact = 0.5

    # Initialize and prepare the Data Loader.
    data_loader = JRInvDataLoader(nb_sims=nb_sims, path=path, estim_params=estim_params, noise_fact=noise_fact)
    data_loader.prepare_data()

    # Extract dimensions from the training data.
    num_channels = data_loader.X_train.shape[1]
    num_timepoints = data_loader.X_train.shape[2]
    output_dim = len(estim_params)

    # Create graph connectivity (edge_index) for the EEG channels.
    edge_index = create_edge_index(num_channels)

    # Initialize the EEGGraphTransformer model.
    # Adjust embed_dim, num_heads, and graph_embed_dim based on your needs.
    model = EEGGraphTransformer(num_channels=num_channels,
                                num_timepoints=num_timepoints,
                                output_dim=output_dim,
                                estim_params=estim_params,
                                embed_dim=64,        # Temporal embedding dimension.
                                num_heads=4,
                                graph_embed_dim=32,  # Graph transformer output dimension per head.
                                dropout=0.1)

    # Define optimizer and loss function.
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # --- Training Loop (Simplified Example) ---
    epochs = 10
    batch_size = 32
    num_samples = data_loader.X_train.shape[0]
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(num_samples)
        epoch_loss = 0.0
        for i in range(0, num_samples, batch_size):
            indices = permutation[i:i+batch_size]
            x_batch = torch.tensor(data_loader.X_train[indices], dtype=torch.float32)
            y_batch = torch.tensor(data_loader.y_train[indices], dtype=torch.float32)
            optimizer.zero_grad()
            # Forward pass: pass both EEG data and edge_index.
            outputs = model(x_batch, edge_index)
            loss = loss_fn(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / (num_samples / batch_size)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    # --- Evaluation Example ---
    model.eval()
    X_test_tensor = torch.tensor(data_loader.X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(data_loader.y_test, dtype=torch.float32)
    with torch.no_grad():
        predictions = model(X_test_tensor, edge_index)
    mse_loss = F.mse_loss(predictions, y_test_tensor)
    print(f"Test MSE Loss: {mse_loss.item():.4f}")

    # --- Plotting Example for the first parameter ---
    predictions_np = predictions.numpy()
    y_test_np = data_loader.y_test
    param = estim_params[0]
    plt.figure()
    sns.regplot(x=y_test_np[:, 0], y=predictions_np[:, 0])
    plt.xlabel("Actual " + param)
    plt.ylabel("Predicted " + param)
    plt.title(f"Actual vs Predicted for {param}")
    plt.plot([y_test_np[:, 0].min(), y_test_np[:, 0].max()],
             [y_test_np[:, 0].min(), y_test_np[:, 0].max()],
             'r--')
    plt.show()

"""
