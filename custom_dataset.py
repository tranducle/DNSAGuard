import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

# --- Assume 'model.py' is in the same directory ---
# You need to import your model definitions
from model import DNSAGuard, GNN_Baseline, GRU_Baseline, MLP_Baseline

class SequenceGraphDataset(Dataset):
    """
    A custom PyTorch Dataset to wrap the preprocessed data dictionaries.
    """
    def __init__(self, data_dict):
        # data_dict is the loaded .pt file (e.g., train_data)
        self.sequences = data_dict['sequences']
        self.labels = data_dict['labels']
        self.client_ids = data_dict['client_ids']

    def __len__(self):
        # Return the total number of samples (clients)
        return len(self.labels)

    def __getitem__(self, idx):
        # Return the data for one sample
        return (
            self.sequences[idx],
            self.labels[idx],
            self.client_ids[idx]
        )