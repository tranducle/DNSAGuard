import torch
import torch.nn as nn
import math
import torch.nn.functional as F

# --- BUILDING BLOCKS ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# ==================================================================
# 1. TRANSFORMER CLASSIFIER (DNSAGuard)
# ==================================================================
class TransformerClassifier(nn.Module):
    def __init__(self, num_features, hidden_dim=128, num_layers=3, nhead=4, dropout=0.2):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Linear(num_features, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim * 4, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1) 
        return self.classifier(x).squeeze(1)

# ==================================================================
# 2. GRU BASELINE (Cần cái này để vẽ t-SNE so sánh)
# ==================================================================
class GRUBaseline(nn.Module):
    def __init__(self, num_features, hidden_dim=128, num_layers=2, dropout=0.2):
        super(GRUBaseline, self).__init__()
        self.gru = nn.GRU(
            input_size=num_features, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True, dropout=dropout, bidirectional=True
        )
        # Bidirectional -> hidden_dim * 2
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        output, _ = self.gru(x)
        output = output.mean(dim=1)
        return self.classifier(output).squeeze(1)

# ==================================================================
# 3. CÁC BASELINE KHÁC (LSTM, CNN, MLP)
# ==================================================================
class LSTMBaseline(nn.Module):
    def __init__(self, num_features, hidden_dim=128, num_layers=2, dropout=0.2):
        super(LSTMBaseline, self).__init__()
        self.lstm = nn.LSTM(
            input_size=num_features, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True, dropout=dropout, bidirectional=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        output, _ = self.lstm(x)
        output = output.mean(dim=1) 
        return self.classifier(output).squeeze(1)

class CNNBaseline(nn.Module):
    def __init__(self, num_features, seq_len=50, hidden_dim=128, dropout=0.2):
        super(CNNBaseline, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(dropout)
        flatten_dim = 128 * (seq_len // 4) 
        self.classifier = nn.Sequential(
            nn.Linear(flatten_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1) 
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1) 
        return self.classifier(x).squeeze(1)

class MLPBaseline(nn.Module):
    def __init__(self, num_features, seq_len=50, hidden_dim=128, dropout=0.2):
        super(MLPBaseline, self).__init__()
        input_dim = num_features * seq_len
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1) 
        return self.net(x).squeeze(1)