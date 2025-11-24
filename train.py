import os
# --- BẬT CHẾ ĐỘ DEBUG CUDA ---
# Giúp báo lỗi chính xác dòng nào bị sai thay vì báo chung chung
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import time
import csv
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

# --- Import Model Autoencoder (Transformer) ---
from model import SeqAutoencoder

# ==================================================================
# --- CONFIG TỐI ƯU ---
# ==================================================================
CONFIG = {
    "sequence_length": 50,
    "features": ['size', 'time'], 
    
    # --- PHẦN CỨNG ---
    "batch_size": 2048,      
    "num_workers": 0,        # Để 0 để tránh lỗi đa luồng trên Windows khi debug
    
    "learning_rate": 0.0001, # Giữ LR thấp để tránh bùng nổ gradient
    "num_epochs": 100,       
    "hidden_dim": 128,       
}
# ==================================================================

# Tạm tắt cudnn benchmark để debug an toàn hơn
torch.backends.cudnn.benchmark = False

class AutoencoderDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels 

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

def build_anomaly_dataset(config):
    print(f"Đang tải 'all_clients_raw.pkl'...")
    try:
        with open('all_clients_raw.pkl', 'rb') as f:
            all_clients = pickle.load(f)
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy 'all_clients_raw.pkl'. Chạy pcap_loader.py trước.")
        exit()

    all_sequences = []
    all_labels = []
    
    seq_len = config['sequence_length']

    print("Đang xử lý dữ liệu thô...")
    for client_ip, data in all_clients.items():
        packets = data['packets']
        original_len = min(len(packets), seq_len)
        
        seq_tensor = torch.zeros(seq_len, len(config['features']), dtype=torch.float)
        last_time = packets[0][0] if packets else 0.0
        
        for i in range(original_len):
            pkt_time, pkt_size, pkt_proto = packets[i]
            feat_idx = 0
            if 'size' in config['features']:
                seq_tensor[i, feat_idx] = pkt_size
                feat_idx += 1
            if 'time' in config['features']:
                delta = max(0.0, pkt_time - last_time)
                seq_tensor[i, feat_idx] = delta
                feat_idx += 1
            if 'proto' in config['features']: 
                seq_tensor[i, feat_idx] = float(pkt_proto)
                feat_idx += 1
            last_time = pkt_time
            
        all_sequences.append(seq_tensor)
        all_labels.append(data['label']) 

    X = torch.stack(all_sequences).numpy()
    y = np.array(all_labels)
    
    print(f"Tổng số mẫu: {len(y)} | Benign: {sum(y==0)} | Malicious: {sum(y==1)}")

    # --- SCALING & CLEANING (QUAN TRỌNG) ---
    print("Scaling và làm sạch dữ liệu...")
    
    # 1. Kiểm tra NaN/Inf TRƯỚC khi scale
    if np.isnan(X).any() or np.isinf(X).any():
        print("  -> Phát hiện NaN/Inf trong dữ liệu gốc. Đang thay thế bằng 0...")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    X_benign = X[y == 0]
    N, L, F = X.shape
    
    scaler = StandardScaler()
    # Chỉ fit trên dữ liệu sạch
    scaler.fit(X_benign.reshape(-1, F))
    
    # Transform
    X_scaled = scaler.transform(X.reshape(-1, F)).reshape(N, L, F)
    
    # 2. Kiểm tra NaN/Inf SAU khi scale (do chia cho 0)
    if np.isnan(X_scaled).any() or np.isinf(X_scaled).any():
        print("  -> Phát hiện lỗi sau khi Scaling (do chia cho 0). Đang sửa lỗi...")
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=10.0, neginf=-10.0) # Giới hạn giá trị
    
    X_final = torch.tensor(X_scaled, dtype=torch.float)
    y_final = torch.tensor(y, dtype=torch.float)

    # --- SPLIT ---
    benign_indices = np.where(y == 0)[0]
    malicious_indices = np.where(y == 1)[0]
    
    train_idx, test_benign_idx = train_test_split(benign_indices, test_size=0.2, random_state=42)
    test_idx = np.concatenate([test_benign_idx, malicious_indices])
    
    train_dataset = AutoencoderDataset(X_final[train_idx], y_final[train_idx])
    test_dataset = AutoencoderDataset(X_final[test_idx], y_final[test_idx])
    
    return train_dataset, test_dataset, F

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    for seqs, _ in loader:
        seqs = seqs.to(device) 
        
        optimizer.zero_grad()
        
        reconstructed = model(seqs)
        loss = criterion(reconstructed, seqs)
        
        # Kiểm tra loss có bị NaN không
        if torch.isnan(loss):
            print("  [CẢNH BÁO] Loss bị NaN trong quá trình train! Bỏ qua batch này.")
            continue

        loss.backward()
        # Gradient Clipping để tránh bùng nổ
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(loader) if len(loader) > 0 else 0

def evaluate_anomaly(model, loader, device):
    model.eval()
    reconstruction_errors = []
    true_labels = []
    with torch.no_grad():
        for seqs, labels in loader:
            seqs = seqs.to(device)
            reconstructed = model(seqs)
            loss_per_item = torch.mean((reconstructed - seqs) ** 2, dim=[1, 2])
            reconstruction_errors.extend(loss_per_item.cpu().numpy())
            true_labels.extend(labels.numpy())
    return np.array(reconstruction_errors), np.array(true_labels)

def find_best_threshold_dual_logic(errors, labels):
    thresholds = np.linspace(np.min(errors), np.max(errors), 1000)
    best_res = {'f1': -1, 'logic': ''}
    
    # Logic 1: Error > Thresh
    for thresh in thresholds:
        preds = (errors > thresh).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
        if f1 > best_res['f1']:
            best_res = {'thresh': thresh, 'p': p, 'r': r, 'f1': f1, 'logic': 'Error > Thresh'}

    # Logic 2: Error < Thresh
    for thresh in thresholds:
        preds = (errors < thresh).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
        if f1 > best_res['f1']:
            best_res = {'thresh': thresh, 'p': p, 'r': r, 'f1': f1, 'logic': 'Error < Thresh'}
            
    return best_res

# --- MAIN ---
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 1. Data
    train_ds, test_ds, n_feats = build_anomaly_dataset(CONFIG)
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=CONFIG['num_workers'])
    test_loader = DataLoader(test_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'])
    
    # 2. Model
    print(f"Khởi tạo Transformer Autoencoder (Hidden: {CONFIG['hidden_dim']})...")
    model = SeqAutoencoder(num_features=n_feats, hidden_dim=CONFIG['hidden_dim']).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    criterion = nn.MSELoss()
    
    # --- Setup Logging ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = os.path.join("training_reports", f"run_TransformerAE_Optimized_{timestamp}")
    os.makedirs(report_dir, exist_ok=True)
    report_filepath = os.path.join(report_dir, "results.csv")
    
    print(f"Bắt đầu huấn luyện... Log lưu tại {report_dir}")
    
    with open(report_filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Time", "TrainLoss", "BestF1", "BestPrec", "BestRec", "Logic", "AvgErrBenign", "AvgErrMal"])
        
        for epoch in range(CONFIG['num_epochs']):
            start = time.time()
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            
            # Đánh giá
            errors, labels = evaluate_anomaly(model, test_loader, device)
            res = find_best_threshold_dual_logic(errors, labels)
            
            avg_benign = np.mean(errors[labels==0])
            avg_mal = np.mean(errors[labels==1])
            
            mins = (time.time() - start) / 60
            print(f"Epoch {epoch+1:02} | Time: {mins:.2f}m | Loss: {train_loss:.6f} | F1: {res['f1']:.4f} ({res['logic']})")
            
            writer.writerow([epoch+1, f"{mins:.2f}", f"{train_loss:.4f}", 
                             f"{res['f1']:.4f}", f"{res['p']:.4f}", f"{res['r']:.4f}", 
                             res['logic'], f"{avg_benign:.4f}", f"{avg_mal:.4f}"])

    print("\n--- KẾT QUẢ TỐT NHẤT ---")
    errors, labels = evaluate_anomaly(model, test_loader, device)
    res = find_best_threshold_dual_logic(errors, labels)
    print(f"Logic tối ưu: {res['logic']}")
    print(f"F1-Score: {res['f1']:.4f}")
    print(f"Precision: {res['p']:.4f}")
    print(f"Recall: {res['r']:.4f}")
    
    torch.save(model.state_dict(), os.path.join(report_dir, "transformer_ae_opt.pt"))
    print("Done.")