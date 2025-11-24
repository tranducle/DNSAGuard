import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import csv
import pickle
import numpy as np
import pandas as pd
    "features": ['size', 'time'], 
    "batch_size": 2048,      
    "learning_rate": 0.0005,
    
    # --- 40 EPOCHS CHO TẤT CẢ ---
    "num_epochs": 40,       
    
    "hidden_dim": 128,
    "num_workers": 0
}

# Danh sách các đấu thủ
MODELS_TO_RUN = {
    "Transformer": TransformerClassifier, 
    "LSTM": LSTMBaseline,                 
    "GRU": GRUBaseline,                   
    "1D-CNN": CNNBaseline,                
    "MLP": MLPBaseline                    
}
# ==================================================================

class SupervisedDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
    def __len__(self): return len(self.sequences)
    def __getitem__(self, idx): return self.sequences[idx], self.labels[idx]

def load_data(config):
    print("Đang tải dữ liệu (all_clients_raw.pkl)...")
    try:
        with open('all_clients_raw.pkl', 'rb') as f:
            all_clients = pickle.load(f)
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy file dữ liệu. Hãy chạy pcap_loader.py trước.")
        exit()
        
    all_sequences, all_labels = [], []
    seq_len = config['sequence_length']

    print("Đang xử lý dữ liệu...")
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
            last_time = pkt_time
        all_sequences.append(seq_tensor)
        all_labels.append(data['label'])

    X = torch.stack(all_sequences).numpy()
    y = np.array(all_labels)
    
    # Tính tỷ lệ mất cân bằng thực tế
    neg = np.sum(y == 0)
    pos = np.sum(y == 1)
    ratio = neg / pos if pos > 0 else 1.0
    print(f"Dữ liệu: {len(y)} mẫu. Benign: {neg}, Malicious: {pos}. Ratio: {ratio:.2f}")

    # Scaling
    N, L, F = X.shape
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.reshape(-1, F)).reshape(N, L, F)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0)

    X_final = torch.tensor(X_scaled, dtype=torch.float)
    y_final = torch.tensor(y, dtype=torch.float)

    X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.2, random_state=42, stratify=y_final)
    return SupervisedDataset(X_train, y_train), SupervisedDataset(X_test, y_test), F, ratio

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for seqs, labels in loader:
        seqs, labels = seqs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(seqs)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for seqs, labels in loader:
            seqs, labels = seqs.to(device), labels.to(device)
            logits = model(seqs)
            preds = (torch.sigmoid(logits) > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    p = precision_score(all_labels, all_preds, zero_division=0)
    r = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    return acc, p, r, f1

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data và lấy ratio thực tế
    train_ds, test_ds, n_feats, ratio = load_data(CONFIG)
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=CONFIG['batch_size'], shuffle=False)
    
    # Thư mục lưu kết quả
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = f"comparison_results_{timestamp}"
    os.makedirs(result_dir, exist_ok=True)
    csv_path = f"{result_dir}/all_models_comparison.csv"
    
    print(f"Kết quả sẽ được lưu vào: {csv_path}")
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Epoch", "Time(m)", "TrainLoss", "Val_F1", "Val_Precision", "Val_Recall", "Val_Accuracy"])

    # --- VÒNG LẶP SO SÁNH ---
    print("\n========== BẮT ĐẦU SO SÁNH CÁC MODEL (40 Epochs) ==========")
    
    # Tính toán trọng số (Dynamic Weighting)
    pos_weight_val = max(1.0, min(ratio, 100.0)) # Clamp tại 100 để an toàn
    pos_weight = torch.tensor([pos_weight_val]).to(device)
    print(f"Sử dụng Class Weight: {pos_weight_val:.2f}")

    for model_name, ModelClass in MODELS_TO_RUN.items():
        print(f"\n>>> Training Model: {model_name} <<<")
        
        # Khởi tạo model
        model = ModelClass(
            num_features=n_feats, 
            seq_len=CONFIG['sequence_length'], 
            hidden_dim=CONFIG['hidden_dim']
        ).to(device)
             
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
        
        best_val_f1 = 0.0
        
        for epoch in range(CONFIG['num_epochs']):
            start = time.time()
            loss = train_epoch(model, train_loader, criterion, optimizer, device)
            acc, p, r, f1 = evaluate(model, test_loader, device)
            mins = (time.time() - start) / 60
            
            print(f"[{model_name}] Epoch {epoch+1:02} | Loss: {loss:.4f} | F1: {f1:.4f} (P:{p:.2f}, R:{r:.2f})")
            
            # Lưu log
            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([model_name, epoch+1, f"{mins:.2f}", f"{loss:.4f}", f"{f1:.4f}", f"{p:.4f}", f"{r:.4f}", f"{acc:.4f}"])
            
            # Lưu model tốt nhất
            if f1 > best_val_f1:
                best_val_f1 = f1
                torch.save(model.state_dict(), f"{result_dir}/best_{model_name}.pt")
        
        print(f"--> {model_name} hoàn tất. Best F1: {best_val_f1:.4f}")

    print(f"\nHoàn tất so sánh! Hãy kiểm tra thư mục: {result_dir}")