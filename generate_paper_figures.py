import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import pickle
import sys
import traceback
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Import các class mô hình
from model import TransformerClassifier, LSTMBaseline, GRUBaseline, CNNBaseline, MLPBaseline

# --- CẤU HÌNH ---
OUTPUT_DIR = "paper_figures_advanced" # Thư mục kết quả
if os.path.exists(OUTPUT_DIR):
    import shutil
    # Xóa thư mục cũ để đảm bảo hình ảnh mới được tạo ra
    try:
        shutil.rmtree(OUTPUT_DIR)
    except: pass
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Dùng CPU cho việc vẽ để tránh lỗi CUDA OOM hoặc xung đột driver
DEVICE = torch.device("cpu") 
print(f"--- ĐANG CHẠY CHẾ ĐỘ VẼ HÌNH (DEVICE: {DEVICE}) ---")

CONFIG = {
    "sequence_length": 50,
    "features": ['size', 'time'], 
    "hidden_dim": 128,
    "test_size": 0.2
}

# Tìm thư mục kết quả mới nhất chứa các file best_*.pt
result_dirs = glob.glob("comparison_results_*")
if not result_dirs:
    print("LỖI: Không tìm thấy thư mục kết quả 'comparison_results_*'!")
    sys.exit(1)
    
latest_dir = max(result_dirs, key=os.path.getmtime)
print(f"--> Đang lấy model từ thư mục: {latest_dir}")

# Kiểm tra xem có file model không
model_files = glob.glob(os.path.join(latest_dir, "best_*.pt"))
if not model_files:
    print("LỖI: Thư mục kết quả không chứa file .pt nào!")
    sys.exit(1)
else:
    print(f"--> Tìm thấy {len(model_files)} models: {[os.path.basename(x) for x in model_files]}")

# --- 1. HÀM LOAD DỮ LIỆU CHUẨN XÁC TỪ PKL ---
def load_data_robust():
    pkl_file = 'all_clients_raw.pkl'
    print(f"\n[1/4] Đang tải dữ liệu gốc '{pkl_file}'...")
    
    if not os.path.exists(pkl_file):
        print(f"LỖI: Không tìm thấy file '{pkl_file}'.")
        sys.exit(1)

    with open(pkl_file, 'rb') as f:
        all_clients = pickle.load(f)

    all_sequences, all_labels = [], []
    seq_len = CONFIG['sequence_length']

    # Xử lý dữ liệu
    for client_ip, data in all_clients.items():
        packets = data['packets']
        original_len = min(len(packets), seq_len)
        seq_tensor = torch.zeros(seq_len, len(CONFIG['features']), dtype=torch.float)
        last_time = packets[0][0] if packets else 0.0
        
        for i in range(original_len):
            pkt_time, pkt_size, pkt_proto = packets[i]
            feat_idx = 0
            if 'size' in CONFIG['features']:
                seq_tensor[i, feat_idx] = pkt_size
                feat_idx += 1
            if 'time' in CONFIG['features']:
                delta = max(0.0, pkt_time - last_time)
                seq_tensor[i, feat_idx] = delta
                feat_idx += 1
            last_time = pkt_time
            
        all_sequences.append(seq_tensor)
        all_labels.append(data['label'])

    X = torch.stack(all_sequences).numpy()
    y = np.array(all_labels)

    # Scaling
    print("      Đang chuẩn hóa dữ liệu...")
    N, L, F = X.shape
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.reshape(-1, F)).reshape(N, L, F)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0)

    X_final = torch.tensor(X_scaled, dtype=torch.float)
    y_final = torch.tensor(y, dtype=torch.float)

    # Split để lấy tập Test chuẩn
    _, X_test, _, y_test = train_test_split(X_final, y_final, test_size=CONFIG['test_size'], random_state=42, stratify=y_final)
    
    print(f"      Đã xong! Số mẫu Test: {len(y_test)} (Benign: {sum(y_test==0)}, Malicious: {sum(y_test==1)})")
    return X_test, y_test

# --- 2. TRÍCH XUẤT ĐẶC TRƯNG ---
def get_embeddings(model, X, model_type):
    model.eval()
    model.to(DEVICE)
    X = X.to(DEVICE)
    
    with torch.no_grad():
        if model_type == "Transformer":
            # Xử lý tên biến linh hoạt
            if hasattr(model, 'embedding'): x = model.embedding(X)
            elif hasattr(model, 'input_proj'): x = model.input_proj(X)
            else: raise AttributeError("Model Transformer không có lớp embedding/input_proj")
            
            x = model.pos_encoder(x)
            x = model.transformer_encoder(x)
            emb = x.mean(dim=1)
        elif model_type == "GRU":
            output, _ = model.gru(X)
            emb = output.mean(dim=1)
        else:
            return None
    return emb.cpu().numpy()

# --- 3. VẼ T-SNE ---
def plot_tsne(X_test, y_test, n_samples=2000):
    print("\n[2/4] Đang vẽ t-SNE (Fig 5)...")
    
    # Lấy mẫu cân bằng
    idx_mal = (y_test == 1).nonzero(as_tuple=True)[0]
    idx_ben = (y_test == 0).nonzero(as_tuple=True)[0]
    
    n_take = min(len(idx_mal), len(idx_ben), n_samples // 2)
    n_take = max(n_take, 50) # Ít nhất 50
    
    sel_mal = idx_mal[torch.randperm(len(idx_mal))[:n_take]]
    sel_ben = idx_ben[torch.randperm(len(idx_ben))[:n_take]]
    indices = torch.cat([sel_mal, sel_ben])
    
    X_sub = X_test[indices]
    y_sub = y_test[indices]

    models = ["Transformer", "GRU"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    for i, m_name in enumerate(models):
        fpath = os.path.join(latest_dir, f"best_{m_name}.pt")
        if not os.path.exists(fpath): continue
        
        # Init model
        if m_name == "Transformer":
            model = TransformerClassifier(len(CONFIG['features']), CONFIG['hidden_dim'])
            title = "DNSAGuard (Ours)"
        else:
            model = GRUBaseline(len(CONFIG['features']), CONFIG['hidden_dim'])
            title = "GRU Baseline"
            
        model.load_state_dict(torch.load(fpath, map_location=DEVICE))
        embeddings = get_embeddings(model, X_sub, m_name)
        
        # t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000, init='pca', learning_rate='auto')
        res = tsne.fit_transform(embeddings)
        
        df_plot = pd.DataFrame({
            'x': res[:,0], 'y': res[:,1], 
            'Label': ['Malicious' if v==1 else 'Benign' for v in y_sub.cpu().numpy()]
        })
        
        sns.scatterplot(data=df_plot, x='x', y='y', hue='Label', 
                        palette={'Benign': '#2ecc71', 'Malicious': '#e74c3c'},
                        alpha=0.7, s=60, ax=axes[i])
        axes[i].set_title(f"{title} Feature Space", fontsize=16, fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "Fig5_tSNE.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"      -> Đã lưu: {save_path}")

# --- 4. VẼ CONFUSION MATRIX GRID ---
def plot_cm_grid(X_test, y_test):
    print("\n[3/4] Đang vẽ Confusion Matrix Grid (Fig 6)...")
    y_true = y_test.cpu().numpy()
    
    # Tìm tất cả model
    files = glob.glob(os.path.join(latest_dir, "best_*.pt"))
    model_map = {
        'Transformer': TransformerClassifier, 'LSTM': LSTMBaseline, 
        'GRU': GRUBaseline, '1D-CNN': CNNBaseline, 'MLP': MLPBaseline
    }
    
    valid_files = []
    for f in files:
        name = os.path.basename(f).replace("best_", "").replace(".pt", "")
        if name in model_map: valid_files.append((name, f))
    
    # Sắp xếp Transformer lên đầu
    valid_files.sort(key=lambda x: x[0] == 'Transformer', reverse=True)
    
    cols = 3
    rows = (len(valid_files) + 2) // 3
    plt.figure(figsize=(5 * cols, 4 * rows))
    
    # Xử lý batch để tránh tràn RAM (kể cả CPU cũng nên batch)
    batch_size = 1000
    
    for i, (m_name, fpath) in enumerate(valid_files):
        ModelClass = model_map[m_name]
        if m_name in ["1D-CNN", "MLP"]:
            model = ModelClass(len(CONFIG['features']), seq_len=CONFIG['sequence_length'], hidden_dim=CONFIG['hidden_dim']).to(DEVICE)
        else:
            model = ModelClass(len(CONFIG['features']), hidden_dim=CONFIG['hidden_dim']).to(DEVICE)
            
        model.load_state_dict(torch.load(fpath, map_location=DEVICE))
        model.eval()
        
        all_preds = []
        with torch.no_grad():
            for j in range(0, len(X_test), batch_size):
                batch_X = X_test[j:j+batch_size].to(DEVICE)
                logits = model(batch_X)
                preds = (torch.sigmoid(logits) > 0.5).float().cpu().numpy()
                all_preds.extend(preds)
                
        cm = confusion_matrix(y_true, all_preds)
        
        ax = plt.subplot(rows, cols, i+1)
        display_name = "DNSAGuard (Ours)" if m_name == "Transformer" else m_name
        cmap = "Reds" if m_name == "Transformer" else "Blues"
        
        sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, cbar=False,
                    xticklabels=['Benign', 'Malicious'], yticklabels=['Benign', 'Malicious'],
                    ax=ax, annot_kws={"size": 14, "weight": "bold"})
        ax.set_title(display_name, fontsize=16, fontweight='bold')
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "Fig6_Confusion_Matrix.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"      -> Đã lưu: {save_path}")

# --- 5. VẼ ATTENTION MAP ---
def plot_attention_map(X_test, y_test):
    print("\n[4/4] Đang vẽ Attention Map (Fig 7)...")
    fpath = os.path.join(latest_dir, "best_Transformer.pt")
    if not os.path.exists(fpath):
        print("      [Skip] Không tìm thấy model Transformer.")
        return

    model = TransformerClassifier(len(CONFIG['features']), CONFIG['hidden_dim']).to(DEVICE)
    model.load_state_dict(torch.load(fpath, map_location=DEVICE))
    model.eval()
    
    # Tìm mẫu Malicious "đậm đặc" (nhiều gói tin thật, ít padding)
    target_idx = -1
    for i in range(len(X_test)):
        if y_test[i] == 1:
            # Đếm số gói tin có size > 0
            real_len = torch.sum(X_test[i, :, 0] != 0).item() # Feature 0 là size
            if real_len > 20: # Lấy mẫu dài hơn 20 gói tin
                target_idx = i
                break
    
    if target_idx == -1: 
        print("      Không tìm thấy mẫu phù hợp để vẽ.")
        return

    sample = X_test[target_idx].unsqueeze(0).to(DEVICE)
    
    # Forward thủ công để lấy attention
    if hasattr(model, 'embedding'): x = model.embedding(sample)
    else: x = model.input_proj(sample)
    x = model.pos_encoder(x)
    enc_layer = model.transformer_encoder.layers[0]
    _, attn_weights = enc_layer.self_attn(x, x, x, need_weights=True)
    
    attn_map = attn_weights[0].cpu().detach().numpy()
    
    # Cắt bỏ padding để hình đẹp
    real_len = 50
    for k in range(49, -1, -1):
        if sample[0, k, 0] != 0: # Check size
            real_len = k + 1
            break
    
    attn_map = attn_map[:real_len, :real_len]
    
    plt.figure(figsize=(9, 8))
    sns.heatmap(attn_map, cmap="viridis", square=True, cbar_kws={'label': 'Attention Weight'})
    plt.title(f"DNSAGuard Self-Attention (Malicious Flow)", fontsize=16, fontweight='bold')
    plt.xlabel("Packet Index", fontsize=12)
    plt.ylabel("Packet Index", fontsize=12)
    
    save_path = os.path.join(OUTPUT_DIR, "Fig7_Attention.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"      -> Đã lưu: {save_path}")

# --- RUN ---
if __name__ == '__main__':
    try:
        X_t, y_t = load_data_robust()
        plot_tsne(X_t, y_t)
        plot_cm_grid(X_t, y_t)
        plot_attention_map(X_t, y_t)
        print(f"\n=== THÀNH CÔNG! Kiểm tra thư mục '{OUTPUT_DIR}' ===")
    except Exception as e:
        print(f"\n[LỖI CHƯƠNG TRÌNH]: {e}")
        traceback.print_exc()