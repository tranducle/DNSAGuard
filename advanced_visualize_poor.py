import torch
import torch.nn as nn
import torch.nn.functional as F # Cần thêm cái này cho CNN
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import pickle
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Import các class mô hình
from model import TransformerClassifier, LSTMBaseline, GRUBaseline, CNNBaseline, MLPBaseline

# --- CẤU HÌNH ---
OUTPUT_DIR = "paper_figures_advanced"
os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG = {
    "sequence_length": 50,
    "features": ['size', 'time'], 
    "hidden_dim": 128,
    "test_size": 0.2
}

# Tìm thư mục kết quả mới nhất
result_dirs = glob.glob("comparison_results_*")
if not result_dirs:
    print("Lỗi: Không tìm thấy thư mục kết quả!")
    exit()
latest_dir = max(result_dirs, key=os.path.getmtime)
print(f"--> Đang lấy model từ: {latest_dir}")

def save_academic_figure(filename):
    formats = ['.png', '.pdf', '.svg']
    for fmt in formats:
        save_path = os.path.join(OUTPUT_DIR, f"{filename}{fmt}")
        try:
            plt.savefig(save_path, dpi=300, format=fmt.replace('.', ''), bbox_inches='tight', pad_inches=0.05)
        except: pass
    print(f"-> Đã lưu {filename}")

# --- 1. LOAD DATA ---
def load_test_data_smart():
    pt_file = "final_test_dataset.pt"
    if os.path.exists(pt_file):
        try:
            data = torch.load(pt_file)
            return data['X'], data['y']
        except: pass

    pkl_file = 'all_clients_raw.pkl'
    if not os.path.exists(pkl_file):
        print("Lỗi: Không tìm thấy dữ liệu.")
        exit()

    with open(pkl_file, 'rb') as f:
        all_clients = pickle.load(f)

    all_sequences, all_labels = [], []
    seq_len = CONFIG['sequence_length']

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
    N, L, F = X.shape
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.reshape(-1, F)).reshape(N, L, F)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0)
    X_final = torch.tensor(X_scaled, dtype=torch.float)
    y_final = torch.tensor(y, dtype=torch.float)
    _, X_test, _, y_test = train_test_split(X_final, y_final, test_size=CONFIG['test_size'], random_state=42, stratify=y_final)
    return X_test, y_test

# --- 2. TRÍCH XUẤT ĐẶC TRƯNG (NÂNG CẤP CHO MỌI MODEL) ---
def get_embeddings(model, X, model_type):
    model.eval()
    model.to(DEVICE)
    embeddings_list = []
    batch_size = 1024
    
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch_X = X[i:i+batch_size].to(DEVICE)
            
            # Logic riêng cho từng loại model để lấy vector đặc trưng (trước lớp classifier cuối)
            if model_type == "Transformer":
                if hasattr(model, 'embedding'): x = model.embedding(batch_X)
                else: x = model.input_proj(batch_X)
                x = model.pos_encoder(x)
                x = model.transformer_encoder(x)
                emb = x.mean(dim=1)
            
            elif model_type == "LSTM":
                output, _ = model.lstm(batch_X)
                emb = output.mean(dim=1)
                
            elif model_type == "GRU":
                output, _ = model.gru(batch_X)
                emb = output.mean(dim=1)
                
            elif model_type == "1D-CNN":
                # Chạy thủ công các lớp Conv để lấy output trước Classifier
                x = batch_X.permute(0, 2, 1)
                x = F.relu(model.conv1(x))
                x = model.pool(x)
                x = F.relu(model.conv2(x))
                x = model.pool(x)
                emb = x.view(x.size(0), -1) # Flattened vector là embedding
                
            elif model_type == "MLP":
                # Chạy thủ công qua các lớp Linear đầu, dừng trước lớp cuối
                x = batch_X.view(batch_X.size(0), -1)
                # MLP net: [Linear, ReLU, Drop, Linear, ReLU, Drop, Linear]
                # Ta muốn lấy output sau lớp Linear thứ 2 (index 3)
                for layer_idx, layer in enumerate(model.net):
                    x = layer(x)
                    if layer_idx == 3: # Sau lớp Linear thứ 2 (hidden layer)
                        break
                emb = x
            
            else:
                return None
                
            embeddings_list.append(emb.cpu())
            
    return torch.cat(embeddings_list).numpy() if embeddings_list else np.array([])

# --- 3. VẼ T-SNE (FULL 5 MODELS) ---
def plot_tsne(X_test, y_test, n_samples=2000):
    print("\n--- VẼ BIỂU ĐỒ t-SNE (Full Models) ---")
    
    idx_mal = (y_test == 1).nonzero(as_tuple=True)[0]
    idx_ben = (y_test == 0).nonzero(as_tuple=True)[0]
    
    n_per_class = min(len(idx_mal), len(idx_ben), n_samples // 2)
    n_per_class = max(n_per_class, 50)
    n_per_class = min(n_per_class, len(idx_mal), len(idx_ben))

    print(f"  Lấy {n_per_class} mẫu mỗi lớp...")
    sel_mal = idx_mal[torch.randperm(len(idx_mal))[:n_per_class]]
    sel_ben = idx_ben[torch.randperm(len(idx_ben))[:n_per_class]]
    indices = torch.cat([sel_mal, sel_ben])
    indices = indices[torch.randperm(len(indices))]
    
    X_subset = X_test[indices]
    y_subset = y_test[indices]

    # Danh sách tất cả model cần vẽ
    models_to_plot = ["Transformer", "LSTM", "GRU", "1D-CNN", "MLP"]
    
    # Tạo lưới 2 hàng, 3 cột (cho 5 hình)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten() # Làm phẳng để dễ loop
    
    plot_idx = 0
    for m_name in models_to_plot:
        model_path = os.path.join(latest_dir, f"best_{m_name}.pt")
        if not os.path.exists(model_path): 
            print(f"  [Skip] {m_name}")
            continue
            
        print(f"  Đang xử lý t-SNE cho {m_name}...")
        
        try:
            # Init đúng loại model
            if m_name == "Transformer":
                model = TransformerClassifier(len(CONFIG['features']), CONFIG['hidden_dim'])
                title = "DNSAGuard (Ours)"
            elif m_name == "LSTM":
                model = LSTMBaseline(len(CONFIG['features']), CONFIG['hidden_dim'])
                title = "LSTM"
            elif m_name == "GRU":
                model = GRUBaseline(len(CONFIG['features']), CONFIG['hidden_dim'])
                title = "GRU"
            elif m_name == "1D-CNN":
                model = CNNBaseline(len(CONFIG['features']), seq_len=CONFIG['sequence_length'], hidden_dim=CONFIG['hidden_dim'])
                title = "1D-CNN"
            elif m_name == "MLP":
                model = MLPBaseline(len(CONFIG['features']), seq_len=CONFIG['sequence_length'], hidden_dim=CONFIG['hidden_dim'])
                title = "MLP"
            
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            embeddings = get_embeddings(model, X_subset, m_name)
            
            tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto')
            res = tsne.fit_transform(embeddings)
            
            df_plot = pd.DataFrame({
                'x': res[:,0], 'y': res[:,1], 
                'Label': ['Malicious' if v==1 else 'Benign' for v in y_subset.cpu().numpy()]
            })
            
            ax = axes[plot_idx]
            sns.scatterplot(data=df_plot, x='x', y='y', hue='Label', 
                            palette={'Benign': '#2ecc71', 'Malicious': '#e74c3c'},
                            alpha=0.6, s=40, ax=ax, legend=(plot_idx==0)) # Chỉ hiện legend ở hình đầu tiên cho gọn
            
            ax.set_title(f"{title}", fontsize=14, fontweight='bold')
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False) # Ẩn trục số cho đẹp
            
            plot_idx += 1
            
        except Exception as e:
            print(f"  Lỗi vẽ {m_name}: {e}")

    # Xóa ô trống cuối cùng (nếu có 5 hình trên lưới 6 ô)
    if plot_idx < 6:
        fig.delaxes(axes[5])

    plt.tight_layout()
    save_academic_figure("Fig5_tSNE_Clusters_All")
    plt.close()

# --- 4. VẼ CM GRID (GIỮ NGUYÊN) ---
def plot_cm_grid(X_test, y_test):
    print("\n--- VẼ CONFUSION MATRIX ---")
    y_true = y_test.cpu().numpy()
    files = glob.glob(os.path.join(latest_dir, "best_*.pt"))
    model_map = {'Transformer': TransformerClassifier, 'LSTM': LSTMBaseline, 'GRU': GRUBaseline, '1D-CNN': CNNBaseline, 'MLP': MLPBaseline}
    
    valid_files = []
    for f in files:
        name = os.path.basename(f).replace("best_", "").replace(".pt", "")
        if name in model_map: valid_files.append((name, f))
    valid_files.sort(key=lambda x: x[0] == 'Transformer', reverse=True)
    
    cols = 3
    rows = (len(valid_files) + 2) // 3
    plt.figure(figsize=(5 * cols, 4 * rows))
    
    for i, (m_name, fpath) in enumerate(valid_files):
        ModelClass = model_map[m_name]
        if m_name in ["1D-CNN", "MLP"]:
            model = ModelClass(len(CONFIG['features']), seq_len=CONFIG['sequence_length'], hidden_dim=CONFIG['hidden_dim']).to(DEVICE)
        else:
            model = ModelClass(len(CONFIG['features']), hidden_dim=CONFIG['hidden_dim']).to(DEVICE)
        model.load_state_dict(torch.load(fpath, map_location=DEVICE))
        model.eval()
        
        all_preds = []
        batch_size = 4096
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
        sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, cbar=False, ax=ax, annot_kws={"size": 12})
        ax.set_title(display_name, fontsize=14, fontweight='bold')

    plt.tight_layout()
    save_academic_figure("Fig6_Confusion_Matrix")
    plt.close()

# --- 5. VẼ ATTENTION MAP (GIỮ NGUYÊN) ---
def plot_attention_map(X_test, y_test):
    print("\n--- VẼ ATTENTION MAP ---")
    fpath = os.path.join(latest_dir, "best_Transformer.pt")
    if not os.path.exists(fpath): return

    model = TransformerClassifier(len(CONFIG['features']), CONFIG['hidden_dim']).to(DEVICE)
    model.load_state_dict(torch.load(fpath, map_location=DEVICE))
    model.eval()
    
    target_idx = -1
    for i in range(len(X_test)):
        if y_test[i] == 1 and torch.sum(X_test[i, :, 0] != 0).item() > 20:
            target_idx = i
            break
    if target_idx == -1: return

    sample = X_test[target_idx].unsqueeze(0).to(DEVICE)
    if hasattr(model, 'embedding'): x = model.embedding(sample)
    else: x = model.input_proj(sample)
    x = model.pos_encoder(x)
    enc_layer = model.transformer_encoder.layers[0]
    _, attn_weights = enc_layer.self_attn(x, x, x, need_weights=True)
    
    attn_map = attn_weights[0].cpu().detach().numpy()
    real_len = 50
    for k in range(49, -1, -1):
        if sample[0, k, 0] != 0:
            real_len = k + 1
            break
    attn_map = attn_map[:real_len, :real_len]
    
    plt.figure(figsize=(8, 7))
    sns.heatmap(attn_map, cmap="viridis", square=True, cbar_kws={'label': 'Attention Weight'})
    plt.title(f"DNSAGuard Self-Attention", fontsize=16, fontweight='bold')
    plt.xlabel("Packet Index", fontsize=12)
    plt.ylabel("Packet Index", fontsize=12)
    save_academic_figure("Fig7_Attention")
    plt.close()

if __name__ == '__main__':
    X_test, y_test = load_test_data_smart()
    plot_tsne(X_test, y_test)
    plot_cm_grid(X_test, y_test)
    plot_attention_map(X_test, y_test)
    print("\n=== HOÀN TẤT! ===")