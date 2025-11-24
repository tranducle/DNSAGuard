import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import pickle
import matplotlib.gridspec as gridspec
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Import models
from model import TransformerClassifier, LSTMBaseline, GRUBaseline, CNNBaseline, MLPBaseline

# --- CẤU HÌNH ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIG = {"sequence_length": 50, "features": ['size', 'time'], "hidden_dim": 128, "test_size": 0.2}
OUTPUT_DIR = "paper_figures" # Lưu chung vào thư mục chính
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- LOAD DATA ---
def load_data():
    print("--- ĐANG TẢI DỮ LIỆU ---")
    if os.path.exists("final_test_dataset.pt"):
        print("Load từ final_test_dataset.pt...")
        d = torch.load("final_test_dataset.pt")
        return d['X'], d['y']
    
    print("Load từ all_clients_raw.pkl...")
    if not os.path.exists("all_clients_raw.pkl"):
        print("LỖI: Không tìm thấy file dữ liệu!")
        exit()
        
    with open("all_clients_raw.pkl", 'rb') as f:
        all_clients = pickle.load(f)
        
    seqs, lbls = [], []
    for ip, data in all_clients.items():
        pkts = data['packets']
        l = min(len(pkts), 50)
        t = torch.zeros(50, 2)
        last = pkts[0][0] if pkts else 0
        for i in range(l):
            t[i, 0] = pkts[i][1] 
            t[i, 1] = max(0, pkts[i][0] - last)
            last = pkts[i][0]
        seqs.append(t)
        lbls.append(data['label'])
        
    X = torch.stack(seqs).numpy()
    y = np.array(lbls)
    
    N, L, F = X.shape
    X = StandardScaler().fit_transform(X.reshape(-1, F)).reshape(N, L, F)
    X = np.nan_to_num(X, nan=0.0)
    
    _, X_test, _, y_test = train_test_split(torch.tensor(X, dtype=torch.float), torch.tensor(y, dtype=torch.float), test_size=0.2, random_state=42, stratify=y)
    return X_test, y_test

# --- HÀM LƯU ĐA ĐỊNH DẠNG ---
def save_multi_format(filename):
    formats = ['.png', '.pdf', '.svg', '.eps']
    for fmt in formats:
        save_path = os.path.join(OUTPUT_DIR, f"{filename}{fmt}")
        try:
            # Sử dụng bbox_inches='tight' cẩn thận để không cắt mất chữ
            plt.savefig(save_path, dpi=300, format=fmt.replace('.', ''), bbox_inches='tight', pad_inches=0.1)
            print(f"  -> Đã lưu: {save_path}")
        except Exception as e:
            print(f"  [Cảnh báo] Lỗi khi lưu {fmt}: {e}")

# --- PLOT SPOTLIGHT ---
def plot_cm_spotlight():
    X_test, y_test = load_data()
    y_true = y_test.numpy()
    print(f"Số mẫu Test: {len(y_true)}")

    dirs = glob.glob("comparison_results_*")
    if not dirs: print("Không tìm thấy thư mục kết quả!"); return
    latest_dir = max(dirs, key=os.path.getmtime)
    print(f"Lấy model từ: {latest_dir}")
    
    files = glob.glob(os.path.join(latest_dir, "best_*.pt"))
    model_map = {
        'Transformer': TransformerClassifier, 'LSTM': LSTMBaseline, 
        'GRU': GRUBaseline, '1D-CNN': CNNBaseline, 'MLP': MLPBaseline
    }
    
    valid_files = []
    for f in files:
        name = os.path.basename(f).replace("best_", "").replace(".pt", "")
        if name in model_map: valid_files.append((name, f))
    
    # Sắp xếp thứ tự vẽ
    order = {'Transformer': 0, 'LSTM': 1, 'GRU': 2, '1D-CNN': 3, 'MLP': 4}
    valid_files.sort(key=lambda x: order.get(x[0], 99))
    
    if not valid_files: print("Không tìm thấy model nào!"); return

    # Setup GridSpec
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 3, width_ratios=[1.3, 1, 1]) 

    print("Đang vẽ Confusion Matrix...")
    for i, (m_name, fpath) in enumerate(valid_files):
        # Init Model
        ModelClass = model_map[m_name]
        kw = {'num_features': 2, 'hidden_dim': 128}
        if m_name in ['1D-CNN', 'MLP']: kw['seq_len'] = 50
        
        model = ModelClass(**kw).to(DEVICE)
        model.load_state_dict(torch.load(fpath, map_location=DEVICE))
        model.eval()
        
        preds = []
        with torch.no_grad():
            for j in range(0, len(X_test), 4096):
                bx = X_test[j:j+4096].to(DEVICE)
                logits = model(bx)
                p = (torch.sigmoid(logits) > 0.5).float().cpu()
                preds.append(p)
        preds = torch.cat(preds).numpy()
        cm = confusion_matrix(y_true, preds)
        
        # Setup vị trí
        if i == 0: # DNSAGuard (To bên trái)
            ax = fig.add_subplot(gs[:, 0])
            title_size, annot_size = 18, 20
            cmap, display_name = "Reds", "DNSAGuard (Ours)"
        else: # Các model khác (Nhỏ bên phải)
            row_idx = (i - 1) // 2
            col_idx = (i - 1) % 2 + 1
            ax = fig.add_subplot(gs[row_idx, col_idx])
            title_size, annot_size = 14, 12
            cmap, display_name = "Blues", m_name

        # Vẽ Heatmap
        sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, cbar=False, 
                    ax=ax, annot_kws={"size": annot_size, "weight": "bold"})
        
        ax.set_title(display_name, fontsize=title_size, fontweight='bold', pad=10)
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("Actual", fontsize=12)
        ax.set_xticklabels(['Benign', 'Malicious'])
        ax.set_yticklabels(['Benign', 'Malicious'])

    plt.tight_layout()
    
    # Lưu đa định dạng
    print("Đang lưu file...")
    save_multi_format("Fig6_Confusion_Matrix_Spotlight")
    print("Hoàn tất!")

if __name__ == '__main__':
    plot_cm_spotlight()