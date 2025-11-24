import torch
import time
import numpy as np
import os
import glob
from model import TransformerClassifier, LSTMBaseline, GRUBaseline, CNNBaseline, MLPBaseline

# --- CẤU HÌNH ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIG = {
    "sequence_length": 50,
    "features": 2, 
    "hidden_dim": 128
}

def measure_inference_time(model, model_name, n_repeats=1000):
    model.eval()
    model.to(DEVICE)
    
    # Tạo dữ liệu giả lập (Batch size = 1)
    # Giả lập 1 dòng dữ liệu đi vào hệ thống
    dummy_input = torch.randn(1, CONFIG['sequence_length'], CONFIG['features']).to(DEVICE)
    
    # 1. Warm-up (Để GPU nóng máy, load cache)
    # Rất quan trọng để đo chính xác
    with torch.no_grad():
        for _ in range(100):
            _ = model(dummy_input)
    
    # 2. Đo thời gian thực
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_repeats):
            _ = model(dummy_input)
            # Nếu dùng GPU, phải đồng bộ hóa để thời gian chính xác
            if DEVICE.type == 'cuda':
                torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    # Tính trung bình (ms)
    total_time = end_time - start_time
    avg_time_ms = (total_time / n_repeats) * 1000
    
    print(f"[{model_name}] Average Latency: {avg_time_ms:.4f} ms/flow")
    return avg_time_ms

if __name__ == '__main__':
    print(f"--- MEASURING LATENCY (Device: {DEVICE}) ---")
    print(f"Simulating Real-time Detection (Batch Size = 1)")
    
    # Tìm thư mục kết quả mới nhất
    result_dirs = glob.glob("comparison_results_*")
    if not result_dirs:
        print("Lỗi: Không tìm thấy thư mục kết quả!")
        exit()
    latest_dir = max(result_dirs, key=os.path.getmtime)
    print(f"Loading models from: {latest_dir}\n")

    # Danh sách model cần đo
    models_to_test = {
        "Transformer": TransformerClassifier,
        "GRU": GRUBaseline,
        "LSTM": LSTMBaseline,
        "1D-CNN": CNNBaseline,
        "MLP": MLPBaseline
    }

    results = {}

    for name, ModelClass in models_to_test.items():
        # Tìm file weight
        fpath = os.path.join(latest_dir, f"best_{name}.pt")
        if not os.path.exists(fpath):
            print(f"Skipping {name} (Weight not found)")
            continue
            
        # Init Model
        if name in ["1D-CNN", "MLP"]:
            model = ModelClass(CONFIG['features'], seq_len=CONFIG['sequence_length'], hidden_dim=CONFIG['hidden_dim'])
        else:
            model = ModelClass(CONFIG['features'], hidden_dim=CONFIG['hidden_dim'])
            
        # Load Weight
        model.load_state_dict(torch.load(fpath, map_location=DEVICE))
        
        # Measure
        latency = measure_inference_time(model, name)
        results[name] = latency

    print("\n--- FINAL RESULTS FOR PAPER ---")
    print("Copy những số này vào bài báo của bạn:")
    for name, lat in results.items():
        print(f"{name}: {lat:.3f} ms")