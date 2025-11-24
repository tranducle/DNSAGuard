import torch
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# --- CẤU HÌNH GIỐNG LÚC TRAIN ---
CONFIG = {
    "sequence_length": 50,
    "features": ['size', 'time'], 
    "test_size": 0.2
}

def prepare_and_save_test_data():
    print("--- BẮT ĐẦU CHUẨN BỊ DỮ LIỆU FINAL ---")
    
    # 1. Load dữ liệu thô
    print("1. Đang đọc file 'all_clients_raw.pkl' (có thể mất vài phút)...")
    if not os.path.exists('all_clients_raw.pkl'):
        print("LỖI: Không tìm thấy file .pkl gốc!")
        return

    with open('all_clients_raw.pkl', 'rb') as f:
        all_clients = pickle.load(f)

    # 2. Trích xuất đặc trưng
    print("2. Đang trích xuất đặc trưng và tạo Tensor...")
    all_sequences, all_labels = [], []
    seq_len = CONFIG['sequence_length']

    for client_ip, data in all_clients.items():
        packets = data['packets']
        original_len = min(len(packets), seq_len)
        
        # Tạo tensor rỗng
        seq_tensor = torch.zeros(seq_len, len(CONFIG['features']), dtype=torch.float)
        last_time = packets[0][0] if packets else 0.0
        
        for i in range(original_len):
            pkt_time, pkt_size, pkt_proto = packets[i]
            
            # Feature 0: Size
            if 'size' in CONFIG['features']:
                seq_tensor[i, 0] = pkt_size
            
            # Feature 1: Time Delta
            if 'time' in CONFIG['features']:
                delta = max(0.0, pkt_time - last_time)
                seq_tensor[i, 1] = delta
            
            last_time = pkt_time
            
        all_sequences.append(seq_tensor)
        all_labels.append(data['label'])

    # Chuyển sang Tensor
    X = torch.stack(all_sequences).numpy()
    y = np.array(all_labels)
    print(f"   -> Tổng số mẫu: {len(y)}")

    # 3. Scaling (Cực kỳ quan trọng để khớp với model)
    print("3. Đang chuẩn hóa dữ liệu (StandardScaler)...")
    N, L, F = X.shape
    scaler = StandardScaler()
    # Fit trên toàn bộ dữ liệu (như cách chúng ta làm ở train_classifier.py đơn giản hóa)
    X_scaled = scaler.fit_transform(X.reshape(-1, F)).reshape(N, L, F)
    
    # Xử lý NaN nếu có
    X_scaled = np.nan_to_num(X_scaled, nan=0.0)

    X_final = torch.tensor(X_scaled, dtype=torch.float)
    y_final = torch.tensor(y, dtype=torch.float)

    # 4. Chia tập Test
    print("4. Đang tách tập Test (20%)...")
    _, X_test, _, y_test = train_test_split(
        X_final, y_final, 
        test_size=CONFIG['test_size'], 
        random_state=42, 
        stratify=y_final
    )
    
    print(f"   -> Số mẫu Test: {len(y_test)}")
    print(f"      Benign: {int(sum(y_test==0))}")
    print(f"      Malicious: {int(sum(y_test==1))}")

    # 5. Lưu file
    output_file = "final_test_dataset.pt"
    print(f"5. Đang lưu vào '{output_file}'...")
    torch.save({'X': X_test, 'y': y_test}, output_file)
    
    print("\n=== HOÀN TẤT! ===")
    print(f"Bây giờ bạn có thể chạy 'advanced_visualize.py' cực nhanh.")

if __name__ == '__main__':
    prepare_and_save_test_data()