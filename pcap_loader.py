import os
import glob
import pickle
import pandas as pd
from scapy.all import rdpcap, IP, TCP, UDP
from collections import defaultdict

def parse_pcap_files(root_path, label):
    """
    Quét và trả về dict flows cho một nhãn cụ thể.
    """
    print(f"--- Đang quét: {root_path} ({label}) ---")
    label_int = 1 if label == 'spoofed' else 0
    local_flows = {}
    file_count = 0
    
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.endswith('.pcap'):
                file_path = os.path.join(root, file)
                file_count += 1
                
                if file_count % 100 == 0:
                    print(f"  > Đã xử lý {file_count} file...")

                try:
                    packets = rdpcap(file_path)
                    for pkt in packets:
                        if IP in pkt:
                            src_ip = pkt[IP].src
                            dst_ip = pkt[IP].dst
                            proto = int(pkt[IP].proto)
                            
                            sport = 0
                            dport = 0
                            if TCP in pkt:
                                sport = pkt[TCP].sport
                                dport = pkt[TCP].dport
                            elif UDP in pkt:
                                sport = pkt[UDP].sport
                                dport = pkt[UDP].dport
                            else:
                                continue
                            
                            # Flow Key 5-tuple
                            flow_key = (src_ip, dst_ip, sport, dport, proto)
                            
                            # Lấy time an toàn
                            try:
                                pkt_time = float(pkt.time)
                            except:
                                pkt_time = 0.0
                            
                            pkt_size = float(len(pkt))
                            
                            if flow_key not in local_flows:
                                local_flows[flow_key] = {
                                    'packets': [],
                                    'dest_ips': set(),
                                    'label': label_int
                                }
                                local_flows[flow_key]['dest_ips'].add(dst_ip)
                            
                            local_flows[flow_key]['packets'].append((pkt_time, pkt_size, proto))
                                
                except Exception as e:
                    print(f"  [BỎ QUA] Lỗi file {file}: {e}")
                    
    print(f"-> Hoàn tất {label}: {file_count} file.")
    return local_flows

if __name__ == '__main__':
    base_path = 'DATASET'
    benign_root = os.path.join(base_path, 'PCAPs', 'DoHBenign-NonDoH')
    malicious_root = os.path.join(base_path, 'PCAPs', 'DoHMalicious')
    
    # --- BƯỚC 1: XỬ LÝ BENIGN (CÓ CHECKPOINT) ---
    benign_cache_file = 'temp_benign.pkl'
    
    if os.path.exists(benign_cache_file):
        print(f"[CHECKPOINT] Tìm thấy file đã lưu {benign_cache_file}. Đang tải lại...")
        with open(benign_cache_file, 'rb') as f:
            benign_flows = pickle.load(f)
        print(f"-> Đã tải xong {len(benign_flows)} luồng Benign.")
    else:
        # Nếu chưa có thì phải chạy
        benign_flows = parse_pcap_files(benign_root, 'Benign')
        print(f"[LƯU] Đang lưu checkpoint Benign...")
        with open(benign_cache_file, 'wb') as f:
            pickle.dump(benign_flows, f)

    # --- BƯỚC 2: XỬ LÝ MALICIOUS (CÓ CHECKPOINT) ---
    malicious_cache_file = 'temp_malicious.pkl'
    
    if os.path.exists(malicious_cache_file):
        print(f"[CHECKPOINT] Tìm thấy file đã lưu {malicious_cache_file}. Đang tải lại...")
        with open(malicious_cache_file, 'rb') as f:
            malicious_flows = pickle.load(f)
        print(f"-> Đã tải xong {len(malicious_flows)} luồng Malicious.")
    else:
        malicious_flows = parse_pcap_files(malicious_root, 'spoofed')
        print(f"[LƯU] Đang lưu checkpoint Malicious...")
        with open(malicious_cache_file, 'wb') as f:
            pickle.dump(malicious_flows, f)

    # --- BƯỚC 3: GỘP DỮ LIỆU ---
    print("Đang gộp dữ liệu...")
    all_flows = benign_flows.copy()
    all_flows.update(malicious_flows) # Gộp Malicious vào
    
    print(f"Tổng số luồng: {len(all_flows)}")

    # --- BƯỚC 4: SẮP XẾP VÀ LƯU FINAL ---
    print("Đang sắp xếp các gói tin...")
    for key, data in all_flows.items():
        try:
            data['packets'] = sorted(data['packets'], key=lambda x: x[0])
        except:
            data['packets'] = []

    output_file = 'all_clients_raw.pkl'
    print(f"Đang lưu file cuối cùng {output_file}...")
    with open(output_file, 'wb') as f:
        pickle.dump(all_flows, f)

    # --- PHÂN TÍCH ---
    print("\n=== KẾT QUẢ (FLOW-BASED) ===")
    labels = [d['label'] for d in all_flows.values() if len(d['packets']) > 0]
    label_counts = pd.Series(labels).value_counts()
    print(f"0 (Benign): {label_counts.get(0, 0)}")
    print(f"1 (Spoofed): {label_counts.get(1, 0)}")
    
    ratio = label_counts.get(0, 0) / label_counts.get(1, 1)
    print(f"Tỷ lệ mất cân bằng: {ratio:.2f} : 1")
    
    with open("imbalance_ratio.txt", "w") as f:
        f.write(str(ratio))

    print("\n[XONG] Hãy chạy train.py")