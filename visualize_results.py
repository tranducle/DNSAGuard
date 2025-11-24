import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import glob

# --- CẤU HÌNH ---
OUTPUT_DIR = "paper_figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Tìm file kết quả mới nhất
result_dirs = glob.glob("comparison_results_*")
if not result_dirs:
    print("Lỗi: Không tìm thấy thư mục kết quả!")
    exit()

latest_dir = max(result_dirs, key=os.path.getmtime)
csv_file = os.path.join(latest_dir, "all_models_comparison.csv")
print(f"--> Đang đọc dữ liệu từ: {csv_file}")

try:
    df = pd.read_csv(csv_file)
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy file csv trong {latest_dir}")
    exit()

# --- CHUẨN BỊ DỮ LIỆU ---
rename_map = {
    'Transformer': 'DNSAGuard (Ours)',
    '1D-CNN': '1D-CNN',
    'LSTM': 'LSTM',
    'GRU': 'GRU',
    'MLP': 'MLP'
}
df['Model'] = df['Model'].replace(rename_map)

# Kiểm tra các model có trong dữ liệu
unique_models = df['Model'].unique()
print(f"Các model tìm thấy: {unique_models}")

# Định nghĩa màu sắc CỐ ĐỊNH cho từng model để nhất quán giữa các biểu đồ
# DNSAGuard luôn là màu đỏ nổi bật
model_colors = {
    'DNSAGuard (Ours)': '#e74c3c', # Đỏ
    'LSTM': '#3498db',             # Xanh dương
    'GRU': '#2ecc71',              # Xanh lá
    '1D-CNN': '#9b59b6',           # Tím
    'MLP': '#95a5a6'               # Xám
}

# Chỉ lấy màu cho các model thực sự có trong dữ liệu
palette = {m: model_colors.get(m, '#333333') for m in unique_models}

# Thiết lập style
sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)

def save_academic_figure(filename):
    formats = ['.png', '.pdf', '.eps', '.svg']
    for fmt in formats:
        save_path = os.path.join(OUTPUT_DIR, f"{filename}{fmt}")
        plt.savefig(save_path, dpi=300, format=fmt.replace('.', ''), bbox_inches='tight', pad_inches=0.05)
    print(f"-> Đã lưu {filename}")

# ========================================================
# FIG 1. F1-SCORE COMPARISON
# ========================================================
plt.figure(figsize=(10, 6))
ax = sns.lineplot(
    data=df, 
    x='Epoch', 
    y='Val_F1', 
    hue='Model', 
    style='Model', 
    markers=True, 
    dashes=False, 
    linewidth=2.5,
    palette=palette, # Sử dụng dict màu đã định nghĩa
    markersize=8
)

plt.title("Validation F1-Score Comparison", fontsize=16, fontweight='bold', pad=15)
plt.ylabel("F1-Score", fontsize=14)
plt.xlabel("Epoch", fontsize=14)
plt.ylim(0.90, 1.002) 

# Legend góc dưới phải
plt.legend(title='Model Architecture', loc='lower right', framealpha=0.95, frameon=True)

plt.grid(True, which='major', linestyle='--', alpha=0.7)
save_academic_figure("Fig1_F1_Score_Comparison")
plt.close()

# ========================================================
# FIG 2. TRAINING LOSS (Log Scale)
# ========================================================
plt.figure(figsize=(10, 6))
ax = sns.lineplot(
    data=df, 
    x='Epoch', 
    y='TrainLoss', 
    hue='Model', 
    linewidth=2,
    palette=palette
)
plt.title("Training Loss Convergence (Log Scale)", fontsize=16, fontweight='bold', pad=15)
plt.ylabel("Binary Cross Entropy Loss", fontsize=14)
plt.xlabel("Epoch", fontsize=14)
plt.yscale('log') 

# Legend góc trên phải
plt.legend(loc='upper right', framealpha=0.95, frameon=True)

plt.grid(True, which='both', linestyle=':', alpha=0.6)
save_academic_figure("Fig2_Training_Loss")
plt.close()

# ========================================================
# FIG 3. RADAR CHART
# ========================================================
best_metrics = df.loc[df.groupby('Model')['Val_F1'].idxmax()]
categories = ['Val_F1', 'Val_Precision', 'Val_Recall', 'Val_Accuracy']
labels = ['F1-Score', 'Precision', 'Recall', 'Accuracy']
N = len(categories)

angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

plt.figure(figsize=(8, 8))
ax = plt.subplot(111, polar=True)

# Sắp xếp để DNSAGuard vẽ cuối cùng
model_names = best_metrics['Model'].unique()
if 'DNSAGuard (Ours)' in model_names:
    # Lọc ra các model khác rồi cộng DNSAGuard vào cuối
    other_models = [m for m in model_names if m != 'DNSAGuard (Ours)']
    model_names = other_models + ['DNSAGuard (Ours)']

for i, model_name in enumerate(model_names):
    row = best_metrics[best_metrics['Model'] == model_name].iloc[0]
    values = row[categories].tolist()
    values += values[:1]
    
    is_ours = (model_name == 'DNSAGuard (Ours)')
    lw = 3.5 if is_ours else 1.5
    ls = 'solid' if is_ours else 'dotted'
    
    # Lấy màu từ palette đã định nghĩa
    color = palette.get(model_name, '#333333')
    
    ax.plot(angles, values, linewidth=lw, linestyle=ls, label=model_name, color=color)
    if is_ours:
        ax.fill(angles, values, color=color, alpha=0.1)

plt.xticks(angles[:-1], labels, color='black', size=12)
ax.set_rlabel_position(0)
plt.yticks([0.95, 0.97, 0.99], ["0.95", "0.97", "0.99"], color="grey", size=10)
plt.ylim(0.94, 1.001) 

plt.title("Peak Performance Comparison", size=16, fontweight='bold', y=1.1)
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()

save_academic_figure("Fig3_Radar_Chart")
plt.close()

# ========================================================
# FIG 4. BAR CHART (Training Time)
# ========================================================
time_df = df.groupby('Model')['Time(m)'].mean().reset_index().sort_values('Time(m)')

plt.figure(figsize=(8, 5))
# Tạo danh sách màu cho Bar Chart dựa trên tên model
bar_colors = [palette.get(x, '#d3d3d3') for x in time_df['Model']]

ax = sns.barplot(data=time_df, x='Model', y='Time(m)', palette=bar_colors)

for p in ax.patches:
    ax.annotate(format(p.get_height(), '.2f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 9), 
                   textcoords = 'offset points',
                   fontweight='bold')

plt.title("Computational Efficiency (Avg Time per Epoch)", fontsize=14, fontweight='bold')
plt.ylabel("Time (minutes)", fontsize=12)
plt.xlabel("")
plt.ylim(0, time_df['Time(m)'].max() * 1.2)
save_academic_figure("Fig4_Training_Time")
plt.close()

print(f"\nHOÀN TẤT! Đã lưu hình ảnh vào thư mục '{OUTPUT_DIR}'.")