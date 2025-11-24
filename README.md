# DNSAGuard: Robust DoH Tunneling Detection using Flow-Based Transformers

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)

## 📝 Abstract

This repository contains the official implementation of **DNSAGuard**, a Flow-based Intrusion Detection System (IDS) designed to detect DNS-over-HTTPS (DoH) tunneling attacks. Unlike traditional methods that rely on client-centric aggregation (prone to IP-based data leakage) or statistical feature engineering (losing temporal context), DNSAGuard leverages a **Transformer Encoder** architecture acting on raw packet sequences.

By utilizing only two privacy-preserving features—**Packet Size** and **Inter-arrival Time**—DNSAGuard achieves state-of-the-art performance on the CIRA-CIC-DoHBrw-2020 dataset, demonstrating superior robustness against topology changes and providing intrinsic interpretability via Self-Attention mechanisms.

## 🚀 Key Features

* **Flow-Based Aggregation:** 5-tuple flow extraction preventing Identity-Based Data Leakage.
* **Transformer Architecture:** Captures global structural patterns ("heartbeats") of tunneling tools.
* **Privacy-Preserving:** Operates without decrypting payloads and excludes IP/Port identifiers.
* **Explainable AI (XAI):** Visualizes attention maps to reveal malicious packet bursts.
* **High Performance:** F1-Score ~99.6%, surpassing LSTM, GRU, and 1D-CNN baselines.

## 🛠 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/tranducle/DNSAGuard.git
   cd DNSAGuard
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Dataset Preparation

We utilize the **CIRA-CIC-DoHBrw-2020** dataset.

1. Download the raw PCAP files from the [official source](https://www.unb.ca/cic/datasets/dohbrw-2020.html).
2. Organize the PCAPs into the `DATASET/` folder structure (see `data/README.md` for details).
3. Run the preprocessing pipeline:

```bash
# Step 1: Extract flows from PCAPs (Slow, run once)
python src/pcap_loader.py

# Step 2: Normalize and create Tensor datasets
python src/prepare_data.py
```

## 🧠 Training & Benchmarking

To train DNSAGuard and compare it with baselines (LSTM, GRU, 1D-CNN, MLP):

```bash
python src/train.py
```

*Models will be saved in `comparison_results_YYYYMMDD/`.*

## 📈 Visualization & Evaluation

Generate the figures used in the paper (t-SNE, Confusion Matrices, Attention Maps):

```bash
python analysis/visualize.py
```

*Outputs are saved to `paper/figures/`.*

To measure inference latency:

```bash
python analysis/latency_test.py
```

## 📊 Results

The proposed Transformer model significantly outperforms traditional Deep Learning baselines on the CIRA-CIC-DoHBrw-2020 dataset.

| Model | F1-Score | Precision | Recall | Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| **DNSAGuard (Transformer)** | **0.9988** | **0.9968** | **0.9990** | **0.9992** |
| LSTM Baseline | 0.9983 | 0.9955 | 0.9986 | 0.9989 |
| GRU Baseline | 0.9984 | 0.9977 | 0.9991 | 0.9990 |
| 1D-CNN Baseline | 0.9981 | 0.9952 | 0.9985 | 0.9988 |
| MLP Baseline | 0.9943 | 0.9912 | 0.9968 | 0.9967 |

## 📂 Project Structure

```
DNSAGuard/
├── README.md                 # Project overview and instructions
├── requirements.txt          # Python dependencies
├── .gitignore                # Git ignore rules
│
├── data/                     # Dataset directory (see data/README.md)
│   ├── README.md             # Dataset download instructions
│   └── .gitkeep
│
├── src/                      # Source code
│   ├── __init__.py
│   ├── model.py              # Transformer and baseline models
│   ├── pcap_loader.py        # PCAP to flow extraction
│   ├── prepare_data.py       # Data preprocessing
│   └── train.py              # Training and comparison script
│
├── analysis/                 # Analysis and visualization
│   ├── visualize.py          # Generate paper figures
│   └── latency_test.py       # Inference latency measurement
│
└── paper/                    # Paper-related materials
    └── figures/              # Generated figures
```

## 🔬 Methodology

### 1. Data Engineering (The Pivot)
The core innovation lies in shifting from Client-based to **Flow-based aggregation**. 
* **Old Approach:** Group by Client IP → High imbalance (~1600:1), Data Leakage.
* **New Approach:** Group by 5-tuple `(Src IP, Dst IP, Src Port, Dst Port, Protocol)` → Balanced data (~3.5:1), Robust generalization.

### 2. Model Architecture
The **DNSAGuard** model is a Supervised Transformer Classifier:
* **Input:** Sequence of 50 packets (Size, Time).
* **Encoder:** Transformer Encoder with Self-Attention.
* **Output:** Binary classification (Benign vs. Malicious DoH).

## 📜 Citation

If you use this code for your research, please cite our paper:

```bibtex
@article{dnsaguard2025,
  title={Beyond IP Memorization: Robust DoH Tunneling Detection using DNSAGuard and Flow-Based Sequence Modeling},
  author={Tran Duc Le and Yida Bao and Mohammad Arifuzzaman and Nam Son Nguyen and Truong Duy Dinh},
  journal={Submitted},
  year={2025}
}
```

## 👥 Authors

* **Tran Duc Le** (University of Wisconsin-Stout)
* **Yida Bao** (University of Wisconsin-Stout)
* **Nam Son Nguyen** (Hewlett Packard Enterprise)
* **Truong Duy Dinh** (PTIT, Vietnam) - *Corresponding Author*

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This research was conducted using the CIRA-CIC-DoHBrw-2020 dataset provided by the Canadian Institute for Cybersecurity (CIC).
