# Beyond IP Memorization: Robust DoH Tunneling Detection using DNSAGuard and Flow-Based Sequence Modeling

## Overview

This repository contains the implementation of **DNSAGuard**, a robust detection system for DNS over HTTPS (DoH) tunneling attacks. Unlike traditional methods that rely on client-based aggregation (which often leads to IP memorization and poor generalization), this project utilizes a **Flow-Based Sequence Modeling** approach.

By aggregating traffic into 5-tuple flows and utilizing a lightweight **Transformer-based architecture**, DNSAGuard effectively learns behavioral signatures (Packet Size and Inter-arrival Time) without relying on source/destination IP addresses.

## Key Features

*   **Flow-Based Aggregation:** Transforms the problem from identifying malicious clients (limited samples) to identifying malicious flows (hundreds of thousands of samples), solving the extreme class imbalance problem.
*   **Privacy-Preserving Features:** Uses only **Packet Size** and **Inter-arrival Time**. No IP addresses or payload contents are used, ensuring the model learns attack behaviors rather than memorizing network identifiers.
*   **Transformer Architecture:** Leverages the Self-Attention mechanism to capture long-range dependencies and the "rhythm" of tunneling traffic better than RNN/LSTM baselines.
*   **High Performance:** Achieves ~98.8% F1-Score with minimal false alarms.

## Project Structure

*   `model.py`: Implementation of the `TransformerClassifier` (DNSAGuard) and baseline models (GRU, LSTM, CNN, MLP).
*   `train.py`: Training loop and evaluation logic.
*   `pcap_loader.py`: Utilities for processing PCAP files into flow-based sequences.
*   `prepare_final_data.py`: Scripts for data preprocessing and feature engineering.
*   `visualize_results.py` & `generate_paper_figures.py`: Tools for visualizing performance metrics and t-SNE embeddings.

## Methodology

### 1. Data Engineering (The Pivot)
The core innovation lies in shifting from Client-based to **Flow-based aggregation**. 
*   **Old Approach:** Group by Client IP $\rightarrow$ High imbalance (~1600:1), Data Leakage.
*   **New Approach:** Group by 5-tuple `(Src IP, Dst IP, Src Port, Dst Port, Protocol)` $\rightarrow$ Balanced data (~3.5:1), Robust generalization.

### 2. Model Architecture
The **DNSAGuard** model is a Supervised Transformer Classifier:
*   **Input:** Sequence of 50 packets (Size, Time).
*   **Encoder:** Transformer Encoder with Self-Attention.
*   **Output:** Binary classification (Benign vs. Malicious DoH).

## Results

The proposed Transformer model significantly outperforms traditional Deep Learning baselines on the CIRA-CIC-DoHBrw-2020 dataset.

| Model | F1-Score | Precision | Recall |
| :--- | :--- | :--- | :--- |
| **DNSAGuard (Transformer)** | **~0.988** | **~0.98** | **~1.00** |
| GRU Baseline | ~0.50 | - | - |

## Usage

### Prerequisites
*   Python 3.x
*   PyTorch
*   NumPy, Pandas, Scikit-learn

### Training
To train the model:
```bash
python train.py
```

### Visualization
To generate performance plots:
```bash
python visualize_results.py
```

## Citation

If you use this code, please cite our paper:
**"Beyond IP Memorization: Robust DoH Tunneling Detection using DNSAGuard and Flow-Based Sequence Modeling"**
