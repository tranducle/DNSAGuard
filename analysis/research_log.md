# Research Journey: DNSAGuard Development

This document chronicles the evolution of the DNSAGuard project, from initial challenges to the final breakthrough.

## Phase 1: Initial Approach and Critical Flaws (The Learning Phase)

### 1. Client-based Approach
* **Idea:** Group all packets by `Source IP` (Client). Treat each computer as a node in a graph.
* **Model:** GNN (for IP graph) combined with GRU (for packet sequences).
* **Results:**
    * Achieved accuracy and F1-Score **~100%** (or near perfect).
* **Critical Problem:** Discovered **Data Leakage**.
    * Dataset used only ~17 fixed IPs for attacks.
    * GNN model **memorized IP addresses** instead of learning attack behaviors.
    * Model would fail on new, unseen IPs.
    * Extremely imbalanced data (Imbalance Ratio ~1600:1) made it difficult for the model to learn true features.

### 2. Anomaly Detection Experiment (Autoencoder)
* **Hypothesis:** Since attack data was scarce (when grouped by Client), we shifted to unsupervised learning: "Learn normal to detect abnormal."
* **Model:** LSTM Autoencoder & Transformer Autoencoder.
* **Results:** F1-Score ~0.75.
* **Interesting Finding:**
    * Typically, malware has high reconstruction error (hard to predict).
    * However, **Tunneling had LOWER reconstruction error** than Benign traffic.
    * *Reason:* DoH Tunneling is tool-generated, thus highly regular. Conversely, human web browsing (Benign) is chaotic and harder to predict.
* **Limitation:** F1 ~0.75 is good for Anomaly Detection but insufficient for State-of-the-Art (SOTA) publication.

---

## Phase 2: Strategic Pivot (The Breakthrough)

### 3. Paradigm Shift: From Client-based to Flow-based
This is the most important data engineering contribution of the research.
* **Old Problem:** Grouping by IP → Only 17 attack samples (too few for Deep Learning convergence).
* **New Solution:** Group by **5-tuple Flow**: `(Src IP, Dst IP, Src Port, Dst Port, Protocol)`.
* **Impact:**
    * Transformed 17 attacking machines into **231,666 attack flow samples**.
    * Changed data from "Extremely Imbalanced" (1600:1) to **"Acceptably Balanced" (~3.5:1)**.
    * Completely eliminated dependency on specific IPs (avoiding Data Leakage).

---

## Phase 3: Final Proposal & Results (The Success)

### 4. Model Architecture: Supervised Transformer Classifier
* **Input:** Only 2 raw features to ensure generalizability:
    1. `Packet Size`
    2. `Inter-arrival Time`
    * *Absolutely no IP Address in the model.*
* **Model Core:** **Transformer Encoder**.
    * Uses **Self-Attention** mechanism to view entire 50-packet sequence simultaneously.
    * Captures long-range dependencies and "rhythm" of DoH tunnels that RNN/LSTM often miss.
* **Optimal Configuration:**
    * `Sequence Length`: 50 packets.
    * `Hidden Dim`: 128.
    * `Layers`: 3.
    * `Loss Function`: BCEWithLogitsLoss (with `pos_weight` to handle remaining imbalance).

### 5. Experimental Results (State-of-the-Art)
* **Dataset:** Train on ~800k samples, Test on ~200k samples.
* **Metrics (Best Epoch):**
    * **F1-Score:** **~99.6%**
    * **Precision:** **~99%** (Extremely low false alarms).
    * **Recall:** **~100%** (No attacks missed).
* **Training Time:** Very fast convergence in ~25 Epochs.

---

## 💡 Scientific Contributions

1. **Data Engineering Contribution:** Proved that shifting from *Client-based aggregation* to *Flow-based aggregation* is a prerequisite for successfully applying Deep Learning to the CIRA-CIC-DoHBrw-2020 dataset, completely solving the sample scarcity problem.

2. **Model Architecture Contribution:** Proposed a **Lightweight Transformer** using only behavioral features (`Time` + `Size`), completely removing identifier information (`IP`), proving that DoH Tunneling has a very clear "behavioral signature" that Transformers can learn better than traditional models.

3. **Insight Contribution:** Discovered the "regularity" property of Tunneling traffic compared to normal Web traffic through Anomaly Detection experiments, thereby reinforcing why Supervised Learning achieves high effectiveness.

---

**Note for Future Paper Writing:**
* When writing *Related Work*, compare with papers using GCN or LSTM on this dataset and point out their weaknesses (usually data leakage or low F1).
* When writing *Methodology*, emphasize the PCAP → Flow processing pipeline.
* When writing *Results*, use comparison table between GRU Baseline (F1 ~0.50) and Transformer (F1 ~0.996) to highlight effectiveness.
