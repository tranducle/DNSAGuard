# Dataset Preparation

## CIRA-CIC-DoHBrw-2020 Dataset

This project uses the **CIRA-CIC-DoHBrw-2020** dataset for DNS-over-HTTPS tunneling detection.

### Download Instructions

1. Visit the official dataset page: [https://www.unb.ca/cic/datasets/dohbrw-2020.html](https://www.unb.ca/cic/datasets/dohbrw-2020.html)
2. Download the PCAP files for both benign and malicious traffic
3. Extract the files and organize them into the following structure:

```
DATASET/
├── Benign/
│   └── *.pcap
└── Malicious/
    └── *.pcap
```

### Preprocessing

After downloading the dataset, run the preprocessing pipeline:

```bash
# Step 1: Extract flows from PCAPs (This may take several hours)
python src/pcap_loader.py

# Step 2: Normalize and create Tensor datasets
python src/prepare_data.py
```

This will generate:
- `all_clients_raw.pkl`: Flow-based packet sequences
- `final_test_dataset.pt`: Preprocessed test set

### Dataset Statistics

- **Total Flows**: ~1M flows
- **Benign Flows**: ~800K
- **Malicious Flows**: ~231K
- **Imbalance Ratio**: ~3.5:1
- **Sequence Length**: 50 packets per flow
- **Features**: Packet Size, Inter-arrival Time
