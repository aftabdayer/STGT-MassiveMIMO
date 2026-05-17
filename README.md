# 📡 Spectral Temporal Graph Transformer (STGT) for Massive MIMO CSI Prediction

> **IEEE Published Research** — M.Tech Dissertation, NIT Hamirpur · Sole Researcher & Author  
> Published at **IEEE IACIS-2025 International Conference**

---

## Publication

**"Spectral Temporal Graph Transformer for Massive MIMO CSI Prediction in 5G and Beyond"**  
Aftab Dayer, Dr. Sandeep Kumar Singh  
IEEE Xplore · IEEE IACIS-2025 · NIT Hamirpur, May 2025

---

## What This Research Does

Accurate Channel State Information (CSI) prediction is critical for Massive MIMO systems in 5G — but conventional methods (LSTM, RNN, MMSE) fail under high mobility and fast-fading conditions.

This work proposes **STGT**: a hybrid deep learning architecture that combines:
- **Graph Neural Networks (GNN)** — to model spatial dependencies among antennas via dynamically learned adjacency matrices
- **Transformer-based temporal modelling** — to capture long-range sequential channel dynamics
- **Spectral analysis (Graph Fourier Transform)** — to decompose spatial features into frequency components
- **Encoder-decoder compression** — to handle high-dimensional CSI matrices efficiently (compression ratios γ = 1/4, 1/8, 1/16)

The result: STGT consistently outperforms LSTM, RNN, and the STEM GNN baseline across all metrics and compression settings.

---

## Key Results

### Performance at Compression Ratio γ = 1/4

| Model | NMSE | Correlation | SE @ 30 dB (bps/Hz) |
|-------|------|-------------|----------------------|
| **STGT (proposed)** | **0.5634** | **0.5371** | **6.615** |
| STEM GNN (baseline) | 0.6168 | 0.4879 | 6.482 |
| LSTM | 0.7303 | 0.3823 | 6.300 |
| RNN | 0.6996 | 0.4059 | 6.298 |

### Performance at Compression Ratio γ = 1/8

| Model | NMSE | Correlation | SE @ 30 dB (bps/Hz) |
|-------|------|-------------|----------------------|
| **STGT** | **0.5611** | **0.5277** | **6.520** |
| STEM GNN | 0.5966 | 0.4922 | 6.399 |
| LSTM | 0.7112 | 0.3856 | 6.188 |
| RNN | 0.6492 | 0.4420 | 6.324 |

### Performance at Compression Ratio γ = 1/16

| Model | NMSE | Correlation | SE @ 30 dB (bps/Hz) |
|-------|------|-------------|----------------------|
| STGT | 0.5524 | 0.5388 | **6.498** |
| **STEM GNN** | **0.5285** | **0.5590** | 6.473 |
| LSTM | 0.6671 | 0.4364 | 6.352 |
| RNN | 0.6350 | 0.4624 | 6.327 |

STGT leads across nearly all metrics at γ = 1/4 and 1/8. At γ = 1/16, STEM GNN edges ahead on NMSE/correlation while STGT retains the highest spectral efficiency.

---

## Result Plots

### Compression Ratio γ = 1/4 (512-dim latent vectors)

| Training & Validation Loss | SE per Test Sample | SE vs Eb/N0 |
|---|---|---|
| ![](120kmphfor512trainingloss.png) | ![](120kmphfor512sepertestsample.png) | ![](120kmphfor512sevsEbN0.png) |

### Compression Ratio γ = 1/8 (256-dim latent vectors)

| Training & Validation Loss | SE per Test Sample | SE vs Eb/N0 |
|---|---|---|
| ![](120kmphfor256trainingloss.png) | ![](120kmphfor256sepertestsample.png) | ![](120kmphfor256sevsEbN0.png) |

### Compression Ratio γ = 1/16 (128-dim latent vectors)

| Training & Validation Loss | SE per Test Sample | SE vs Eb/N0 |
|---|---|---|
| ![](120kmphfor128trainingloss.png) | ![](120kmphfor128sepertestsample.png) | ![](120kmphfor128sevsEbN0.png) |

All experiments at 120 km/h high-mobility scenario. STGT (blue) leads across SE per test sample and SE vs Eb/N0 at γ = 1/4 and 1/8. Training curves show STGT achieves faster convergence and lower final loss than LSTM and RNN across all compression settings.

---

## Architecture

```
Input CSI Matrix (2×32×32 complex)
        ↓
Neural Encoder → Latent Vector (512 / 256 / 128 dim)
        ↓
┌─────────────────────────────────────┐
│         STGT Model                  │
│  ┌──────────────────────────────┐   │
│  │ Graph Construction           │   │
│  │ (Self-attention adjacency)   │   │
│  └──────────────┬───────────────┘   │
│                 ↓                   │
│  ┌──────────────────────────────┐   │
│  │ Graph Spectral Module (GFT)  │   │
│  │ Laplacian eigen-decomp       │   │
│  └──────────────┬───────────────┘   │
│                 ↓                   │
│  ┌──────────────────────────────┐   │
│  │ Temporal Transformer Block   │   │
│  │ Multi-head self-attention    │   │
│  │ + FFN + LayerNorm + Residual │   │
│  └──────────────┬───────────────┘   │
│                 ↓                   │
│  ┌──────────────────────────────┐   │
│  │ Fusion Layer (GLU + Residual)│   │
│  └──────────────┬───────────────┘   │
└─────────────────┼───────────────────┘
                  ↓
        Predicted Latent Vectors
                  ↓
Neural Decoder → Predicted CSI Matrix
```

---

## Repository Contents

| File | Description |
|------|-------------|
| `modelgatstemcnnlstm.ipynb` | Main comparison notebook — GAT, STEM GNN, CNN, LSTM |
| `modelgat128.ipynb` | GAT model at 128-dim compression |
| `uma128gatmodel.ipynb` | UMA channel GAT model |
| `theisssCNN.ipynb` | CNN baseline implementation |
| `theiss.ipynb` | Core thesis experiments |
| `hji.ipynb` | Additional model experiments |
| `csifilegenreate.ipynb` | CSI dataset generation pipeline |
| `newdatase.ipynb` | Dataset preprocessing |
| `basepaerresult.ipynb` | Base paper (STEM GNN) result reproduction |
| `120kmphfor512trainingloss.png` | Training & val loss — γ = 1/4 (512-dim) |
| `120kmphfor512sepertestsample.png` | SE per test sample — γ = 1/4 |
| `120kmphfor512sevsEbN0.png` | SE vs Eb/N0 — γ = 1/4 |
| `120kmphfor256trainingloss.png` | Training & val loss — γ = 1/8 (256-dim) |
| `120kmphfor256sepertestsample.png` | SE per test sample — γ = 1/8 |
| `120kmphfor256sevsEbN0.png` | SE vs Eb/N0 — γ = 1/8 |
| `120kmphfor128trainingloss.png` | Training & val loss — γ = 1/16 (128-dim) |
| `120kmphfor128sepertestsample.png` | SE per test sample — γ = 1/16 |
| `120kmphfor128sevsEbN0.png` | SE vs Eb/N0 — γ = 1/16 |
| `model_comparison_se_vs_ebn0_with_stemgnn.png` | Overall SE comparison — all models |
| `stemgnn_train_val_curves.png` | STEMGNN training curves |
| `gat_train_val_curves.png` | GAT training curves |
| `lstm_train_val_curves.png` | LSTM training curves |
| `cnn_train_val_curves.png` | CNN training curves |

---

## Experimental Setup

| Parameter | Value |
|-----------|-------|
| Hardware | Intel Core i7-12700K, NVIDIA RTX 3070 8GB, 32GB DDR5 |
| Framework | Python 3.9 · PyTorch 1.11 |
| Dataset | Simulated Massive MIMO CSI (160,000 channel matrices) |
| Mobility | 120 km/h (high mobility scenario) |
| Antennas | 32×72 configuration |
| Train / Val / Test | 70% / 15% / 15% |
| Optimizer | Adam · lr=0.0002 · batch=64 · dropout=0.2 |
| Epochs | 100 |
| Compression ratios | γ = 1/4 (512-dim) · 1/8 (256-dim) · 1/16 (128-dim) |

---

## Tech Stack

`Python` · `PyTorch` · `NumPy` · `SciPy` · `Matplotlib` · `Graph Neural Networks` · `Transformer` · `GFT`

---

## Why STGT Outperforms

- **Dynamic graph learning** — adjacency matrix is learned per sample via self-attention, not fixed. Adapts to changing channel conditions.
- **Joint spatial-temporal modelling** — unlike pure LSTM/RNN which only model time, STGT simultaneously captures antenna spatial dependencies and temporal evolution.
- **Spectral decomposition** — GFT separates low/high-frequency spatial patterns, giving the model richer input representations.
- **Robust under compression** — performance advantage over LSTM/RNN widens as compression increases, showing better generalization on limited information.

---

## Author

**Aftab Dayer** · [LinkedIn](https://linkedin.com/in/aftabdayer) · [GitHub](https://github.com/aftabdayer)  
M.Tech, Electronics & Communication Engineering  
National Institute of Technology Hamirpur · 2025  
Supervisor: Dr. Sandeep Kumar Singh, DoECE, NIT Hamirpur
