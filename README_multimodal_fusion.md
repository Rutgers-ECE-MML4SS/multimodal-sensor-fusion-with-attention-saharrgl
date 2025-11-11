# Multimodal Sensor Fusion with Attention on PAMAP2

This repository implements and evaluates **multimodal sensor fusion** architectures for human activity recognition on the **PAMAP2** dataset.  
It includes Early, Late, and Hybrid (Attention-based) fusion strategies, uncertainty calibration, and attention interpretability.

---

## 📁 Repository Structure

```
.
├── config/                  # YAML experiment configs
├── data/                    # PAMAP2 dataset (raw + preprocessed)
├── src/                     # Source code (encoders, fusion, training, etc.)
├── analysis/                # Plots and evaluation outputs
├── experiments/             # JSON logs and experiment summaries
└── README.md
```

---

## ⚙️ Setup

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install pytorch-lightning hydra-core numpy scipy scikit-learn matplotlib tqdm
```

---

## 📊 Training Examples

### Early Fusion
```bash
python src/train.py model.fusion_type=early device=cuda
```

### Late Fusion
```bash
python src/train.py model.fusion_type=late device=cuda
```

### Hybrid Fusion (Attention-based)
```bash
python src/train.py model.fusion_type=hybrid device=cuda
```

Each run saves results under `runs/a2_<type>_pamap2/`.

---

## 📈 Evaluation

```bash
python src/eval.py --checkpoint runs/a2_hybrid_pamap2/checkpoints/last.ckpt --device cuda
```

Generates metrics:
- Accuracy
- F1 (macro)
- Loss
- ECE (Expected Calibration Error)

Results stored in `analysis/<fusion_type>/evaluation_results.json`.

---

## 🧠 Visualization

### Fusion Comparison Plot
```bash
python src/experiments_fusion.py
```
Outputs: `analysis/fusion_comparison.png`

### Attention Visualization
```bash
python src/visualize_attention.py
```
Outputs: `analysis/attention_viz.png`

---

## 📉 Typical Results (PAMAP2)

| Fusion Type   | Accuracy | F1 (Macro) | ECE   | Params (M) |
|----------------|----------|-----------:|------:|-----------:|
| Early          | 0.7514   | 0.732      | 0.030 | 1.10 |
| Late           | 0.7514   | 0.736      | 0.032 | 1.11 |
| Hybrid (Attn)  | 0.7636   | 0.748      | 0.020 | 1.18 |

---

## ✅ Highlights

- **Graceful Degradation**: <10% accuracy drop when an IMU fails.  
- **Calibration**: ECE = 0.03 → excellent confidence calibration.  
- **Interpretability**: Attention maps align with motion semantics (e.g., ankle = locomotion).

---

## ⚠️ Limitations

- Heart-rate interpolation required due to low frequency.  
- CPU training ≈ 25 min per model.

---

## 📚 Citation

**Dataset:**  (https://archive.ics.uci.edu/ml/datasets/pamap2+physical+activity+monitoring)
