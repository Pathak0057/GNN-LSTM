# GNN_LSTM — Physics-Informed GNN + LSTM Stress–Strain Predictor

## What This Folder Does

This folder contains a machine learning pipeline that predicts the fullstress–strain curve of individual Voronoi regions of a short-fiber reinforced ABS composite, using only the region's fiber microstructure geometry (fiber endpoint coordinates and volume fraction) as input.

The model combines two architectures working in sequence:

- Graph Neural Network (GNN): Each region's fibers are represented as a graph — fibers are nodes, inter-fiber proximity is encoded as weighted edges. A two-layer GCN processes this graph and produces a fixed-length region embedding vector that encodes the spatial arrangement and connectivity of fibers in that region.
- LSTM (Long Short-Term Memory): Takes the region embedding together with strain values and physics-derived descriptors (elastic modulus, yield stress, area under curve) as a time-sequence input. Predicts the stress value at each strain point along the curve.

The training loss includes a physics-informed penalty that penalises non-monotone stress predictions (stress must not decrease with increasing strain in the pre-failure regime) and directly penalises errors in the predicted elastic modulus, yield stress, and energy absorption (AUC).

---

## Files in This Folder

| File | Type | Description |
|---|---|---|
| `GNN_final_version.py` | Python script | Main training and evaluation script. Trains the GNN+LSTM model from scratch, saves checkpoints, evaluates all regions, and launches a Tkinter GUI for interactive region inspection and custom microstructure prediction. |
| `GNN_final_version_app.py` | Python script | Standalone GUI application. A clean wrapper that loads a pre-trained checkpoint and launches the interactive interface without re-training. Run this after training is complete. |
| `RegionID_fiber.xlsx` | Excel input | Fiber geometry data. One row per Voronoi region. Contains the 3D fiber start/end point coordinates and the region fiber volume fraction (Vf). Sheet name: `RegionID_fiber`. |
| `Static_data.xlsx` | Excel input | Stress–strain training data. Contains the FE-simulated (or experimental) stress–strain curves for each region across multiple strain points. Sheet name: `Stress-Strain_Data_for_GNN`. |
| `region_embeddings.pkl` | Pickle cache | Pre-computed GNN region embeddings. Saved automatically after training. If present, the app script loads this directly instead of recomputing embeddings. Delete this file to force recomputation. |

---
## How to Run

### Step 1 — Install Dependencies

```bash
pip install torch torchvision torchaudio
pip install torch-geometric
pip install numpy pandas openpyxl matplotlib scikit-learn
```

For `torch-geometric`, use the version matching your PyTorch and CUDA installation. See https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html.

### Step 2 — Place Input Files

Ensure these two files are in the same folder as the scripts:

```
GNN_LSTM/
├── GNN_final_version.py
├── GNN_final_version_app.py
├── RegionID_fiber.xlsx
├── Static_data.xlsx
└── region_embeddings.pkl     ← optional cache, ignored if not present
```

### Step 3 — Train the Model

```bash
python GNN_final_version.py
```

This will:
1. Load both Excel files and build the fiber graphs.
2. Compute GNN embeddings for every region.
3. Train the LSTM model for up to 300 epochs with early stopping (patience = 25).
4. Save the best checkpoint to `checkpoints/best_model.pth`.
5. Train the physics regressor and sequence calibrators.
6. Print per-region R² scores and the global R² across all data points.
7. Show a training/validation loss plot.
8. Launch the interactive GUI automatically.

Expected training time: a few minutes on CPU for ~27 regions with 16 strain points each.

### Step 4 — Run the GUI App (after training)

Once `checkpoints/best_model.pth` exists, you can launch the standalone interface without retraining:

```bash
python GNN_final_version_app.py
```

---
## Generated Files (Created Automatically on First Run)

These files do not need to be provided — they are created and saved by the scripts:

| File / Folder | Created by | Description |
|---|---|---|
| `checkpoints/best_model.pth` | `GNN_final_version.py` | PyTorch state dictionary of the best LSTM model (lowest validation loss) |
| `checkpoints/physics_regressor.pkl` | `GNN_final_version.py` | Trained Random Forest that maps (Vf + GNN embedding) → [E, σy, AUC] for new microstructures |
| `checkpoints/seq_calibrators.pkl` | `GNN_final_version.py` | Per-strain-index isotonic regression calibrators that correct systematic bias in predictions |
| `checkpoints/train_losses.npy` | `GNN_final_version.py` | Training loss history (one value per epoch) |
| `checkpoints/val_losses.npy` | `GNN_final_version.py` | Validation loss history |
| `checkpoints/last_path.txt` | `GNN_final_version.py` | Records the actual path used for the best model checkpoint (handles Windows path issues) |

---

## Input Data Format

### `RegionID_fiber.xlsx` — Sheet: `RegionID_fiber`

| Column | Format | Description |
|---|---|---|
| `RegionID` | Integer | Unique region identifier (must match IDs in Static_data.xlsx) |
| `Vf` | Float (0–1) | Fiber volume fraction for this region |
| `FiberStartPoint` | String `[[x,y,z],[x,y,z],...]` | Python-literal list of 3D start coordinates for all fibers in this region |
| `FiberEndPoint` | String `[[x,y,z],[x,y,z],...]` | Python-literal list of 3D end coordinates (same order as start points) |

Each region occupies exactly one row. The `FiberStartPoint` and `FiberEndPoint` columns store lists of 3D coordinates as Python-literal strings (e.g., `[[10.5, -3.2, 0.0], [20.1, 5.6, 8.3]]`).

### `Static_data.xlsx` — Sheet: `Stress-Strain_Data_for_GNN`

| Column | Format | Description |
|---|---|---|
| `RegionID` | Integer | Region identifier (same as in fiber file) |
| `Vf` | Float (0–1) | Fiber volume fraction (repeated for each strain point of the region) |
| `Strain` | Float | Engineering strain value at this data point |
| `Stress` | Float (MPa) | Corresponding stress value from FE simulation or experiment |

One row per (region, strain point) pair. All strain points for a region must be in the same order (increasing strain). The model uses 16 strain points per region by default (0 to 2.0).

---

## GUI Features

The GUI has two tabs:

### Tab 1 — Region Validation (True vs Predicted)
Enter any `RegionID` from the training data. The app plots:
- The true vs predicted stress–strain curve with R² score.
- A scatter plot of predicted vs true stress values.
- The fiber-fiber adjacency matrix heatmap for that region.

### Tab 2 — Custom Microstructure Prediction
Input a new microstructure not seen during training:
- Set the number of fibers, volume fraction (Vf), and strain range.
- Enter start and end coordinates for each fiber (format: `x,y,z`).
- Click **Predict** to get the full predicted stress–strain curve and adjacency matrix.

If the input exactly matches an existing training region, the app uses the trained LSTM directly and shows both true and predicted curves. Otherwise, it uses the physics regressor + calibrated prediction pipeline.

---

## Model Architecture Summary

```
Per-region fiber graph
  Nodes: [length, Vf_fiber, x1, y1, z1, x2, y2, z2]  (8 features per fiber)
  Edges: weighted by inter-fiber proximity
        ↓
  FiberGNN  (GCNConv × 2 → global_mean_pool → Linear)
        ↓
  Region embedding  (16-dim vector)
        ↓
  Concatenate per strain point:
    [strain, strain², strain³, log(1+strain), Vf, embedding(×16), E, σy, AUC]
    → input_dim = 5 + 16 + 3 = 24
        ↓
  StressPredictor  (LSTM 2-layer 128-dim → FC 128→64→1)
        ↓
  Predicted stress at each strain point
```

**Physics-informed loss:**
```
L_total = MSE(σ_pred, σ_true)
        + 0.1 × monotonicity_penalty    (penalise dσ/dε < 0)
        + 0.2 × physics_constraint      (penalise errors in E, σy, AUC)
```

---

## Notes

- The model trains an 80/20 region-level split. With ~27 regions, roughly 22 train and 5 validate.
- `region_embeddings.pkl` stores pre-computed GNN embeddings. If the fiber data changes, delete this file so embeddings are recomputed.
- Checkpoints are saved to `./checkpoints/` by default. If that directory is not writable (Windows permission issues), the script falls back to `~/GNN_checkpoints/`.
- The sequence calibrators (`seq_calibrators.pkl`) apply per-strain-point isotonic regression corrections on top of the LSTM output. Disable them with the "Disable calibration" checkbox in the GUI to see raw model predictions.
