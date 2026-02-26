# Physics-Informed GNN + LSTM Framework  
## Region-Based Stress–Strain Prediction for Short-Fiber Composites

This repository contains a **physics-guided Graph Neural Network (GNN) + LSTM framework** developed to predict the stress–strain response of short-fiber composites using microstructure-derived fiber features.

The framework links:

Microstructure (fiber geometry + region statistics)  
→ Graph representation (fiber interactions)  
→ GNN encoder (structural embedding)  
→ LSTM (strain-sequence learning)  
→ Predicted stress–strain curve  

---

## Objective

To develop a data-driven yet physics-informed surrogate model that:

- Captures fiber–fiber interactions through graph connectivity
- Encodes local microstructural heterogeneity at the region level
- Predicts full stress–strain curves (not just scalar properties)
- Enables fast surrogate evaluation compared to FEM

---

## Repository Contents

### Core Training Script
- **`GNN_final_version.py`**  
  Main training pipeline:
  - Graph construction from fiber features
  - GCN layers for spatial encoding
  - LSTM for strain-sequence modeling
  - Training/validation split
  - Loss computation
  - Model saving

### Application / Deployment Script
- **`GNN_final_version_app.py`**  
  Lightweight application version:
  - Loads trained model
  - Accepts region features
  - Predicts stress–strain response
  - Suitable for GUI or deployment

### Data Files
- **`RegionID_fiber.xlsx`**  
  Region-wise fiber data (fiber start/end, length, volume contribution, etc.)

- **`Static_data.xlsx`**  
  Experimental stress–strain data per RegionID

- **`region_embeddings.pkl`**  
  Saved learned region embeddings from trained GNN encoder

---

## Model Architecture

### 1) Input Features (per fiber node)

Each node represents one fiber.

Typical node features include:
- Fiber length
- Start coordinates (x, y, z)
- End coordinates (x, y, z)
- Local volume fraction contribution
- Optional: orientation descriptors

### 2) Graph Construction

Edges created using:
- Segment-to-segment minimum distance
- Distance threshold
- Exponential distance weighting

Output:
```
x (node feature matrix)
edge_index (connectivity)
edge_weight (distance-based weights)
````

### 3) GNN Encoder

Example structure:

- GCNConv layer 1 + ReLU
- GCNConv layer 2 + ReLU
- Global mean pooling
- Fully connected layer

Output:
- Region-level embedding vector

### 4) LSTM Module

Input:
- Region embedding
- Strain sequence

Output:
- Predicted stress sequence

---

## Requirements

### Python Version
Python 3.9+ recommended

### Required Packages

Install using:
```bash
pip install torch torchvision torchaudio
pip install torch-geometric
pip install pandas numpy scikit-learn matplotlib
````

## How to Run

### 1) Training

```bash
python GNN_final_version.py
```

This will:

* Load region + stress–strain data
* Construct graphs
* Train GNN + LSTM
* Save model weights
* Optionally export embeddings

Outputs may include:

* Trained model `.pt`
* `region_embeddings.pkl`
* Training loss plots

---

### 2) Application / Prediction

```bash
python GNN_final_version_app.py
```

This:

* Loads trained model
* Loads region features
* Predicts stress–strain response
* Outputs predicted curve

---

## Data Format Expectations

### RegionID_fiber.xlsx

Minimum required columns:

* RegionID
* FiberID
* Start_X, Start_Y, Start_Z
* End_X, End_Y, End_Z
* FiberLength
* VolumeContribution

### Static_data.xlsx

Minimum required columns:

* RegionID
* Strain
* Stress
---

## Training Strategy

* Region-based split (no data leakage across regions)
* MSE loss on stress prediction
* Optional regularization
* Early stopping (if enabled)

---

## Reproducibility

For consistent results:

* Fix random seed inside script
* Record:

  * PyTorch version
  * Torch Geometric version
  * Python version
  * GPU/CPU used

---

## Scientific Contribution

This framework enables:

* Direct mapping of measured fiber networks → stress–strain behavior
* Surrogate replacement of FEM for region-level modeling
* Scalable prediction for digital twin applications
* Integration with physics-based constraints

---

