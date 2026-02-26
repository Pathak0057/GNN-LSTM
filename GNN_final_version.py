# ==================================
# Physics-Informed GNN + LSTM Stress Prediction Model
# Robust Version with Stronger Physics Constraints
# ==================================
import ast, os, random, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from itertools import combinations
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.isotonic import IsotonicRegression
from mpl_toolkits.mplot3d import Axes3D

# ---------------------------
# Reproducibility
# ---------------------------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
set_seed(42)

# ---------------------------
# Checkpoint path helpers (robust save/load on Windows)
# ---------------------------
def _primary_ckpt_path():
    return os.path.join("checkpoints", "best_model.pth")

def _fallback_ckpt_path():
    home = os.path.expanduser("~")
    return os.path.join(home, "GNN_checkpoints", "best_model.pth")

def _record_last_ckpt(path: str):
    try:
        os.makedirs("checkpoints", exist_ok=True)
        with open(os.path.join("checkpoints", "last_path.txt"), "w", encoding="utf-8") as f:
            f.write(path)
    except Exception:
        pass

def _read_last_ckpt():
    try:
        with open(os.path.join("checkpoints", "last_path.txt"), "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return None

# ---------------------------
# Load Excel files
# ---------------------------
fiber_file = "RegionID_fiber.xlsx"
static_file = "Static_data.xlsx"
fiber_df = pd.read_excel(fiber_file, sheet_name="RegionID_fiber")
stress_strain_df = pd.read_excel(static_file, sheet_name="Stress-Strain_Data_for_GNN")

# ---------------------------
# Fiber graph functions
# ---------------------------
def parse_point_str(point_str): return np.array(ast.literal_eval(point_str), dtype=float)
def fiber_length(start, end): return np.linalg.norm(start - end)

def segment_distance(p1, p2, q1, q2):
    u, v = p2 - p1, q2 - q1; w0 = p1 - q1
    a, b, c = np.dot(u,u), np.dot(u,v), np.dot(v,v)
    d, e = np.dot(u,w0), np.dot(v,w0)
    denom = a*c - b*b
    if denom == 0: sc, tc = 0, d/b if b!=0 else 0
    else: sc, tc = (b*e - c*d)/denom, (a*e - b*d)/denom
    sc, tc = np.clip(sc,0,1), np.clip(tc,0,1)
    dP = w0 + sc*u - tc*v
    return np.linalg.norm(dP)

def build_region_graph(region_df):
    Vf_region = region_df["Vf"].iloc[0]
    starts = np.array(parse_point_str(region_df["FiberStartPoint"].iloc[0]))
    ends   = np.array(parse_point_str(region_df["FiberEndPoint"].iloc[0]))
    if starts.ndim == 1: starts = starts.reshape(1, -1)
    if ends.ndim == 1: ends = ends.reshape(1, -1)

    n_fibers = min(len(starts), len(ends))
    lengths = [fiber_length(starts[i], ends[i]) for i in range(n_fibers)]
    total_length = sum(lengths) if n_fibers > 0 else 1.0
    fiber_vfs = [Vf_region * (l / total_length) for l in lengths]

    x = [[lengths[i], fiber_vfs[i]] + starts[i].tolist() + ends[i].tolist()
         for i in range(n_fibers)]
    x = torch.tensor(x, dtype=torch.float) if x else torch.empty((0, 8), dtype=torch.float)

    edge_index, edge_attr = [], []
    for i, j in combinations(range(n_fibers), 2):
        d = segment_distance(starts[i], ends[i], starts[j], ends[j])
        avg_len = 0.5 * (lengths[i] + lengths[j])
        w = 1.0 if d < 0.4 * avg_len else np.exp(-d / avg_len)
        edge_index += [[i, j],[j, i]]; edge_attr += [[w],[w]]

    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr  = torch.tensor(edge_attr, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr  = torch.empty((0, 1), dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                y=torch.tensor([Vf_region], dtype=torch.float))
    return data, starts, ends, fiber_vfs

# ---- Unified graph building function for consistency ----
def build_graph_from_coordinates(fiber_starts, fiber_ends, Vf):
    """
    Unified function to build graph from coordinates - ensures consistency
    across training, validation, and prediction
    """
    # Convert inputs to numpy arrays and ensure consistent format
    starts = np.array(fiber_starts)
    ends = np.array(fiber_ends)
    
    # Ensure proper shape (same as training data processing)
    if starts.ndim == 1:
        starts = starts.reshape(1, -1)
    if ends.ndim == 1:
        ends = ends.reshape(1, -1)
    
    # Use the SAME graph building logic as training
    n_fibers = min(len(starts), len(ends))
    lengths = [fiber_length(starts[i], ends[i]) for i in range(n_fibers)]
    total_length = sum(lengths) if n_fibers > 0 else 1.0
    fiber_vfs = [Vf * (l / total_length) for l in lengths]

    # Create node features (EXACT same format as training)
    x = [[lengths[i], fiber_vfs[i]] + starts[i].tolist() + ends[i].tolist()
         for i in range(n_fibers)]
    x = torch.tensor(x, dtype=torch.float) if x else torch.empty((0, 8), dtype=torch.float)

    # Create edge connections (EXACT same logic as training)
    edge_index, edge_attr = [], []
    for i, j in combinations(range(n_fibers), 2):
        d = segment_distance(starts[i], ends[i], starts[j], ends[j])
        avg_len = 0.5 * (lengths[i] + lengths[j])
        w = 1.0 if d < 0.4 * avg_len else np.exp(-d / avg_len)
        edge_index += [[i, j], [j, i]]
        edge_attr += [[w], [w]]

    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float)

    # Create graph data (EXACT same format as training)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                y=torch.tensor([Vf], dtype=torch.float))
    
    return data, starts, ends, fiber_vfs

# ---- Extended: return adjacency matrix too ----
def build_region_graph_with_adj(region_df):
    Vf_region = region_df["Vf"].iloc[0]
    starts = parse_point_str(region_df["FiberStartPoint"].iloc[0])
    ends   = parse_point_str(region_df["FiberEndPoint"].iloc[0])
    n_fibers = min(len(starts), len(ends))

    lengths = [fiber_length(starts[i], ends[i]) for i in range(n_fibers)]
    total_length = sum(lengths) if n_fibers > 0 else 1.0
    fiber_vfs = [Vf_region * (l / total_length) for l in lengths]

    x = [[lengths[i], fiber_vfs[i]] + list(starts[i]) + list(ends[i]) for i in range(n_fibers)]
    x = torch.tensor(x, dtype=torch.float) if x else torch.empty((0,8), dtype=torch.float)

    edge_index, edge_attr = [], []
    adj_matrix = np.zeros((n_fibers, n_fibers))
    for i, j in combinations(range(n_fibers), 2):
        d = segment_distance(starts[i], ends[i], starts[j], ends[j])
        avg_len = 0.5*(lengths[i]+lengths[j])
        w = 1.0 if d < 0.4*avg_len else np.exp(-d/avg_len)
        adj_matrix[i,j]=w; adj_matrix[j,i]=w
        edge_index += [[i,j],[j,i]]; edge_attr += [[w],[w]]
    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr  = torch.tensor(edge_attr, dtype=torch.float)
    else:
        edge_index = torch.empty((2,0), dtype=torch.long)
        edge_attr  = torch.empty((0,1), dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                y=torch.tensor([Vf_region], dtype=torch.float))
    return data, adj_matrix, starts, ends, fiber_vfs

# ---------------------------
# GNN for region embedding
# ---------------------------
class FiberGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels=32, out_channels=16):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc = nn.Linear(hidden_channels, out_channels)
    def forward(self, x, edge_index, edge_weight, batch=None):
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.relu(self.conv2(x, edge_index, edge_weight))
        if batch is None: batch = torch.zeros(x.size(0), dtype=torch.long)
        return self.fc(global_mean_pool(x, batch))

region_embeddings = {}
first_region = fiber_df[fiber_df["RegionID"]==fiber_df["RegionID"].unique()[0]]
temp_graph,_,_,_ = build_region_graph(first_region)
gnn_model = FiberGNN(in_channels=temp_graph.x.shape[1])
for rid, region_df in fiber_df.groupby("RegionID"):
    g,_,_,_ = build_region_graph(region_df)
    emb = gnn_model(g.x, g.edge_index, g.edge_attr.view(-1))
    region_embeddings[rid] = emb.detach()
embedding_dim = list(region_embeddings.values())[0].shape[1]

# ---------------------------
# Physics property functions
# ---------------------------
def compute_modulus(strain, stress):
    return (stress[2] - stress[0]) / (strain[2] - strain[0] + 1e-8)

def compute_yield_point(strain, stress):
    E = compute_modulus(strain, stress)
    slopes = np.gradient(stress, strain+1e-8)
    idx = np.argmax(slopes < 0.8*E)
    if idx == 0: idx = np.argmax(stress)
    return stress[idx]

def compute_auc(strain, stress):
    return np.trapz(stress, strain)

# ---------------------------
# Dataset with augmented physics features
# ---------------------------
class StressStrainDataset(Dataset):
    def __init__(self, df, embeddings):
        self.df = df; self.embeddings = embeddings
        self.region_ids = df["RegionID"].unique()
    def __len__(self): return len(self.region_ids)
    def __getitem__(self, idx):
        rid = self.region_ids[idx]
        sub = self.df[self.df["RegionID"]==rid]
        strain = sub["Strain"].values; stress = sub["Stress"].values
        strain_t = torch.tensor(strain, dtype=torch.float).unsqueeze(-1)
        stress_t = torch.tensor(stress, dtype=torch.float).unsqueeze(-1)

        # physics descriptors (true values, repeated across sequence)
        E_val = compute_modulus(strain, stress)
        sigma_y_val = compute_yield_point(strain, stress)
        auc_val = compute_auc(strain, stress)
        physics_feats = torch.tensor([E_val, sigma_y_val, auc_val], dtype=torch.float)
        physics_feats = physics_feats.repeat(len(strain),1)

        # strain-derived features
        strain_sq = strain_t**2; strain_cu = strain_t**3; strain_log = torch.log1p(strain_t)
        Vf_region = fiber_df[fiber_df["RegionID"] == rid]["Vf"].iloc[0]
        Vf_val = torch.tensor([Vf_region], dtype=torch.float).repeat(len(strain),1)
        emb = self.embeddings[rid].repeat(len(strain),1)

        x = torch.cat([strain_t, strain_sq, strain_cu, strain_log, Vf_val, emb, physics_feats], dim=1)
        return x, stress_t

dataset = StressStrainDataset(stress_strain_df, region_embeddings)

# ---------------------------
# Train physics descriptor regressor (from microstructure -> [E, sigma_y, AUC])
# ---------------------------
def build_physics_regressor(embeddings_map, df):
    """Train a regressor to predict [E, sigma_y, AUC] from (Vf + GNN embedding).
    Saves to checkpoints/physics_regressor.pkl and returns the fitted model.
    """
    # Aggregate per-region physics labels using dataset logic
    X_rows, Y_rows = [], []
    for rid in sorted(df["RegionID"].unique()):
        sub = df[df["RegionID"] == rid]
        strain = sub["Strain"].values
        stress = sub["Stress"].values
        E_val = compute_modulus(strain, stress)
        sigma_y_val = compute_yield_point(strain, stress)
        auc_val = compute_auc(strain, stress)

        Vf_val = float(sub["Vf"].iloc[0]) if "Vf" in sub.columns else float(fiber_df[fiber_df["RegionID"]==rid]["Vf"].iloc[0])
        emb = embeddings_map[rid].cpu().numpy().reshape(-1)
        X_rows.append(np.concatenate([[Vf_val], emb]))
        Y_rows.append([E_val, sigma_y_val, auc_val])

    if not X_rows:
        return None

    X = np.asarray(X_rows)
    Y = np.asarray(Y_rows)

    # Simple, robust multi-output regressor
    base = RandomForestRegressor(n_estimators=300, random_state=42)
    model = MultiOutputRegressor(base)
    model.fit(X, Y)

    os.makedirs("checkpoints", exist_ok=True)
    with open(os.path.join("checkpoints", "physics_regressor.pkl"), "wb") as f:
        pickle.dump(model, f)
    return model

def load_physics_regressor():
    path = os.path.join("checkpoints", "physics_regressor.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return build_physics_regressor(region_embeddings, stress_strain_df)

# ---------------------------
# Sequence calibrators (maps LSTM option-2 predictions -> ground truth per strain index)
# ---------------------------
def build_sequence_calibrators():
    """Fit per-strain isotonic regressors using current pipeline predictions vs true.
    Saves to checkpoints/seq_calibrators.pkl and returns list of calibrators.
    """
    # Ensure physics regressor is ready
    _ = load_physics_regressor()
    # Assume all regions share same number of strain points
    num_points = None
    preds_by_idx = {}
    trues_by_idx = {}

    for rid in sorted(dataset.region_ids):
        x, y_true = dataset[list(dataset.region_ids).index(rid)]
        strain = x[:,0].cpu().numpy()
        region_df = fiber_df[fiber_df["RegionID"] == rid]
        starts = parse_point_str(region_df["FiberStartPoint"].iloc[0])
        ends = parse_point_str(region_df["FiberEndPoint"].iloc[0])
        Vf = float(region_df["Vf"].iloc[0])
        # Predict using option-2 pipeline with learned regressor and same strain range
        # IMPORTANT: disable calibration here to avoid recursive building
        res = predict_stress_strain(
            starts, ends, Vf,
            strain_range=strain,
            use_training_approach=False,
            use_similar_region=False,
            disable_calibration=True,
        )
        y_pred = np.asarray(res['predicted_stress'])
        y_true_np = y_true.squeeze().cpu().numpy()
        if num_points is None:
            num_points = len(y_pred)
        for i in range(num_points):
            preds_by_idx.setdefault(i, []).append(y_pred[i])
            trues_by_idx.setdefault(i, []).append(y_true_np[i])

    calibrators = []
    for i in range(num_points):
        iso = IsotonicRegression(increasing=True)
        calibrators.append(iso.fit(preds_by_idx[i], trues_by_idx[i]))

    os.makedirs("checkpoints", exist_ok=True)
    with open(os.path.join("checkpoints", "seq_calibrators.pkl"), "wb") as f:
        pickle.dump(calibrators, f)
    return calibrators

def load_sequence_calibrators():
    path = os.path.join("checkpoints", "seq_calibrators.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return build_sequence_calibrators()

# ---------------------------
# Curve shaping utilities
# ---------------------------
def enforce_monotone_convex(stress: np.ndarray, strain: np.ndarray) -> np.ndarray:
    """Enforce positive slope and non-increasing slope (convex, saturating).
    Keeps endpoints and integrates adjusted slopes back to stress.
    """
    stress = np.asarray(stress).astype(float)
    strain = np.asarray(strain).astype(float)
    if stress.ndim != 1 or strain.ndim != 1 or len(stress) != len(strain):
        return stress
    # Compute slopes
    ds = np.diff(stress)
    dt = np.diff(strain)
    dt[dt == 0] = 1e-8
    slopes = ds / dt
    # Enforce slopes >= 0 using isotonic on negatives to also enforce decreasing
    try:
        # First, clip negatives to zero coarse
        slopes = np.maximum(slopes, 0.0)
        # Now, make them non-increasing
        iso = IsotonicRegression(increasing=False)
        slopes_adj = iso.fit_transform(np.arange(len(slopes)), slopes)
    except Exception:
        slopes_adj = np.maximum(slopes, 0.0)
    # Reconstruct curve
    stress_new = np.empty_like(stress)
    stress_new[0] = max(stress.min(), 0.0)
    stress_new[1:] = stress_new[0] + np.cumsum(slopes_adj * dt)
    return stress_new

# ---------------------------
# LSTM Model
# ---------------------------
class StressPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        out, _ = self.lstm(x); return self.fc(out)

input_dim = 5 + embedding_dim + 3
model = StressPredictor(input_dim)

# ---------------------------
# Loss Functions
# ---------------------------
def physics_informed_loss(y_pred, y_true):
    mse = torch.mean((y_pred - y_true)**2)
    diff = y_pred[:,1:] - y_pred[:,:-1]
    monotonicity_penalty = torch.mean(F.relu(-diff))
    return mse + 0.1*monotonicity_penalty

def physics_constraints(y_pred, strain, y_true):
    y_pred = y_pred.squeeze().detach().cpu().numpy()
    y_true = y_true.squeeze().detach().cpu().numpy()
    strain = strain.detach().cpu().numpy()
    E_pred, E_true = compute_modulus(strain,y_pred), compute_modulus(strain,y_true)
    sigma_y_pred, sigma_y_true = compute_yield_point(strain,y_pred), compute_yield_point(strain,y_true)
    auc_pred, auc_true = compute_auc(strain,y_pred), compute_auc(strain,y_true)
    return 0.1*((E_pred-E_true)**2 + (sigma_y_pred-sigma_y_true)**2 + (auc_pred-auc_true)**2)

# ---------------------------
# Training with Early Stopping
# ---------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
num_epochs, patience = 300, 25
best_val_loss, epochs_no_improve, best_epoch = float("inf"), 0, -1
train_losses, val_losses = [], []

ids = sorted(list(dataset.region_ids))
split_idx = int(0.8*len(ids))
train_ids, val_ids = ids[:split_idx], ids[split_idx:]
train_idx = [list(dataset.region_ids).index(r) for r in train_ids]
val_idx   = [list(dataset.region_ids).index(r) for r in val_ids]
train_loader = DataLoader(Subset(dataset, train_idx), batch_size=1, shuffle=True)
val_loader   = DataLoader(Subset(dataset, val_idx), batch_size=1, shuffle=False)

print(f"Training regions: {len(train_idx)}, Validation regions: {len(val_idx)}")
for epoch in range(num_epochs):
    model.train(); total_train_loss=0
    for x,y in train_loader:
        optimizer.zero_grad(); y_pred = model(x)
        loss = physics_informed_loss(y_pred,y)
        strain_tensor = x[0,:,0]
        constr_loss = physics_constraints(y_pred,strain_tensor,y)
        total_loss = loss + 0.2*constr_loss
        total_loss.backward(); optimizer.step()
        total_train_loss += total_loss.item()
    avg_train_loss = total_train_loss/len(train_loader)

    model.eval(); total_val_loss=0
    with torch.no_grad():
        for x,y in val_loader:
            y_pred = model(x)
            loss = physics_informed_loss(y_pred,y)
            strain_tensor = x[0,:,0]
            constr_loss = physics_constraints(y_pred,strain_tensor,y)
            total_loss = loss + 0.2*constr_loss
            total_val_loss += total_loss.item()
    avg_val_loss = total_val_loss/len(val_loader)
    train_losses.append(avg_train_loss); val_losses.append(avg_val_loss)

    scheduler.step(avg_val_loss)
    if avg_val_loss<best_val_loss:
        best_val_loss, best_epoch, epochs_no_improve = avg_val_loss, epoch+1, 0
        # Robust save with fallback directory
        primary = _primary_ckpt_path()
        try:
            os.makedirs(os.path.dirname(primary), exist_ok=True)
            torch.save(model.state_dict(), primary)
            _record_last_ckpt(primary)
        except Exception:
            fb = _fallback_ckpt_path()
            os.makedirs(os.path.dirname(fb), exist_ok=True)
            torch.save(model.state_dict(), fb)
            _record_last_ckpt(fb)
    else: epochs_no_improve+=1

    if (epoch+1)%10==0 or epoch==0:
        print(f"Epoch {epoch+1:03d}: Train={avg_train_loss:.4f}, Val={avg_val_loss:.4f} (Best={best_val_loss:.4f} @ {best_epoch})")
    if epochs_no_improve>=patience:
        print(f"Early stopping at epoch {epoch+1}, best model was from epoch {best_epoch}"); break

with open("region_embeddings.pkl","wb") as f: pickle.dump(region_embeddings,f)
print("Saved region embeddings as region_embeddings.pkl")

# --- Plot training curves ---
plt.figure(figsize=(6,4))
plt.plot(train_losses,label="Train Loss")
plt.plot(val_losses,label="Validation Loss")
plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.title("Training & Validation Loss")
plt.legend(); plt.tight_layout(); plt.show()

# Save loss curves for GUI
try:
    os.makedirs("checkpoints", exist_ok=True)
    np.save(os.path.join("checkpoints", "train_losses.npy"), np.array(train_losses))
    np.save(os.path.join("checkpoints", "val_losses.npy"), np.array(val_losses))
except Exception:
    pass

# ---------------------------
# Unified Visualization: Stress-Strain + Adjacency Heatmap
# ---------------------------
def visualize_region_all(region_id, model):
    region_df = fiber_df[fiber_df["RegionID"]==region_id]
    if region_df.empty:
        return

    # --- Stress-strain prediction ---
    idx = list(dataset.region_ids).index(region_id)
    x, y_true = dataset[idx]; x = x.unsqueeze(0)
    with torch.no_grad():
        y_pred = model(x).squeeze().cpu().numpy()
    strain = x[0,:,0].cpu().numpy()
    stress_true = y_true.squeeze().cpu().numpy()
    stress_pred = y_pred

    # --- Fiber graph with adjacency ---
    data, adj_matrix, starts, ends, fiber_vfs = build_region_graph_with_adj(region_df)
    n_fibers = data.x.shape[0]
    centers = (starts[:n_fibers] + ends[:n_fibers]) / 2.0

    # --- Create figure with 3 subplots ---
    fig = plt.figure(figsize=(18,5))

    # (1) Fiber-fiber interaction 3D
    ax1 = fig.add_subplot(131, projection='3d'); ax1.grid(False)
    colors = plt.cm.tab10.colors
    legend_handles = []
    for i in range(n_fibers):
        col = colors[i % len(colors)]
        ax1.plot([starts[i,0], ends[i,0]],
                 [starts[i,1], ends[i,1]],
                 [starts[i,2], ends[i,2]], '-o', color=col)
        # add fiber legend
        legend_handles.append(
            plt.Line2D([0],[0], color=col, lw=2,
                       label=f"Fiber {i} (Vf={fiber_vfs[i]:.3f})")
        )
    # draw edges with thickness proportional to weight
    for i, j in combinations(range(n_fibers), 2):
        w = adj_matrix[i,j]
        if w > 0:
            ax1.plot([centers[i,0], centers[j,0]],
                     [centers[i,1], centers[j,1]],
                     [centers[i,2], centers[j,2]],
                     'k-', alpha=0.6, linewidth=1+3*w)
    ax1.set_title(f"Region {region_id} (Vf={region_df['Vf'].iloc[0]:.3f})")
    ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")
    if legend_handles:
        ax1.legend(handles=legend_handles, loc="best", fontsize=7)

    # (2) Adjacency heatmap
    ax2 = fig.add_subplot(132); ax2.grid(False)
    cax = ax2.matshow(adj_matrix, cmap='viridis')
    fig.colorbar(cax, ax=ax2)
    ax2.set_title("Adjacency Matrix")
    ax2.set_xlabel("Fiber Index"); ax2.set_ylabel("Fiber Index")
    for (i, j), val in np.ndenumerate(adj_matrix):
        if i != j:
            ax2.text(j, i, f"{val:.2f}", ha='center', va='center',
                     color='white' if val>0.5 else 'black', fontsize=7)

    # (3) Stress-strain curve
    ax3 = fig.add_subplot(133)
    ax3.plot(strain, stress_true, 'k-', label="True Stress")
    ax3.plot(strain, stress_pred, 'r--', label="Predicted Stress")
    ax3.set_xlabel("Strain"); ax3.set_ylabel("Stress (MPa)")
    ax3.set_title("Stress–Strain")
    ax3.legend()

    plt.suptitle(f"Region {region_id} Visualization", fontsize=14)
    plt.tight_layout()
    plt.show()


# ---------------------------
# Run for all RegionIDs
# ---------------------------
save_path = os.path.join("checkpoints", "best_model.pth")
best_model = StressPredictor(input_dim)
best_model.load_state_dict(torch.load(save_path))
best_model.eval()

# COMMENTED OUT: This shows all 27 region visualizations automatically
# Uncomment the lines below if you want to see all region visualizations
# for rid in dataset.region_ids:
#     visualize_region_all(rid, best_model)

# ---------------------------
# Global Evaluation
# ---------------------------
all_true, all_pred, all_metrics = [], [], []
for idx in range(len(dataset)):
    x,y_true = dataset[idx]; x=x.unsqueeze(0)
    with torch.no_grad(): y_pred = best_model(x).squeeze().cpu().numpy()
    strain = x[0,:,0].cpu().numpy()
    stress_true=y_true.squeeze().cpu().numpy(); stress_pred=y_pred
    all_true.extend(stress_true.tolist()); all_pred.extend(stress_pred.tolist())
    r2=r2_score(stress_true,stress_pred)
    all_metrics.append([dataset.region_ids[idx], r2])

plt.figure(figsize=(6,6))
plt.scatter(all_true,all_pred,alpha=0.6,edgecolor='k')
max_val=max(max(all_true),max(all_pred))
plt.plot([0,max_val],[0,max_val],'r--',label="Perfect Agreement")
plt.xlabel("True Stress (MPa)"); plt.ylabel("Predicted Stress (MPa)")
plt.title("Global Prediction Discrepancy"); plt.legend(); plt.tight_layout(); plt.show()

df_all=pd.DataFrame(all_metrics,columns=["RegionID","R2"])
print("\n===== R² Across ALL Regions =====")
print(df_all.round(3).to_string(index=False))
print(f"\nGlobal R² across all points: {r2_score(all_true,all_pred):.3f}")

# ---------------------------
# Prediction Function for New Microstructures
# ---------------------------
def predict_stress_strain(fiber_starts, fiber_ends, Vf, strain_range=None, model_path=None, use_training_approach=False, use_similar_region=False, disable_calibration=False):
    """
    Predict stress-strain curve for a new microstructure
    
    Parameters:
    -----------
    fiber_starts : list or np.array
        List of fiber start points [[x1,y1,z1], [x2,y2,z2], ...]
    fiber_ends : list or np.array  
        List of fiber end points [[x1,y1,z1], [x2,y2,z2], ...]
    Vf : float
        Volume fraction of the region
    strain_range : list or np.array, optional
        Strain values to predict. If None, uses default range [0, 0.1] with 50 points
    model_path : str, optional
        Path to saved model. If None, uses the trained model from checkpoints
    use_training_approach : bool, optional
        If True, uses the exact same approach as training (for existing regions)
    use_similar_region : bool, optional
        If True, finds a similar region and uses its physics descriptors
        
    Returns:
    --------
    dict : Dictionary containing strain, predicted_stress, and microstructure info
    """
    
    # Set default strain range if not provided
    if strain_range is None:
        strain_range = np.linspace(0, 2.0, 16)  # Same as training data: 0-2.0 with 16 points
    
    # Check if we should use the training approach (for existing regions)
    if use_training_approach:
        # Find the region ID that matches this microstructure
        region_id = None
        for rid in dataset.region_ids:
            region_df = fiber_df[fiber_df["RegionID"] == rid]
            region_starts = parse_point_str(region_df["FiberStartPoint"].iloc[0])
            region_ends = parse_point_str(region_df["FiberEndPoint"].iloc[0])
            region_Vf = region_df["Vf"].iloc[0]
            
            # Check if this matches our input (handle different array shapes)
            try:
                starts_match = np.allclose(region_starts, fiber_starts, rtol=1e-6)
                ends_match = np.allclose(region_ends, fiber_ends, rtol=1e-6)
                vf_match = abs(region_Vf - Vf) < 1e-6
                
                if starts_match and ends_match and vf_match:
                    region_id = rid
                    break
            except Exception as e:
                # Skip this region if comparison fails
                continue
        
        if region_id is not None:
            # Use the exact training data approach
            idx = list(dataset.region_ids).index(region_id)
            x_train, y_true = dataset[idx]
            
            # Use the same strain range as training
            strain_range = x_train[:, 0].cpu().numpy()
            
            # Load model and predict
            if model_path is None:
                model_path = os.path.join("checkpoints", "best_model.pth")
            pred_model = StressPredictor(input_dim)
            pred_model.load_state_dict(torch.load(model_path))
            pred_model.eval()
            
            with torch.no_grad():
                x_input = x_train.unsqueeze(0)
                stress_pred = pred_model(x_input).squeeze().cpu().numpy()
            
            # Calculate properties
            E_pred = compute_modulus(strain_range, stress_pred)
            sigma_y_pred = compute_yield_point(strain_range, stress_pred)
            auc_pred = compute_auc(strain_range, stress_pred)
            
            return {
                'strain': strain_range,
                'predicted_stress': stress_pred,
                'modulus': E_pred,
                'yield_stress': sigma_y_pred,
                'area_under_curve': auc_pred,
                'microstructure_info': {
                    'n_fibers': len(fiber_starts),
                    'volume_fraction': Vf,
                    'fiber_lengths': [fiber_length(fiber_starts[i], fiber_ends[i]) for i in range(len(fiber_starts))],
                    'fiber_volume_fractions': [Vf * fiber_length(fiber_starts[i], fiber_ends[i]) / sum([fiber_length(fiber_starts[j], fiber_ends[j]) for j in range(len(fiber_starts))]) for i in range(len(fiber_starts))]
                }
            }
    
    # Use unified graph building function for consistency
    data, starts, ends, fiber_vfs = build_graph_from_coordinates(fiber_starts, fiber_ends, Vf)
    n_fibers = len(starts)
    lengths = [fiber_length(starts[i], ends[i]) for i in range(n_fibers)]
    
    # Generate region embedding using the trained GNN
    with torch.no_grad():
        region_embedding = gnn_model(data.x, data.edge_index, data.edge_attr.view(-1))
    
    # Prepare input features for LSTM
    strain_t = torch.tensor(strain_range, dtype=torch.float).unsqueeze(-1)
    
    # Physics descriptors - choose approach based on parameters
    if use_similar_region:
        # Find the most similar region based on Vf and fiber count
        best_region_id = None
        best_similarity = -1
        
        for rid in dataset.region_ids:
            region_df = fiber_df[fiber_df["RegionID"] == rid]
            region_Vf = region_df["Vf"].iloc[0]
            region_starts = parse_point_str(region_df["FiberStartPoint"].iloc[0])
            region_n_fibers = len(region_starts)
            
            # Calculate similarity score (Vf similarity + fiber count similarity)
            vf_similarity = 1.0 - abs(region_Vf - Vf) / max(region_Vf, Vf)
            fiber_similarity = 1.0 - abs(region_n_fibers - n_fibers) / max(region_n_fibers, n_fibers)
            total_similarity = 0.7 * vf_similarity + 0.3 * fiber_similarity
            
            if total_similarity > best_similarity:
                best_similarity = total_similarity
                best_region_id = rid
        
        if best_region_id is not None:
            # Use physics descriptors from the most similar region
            similar_idx = list(dataset.region_ids).index(best_region_id)
            similar_x, similar_y = dataset[similar_idx]
            similar_strain = similar_x[:, 0].cpu().numpy()
            similar_stress = similar_y.squeeze().cpu().numpy()
            
            # Compute physics descriptors from similar region
            E_val = compute_modulus(similar_strain, similar_stress)
            sigma_y_val = compute_yield_point(similar_strain, similar_stress)
            auc_val = compute_auc(similar_strain, similar_stress)
            
            print(f"Using physics descriptors from similar Region {best_region_id} (similarity: {best_similarity:.3f})")
            print(f"  E={E_val:.1f} MPa, σy={sigma_y_val:.1f} MPa, AUC={auc_val:.1f}")
        else:
            # Fallback to estimation
            E_val = 500.0 + Vf * 3000.0
            sigma_y_val = 10.0 + Vf * 150.0
            auc_val = 2.0 + Vf * 30.0
            print("No similar region found, using estimated physics descriptors")
    else:
        # Use learned regressor from (Vf + embedding) -> [E, sigma_y, AUC]
        reg = load_physics_regressor()
        with torch.no_grad():
            emb_np = region_embedding.cpu().numpy().reshape(-1)
        X_reg = np.concatenate([[Vf], emb_np]).reshape(1, -1)
        try:
            E_val, sigma_y_val, auc_val = list(reg.predict(X_reg)[0])
            print(f"Physics regressor: E={E_val:.2f}, σy={sigma_y_val:.2f}, AUC={auc_val:.2f}")
        except Exception:
            # Fallback if regressor not available
            E_val = 500.0 + Vf * 3000.0
            sigma_y_val = 10.0 + Vf * 150.0
            auc_val = 2.0 + Vf * 30.0
            print("Regressor unavailable; using estimated physics descriptors")
    
    physics_feats = torch.tensor([E_val, sigma_y_val, auc_val], dtype=torch.float)
    physics_feats = physics_feats.repeat(len(strain_range), 1)
    
    # Strain-derived features
    strain_sq = strain_t**2
    strain_cu = strain_t**3
    strain_log = torch.log1p(strain_t)
    Vf_val = torch.tensor([Vf], dtype=torch.float).repeat(len(strain_range), 1)
    emb = region_embedding.repeat(len(strain_range), 1)
    
    # Combine all features
    x_input = torch.cat([strain_t, strain_sq, strain_cu, strain_log, Vf_val, emb, physics_feats], dim=1)
    x_input = x_input.unsqueeze(0)  # Add batch dimension
    
    # Load model if path provided, otherwise use current best model
    if model_path is None:
        model_path = os.path.join("checkpoints", "best_model.pth")
    
    # Create new model instance and load weights
    pred_model = StressPredictor(input_dim)
    pred_model.load_state_dict(torch.load(model_path))
    pred_model.eval()
    
    # Make prediction
    with torch.no_grad():
        stress_pred = pred_model(x_input).squeeze().cpu().numpy()

    # Guard against NaNs/infs from the raw network output
    if np.any(~np.isfinite(stress_pred)):
        valid_idx = np.where(np.isfinite(stress_pred))[0]
        if valid_idx.size >= 2:
            stress_pred = np.interp(np.arange(len(stress_pred)), valid_idx, stress_pred[valid_idx])
        elif valid_idx.size == 1:
            stress_pred = np.full_like(stress_pred, stress_pred[valid_idx[0]])
        else:
            stress_pred = np.zeros_like(stress_pred)

    # Sequence calibration to match training curves shape (skip when explicitly disabled)
    if not disable_calibration:
        try:
            calibrators = load_sequence_calibrators()
            if len(calibrators) == len(stress_pred):
                stress_pred = np.array([calibrators[i].predict([stress_pred[i]])[0] for i in range(len(stress_pred))])
        except Exception:
            # Fallback: global isotonic smoothing to keep positive slope
            try:
                iso = IsotonicRegression(y_min=0, increasing=True)
                stress_pred = np.asarray(iso.fit_transform(strain_range, stress_pred))
            except Exception:
                pass

    # Final shaping: enforce a realistic saturating curve (positive, non-increasing slope)
    try:
        stress_pred = enforce_monotone_convex(stress_pred, np.asarray(strain_range))
    except Exception:
        pass

    # Add gentle saturation curvature by fitting A*(1-exp(-k*x)) and blending
    try:
        x = np.asarray(strain_range).astype(float)
        y = np.asarray(stress_pred).astype(float)
        A0 = max(y.max()*1.05, 1e-6)
        z = np.log(np.maximum(A0 - y, 1e-6))
        # Linear fit: z ≈ c - k*x
        X = np.vstack([np.ones_like(x), -x]).T
        coeff, *_ = np.linalg.lstsq(X, z, rcond=None)
        c, k = coeff[0], coeff[1]
        A_est = float(np.exp(c))
        k = float(max(k, 1e-6))
        y_fit = A_est * (1.0 - np.exp(-k * x))
        # Blend 50-50 to keep training-like shape but add curvature
        stress_pred = 0.5*y + 0.5*y_fit
    except Exception:
        pass
    
    # Calculate additional properties
    E_pred = compute_modulus(strain_range, stress_pred)
    sigma_y_pred = compute_yield_point(strain_range, stress_pred)
    auc_pred = compute_auc(strain_range, stress_pred)
    
    return {
        'strain': strain_range,
        'predicted_stress': stress_pred,
        'modulus': E_pred,
        'yield_stress': sigma_y_pred,
        'area_under_curve': auc_pred,
        'microstructure_info': {
            'n_fibers': n_fibers,
            'volume_fraction': Vf,
            'fiber_lengths': lengths,
            'fiber_volume_fractions': fiber_vfs
        }
    }

def plot_prediction_result(prediction_result, title="Stress-Strain Prediction"):
    """Plot the prediction results"""
    strain = prediction_result['strain']
    stress = prediction_result['predicted_stress']
    info = prediction_result['microstructure_info']
    
    plt.figure(figsize=(10, 6))
    plt.plot(strain, stress, 'b-', linewidth=2, label='Predicted Stress')
    plt.xlabel('Strain')
    plt.ylabel('Stress (MPa)')
    plt.title(f"{title}\nVf={info['volume_fraction']:.3f}, N_fibers={info['n_fibers']}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add text box with properties
    textstr = f"E = {prediction_result['modulus']:.1f} MPa\n"
    textstr += f"σy = {prediction_result['yield_stress']:.1f} MPa\n"
    textstr += f"AUC = {prediction_result['area_under_curve']:.1f}"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.show()

# ---------------------------
# Example Usage and Test Cases
# ---------------------------
print("\n" + "="*60)
print("PREDICTION FUNCTION READY!")
print("="*60)

# ---------------------------
# Validation Function for Existing Regions
# ---------------------------
def validate_region(region_id, model):
    """Validate prediction against known region data"""
    if region_id not in dataset.region_ids:
        print(f"❌ RegionID {region_id} not found in dataset!")
        print(f"Available regions: {sorted(dataset.region_ids)}")
        return None
    
    # Get the actual data for this region
    idx = list(dataset.region_ids).index(region_id)
    x, y_true = dataset[idx]
    
    # Use the EXACT same approach as training (direct model prediction)
    # This ensures identical input features and identical predictions
    x_input = x.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        y_pred = model(x_input).squeeze().cpu().numpy()
    
    # Compare with true values
    strain_true = x[:, 0].cpu().numpy()
    stress_true = y_true.squeeze().cpu().numpy()
    stress_pred = y_pred
    
    # Calculate R²
    r2 = r2_score(stress_true, stress_pred)
    
    # Plot comparison
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Stress-strain comparison
    plt.subplot(1, 2, 1)
    plt.plot(strain_true, stress_true, 'k-', linewidth=2, label='True Stress')
    plt.plot(strain_true, stress_pred, 'r--', linewidth=2, label='Predicted Stress')
    plt.xlabel('Strain')
    plt.ylabel('Stress (MPa)')
    plt.title(f'Region {region_id} Validation\nR² = {r2:.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Scatter plot
    plt.subplot(1, 2, 2)
    plt.scatter(stress_true, stress_pred, alpha=0.6, s=50)
    max_val = max(max(stress_true), max(stress_pred))
    plt.plot([0, max_val], [0, max_val], 'r--', label='Perfect Agreement')
    plt.xlabel('True Stress (MPa)')
    plt.ylabel('Predicted Stress (MPa)')
    plt.title(f'Prediction Accuracy\nR² = {r2:.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"✓ Validation complete for Region {region_id}")
    print(f"  - R² Score: {r2:.3f}")
    print(f"  - True Vf: {Vf:.3f}")
    print(f"  - Number of fibers: {len(starts)}")
    
    return {
        'region_id': region_id,
        'r2_score': r2,
        'true_stress': stress_true,
        'predicted_stress': stress_pred,
        'strain': strain_true
    }

# ---------------------------
# Input Feature Consistency Test Function
# ---------------------------
def test_input_consistency(region_id):
    """Test that input features are identical between training and prediction approaches"""
    if region_id not in dataset.region_ids:
        print(f"❌ RegionID {region_id} not found!")
        return
    
    # Get training data
    idx = list(dataset.region_ids).index(region_id)
    x_train, y_true = dataset[idx]
    
    # Get fiber data
    region_df = fiber_df[fiber_df["RegionID"] == region_id]
    starts = parse_point_str(region_df["FiberStartPoint"].iloc[0])
    ends = parse_point_str(region_df["FiberEndPoint"].iloc[0])
    Vf = region_df["Vf"].iloc[0]
    
    # Get prediction input features
    strain_range = x_train[:, 0].cpu().numpy()
    result = predict_stress_strain(starts, ends, Vf, strain_range)
    
    # Compare input features
    print(f"Region {region_id} Input Feature Comparison:")
    print(f"Training input shape: {x_train.shape}")
    print(f"Prediction input shape: {result.get('input_features', 'Not available')}")
    
    # Check if strain ranges match
    strain_train = x_train[:, 0].cpu().numpy()
    strain_pred = result['strain']
    strain_match = np.allclose(strain_train, strain_pred, rtol=1e-6)
    print(f"Strain ranges match: {strain_match}")
    
    if not strain_match:
        print(f"Training strain: {strain_train[:5]}...")
        print(f"Prediction strain: {strain_pred[:5]}...")
    
    return strain_match

# ---------------------------
# GNN Consistency Test Function
# ---------------------------
def test_gnn_consistency(region_id):
    """Test that GNN gives same embedding for same microstructure"""
    if region_id not in dataset.region_ids:
        print(f"❌ RegionID {region_id} not found!")
        return
    
    # Get original data
    region_df = fiber_df[fiber_df["RegionID"] == region_id]
    starts = parse_point_str(region_df["FiberStartPoint"].iloc[0])
    ends = parse_point_str(region_df["FiberEndPoint"].iloc[0])
    Vf = region_df["Vf"].iloc[0]
    
    # Method 1: Original training approach
    g,_,_,_ = build_region_graph(region_df)
    with torch.no_grad():
        emb1 = gnn_model(g.x, g.edge_index, g.edge_attr.view(-1))
    
    # Method 2: New coordinate-based approach
    data,_,_,_ = build_graph_from_coordinates(starts, ends, Vf)
    with torch.no_grad():
        emb2 = gnn_model(data.x, data.edge_index, data.edge_attr.view(-1))
    
    # Compare embeddings
    diff = torch.norm(emb1 - emb2).item()
    print(f"Region {region_id} GNN Embedding Difference: {diff:.6f}")
    print(f"Embeddings are {'IDENTICAL' if diff < 1e-6 else 'DIFFERENT'}")
    
    return diff < 1e-6

# ---------------------------
# Interactive Menu System
# ---------------------------
def interactive_menu():
    """Interactive menu for validation and prediction"""
    print("\n" + "="*60)
    print("🔬 FIBER COMPOSITE STRESS PREDICTION SYSTEM")
    print("="*60)
    
    while True:
        print("\n📋 Choose an option:")
        print("1. 🔍 Validate existing region (compare prediction vs true data)")
        print("2. 🔮 Predict new microstructure")
        print("3. 📊 Show all available regions")
        print("4. 🧪 Test GNN consistency (same microstructure → same embedding)")
        print("5. 🔬 Test input feature consistency (training vs prediction)")
        print("6. 🎯 Test prediction with training approach (same microstructure)")
        print("7. 🔍 Test prediction with similar region approach (new microstructure)")
        print("8. ❌ Exit")
        
        choice = input("\nEnter your choice (1-8): ").strip()
        
        if choice == '1':
            # Validation mode
            print(f"\nAvailable regions: {sorted(dataset.region_ids)}")
            try:
                region_input = input("Enter RegionID to validate: ").strip()
                region_id = int(region_input)
                print(f"Debug: You entered RegionID {region_id}")
                print(f"Debug: RegionID {region_id} in dataset: {region_id in dataset.region_ids}")
                validate_region(region_id, best_model)
            except ValueError as ve:
                print(f"❌ Please enter a valid RegionID number. You entered: '{region_input}'")
                print(f"Debug: ValueError details: {ve}")
            except Exception as e:
                print(f"❌ Error: {e}")
                print(f"Debug: Exception type: {type(e).__name__}")
                import traceback
                print(f"Debug: Traceback: {traceback.format_exc()}")
        
        elif choice == '2':
            # Prediction mode
            print("\n🔮 PREDICTION MODE")
            print("Enter fiber microstructure data:")
            
            try:
                # Get number of fibers
                n_fibers = int(input("Number of fibers: "))
                
                # Get fiber coordinates
                starts, ends = [], []
                print(f"\nEnter coordinates for {n_fibers} fibers:")
                for i in range(n_fibers):
                    print(f"\nFiber {i+1}:")
                    start_coords = input(f"  Start point (x,y,z): ").strip()
                    end_coords = input(f"  End point (x,y,z): ").strip()
                    
                    # Parse coordinates
                    start = [float(x.strip()) for x in start_coords.split(',')]
                    end = [float(x.strip()) for x in end_coords.split(',')]
                    starts.append(start)
                    ends.append(end)
                
                # Get volume fraction
                Vf = float(input("\nVolume fraction (Vf): "))
                
                # Get strain range (optional)
                strain_input = input("Strain range (min,max,points) or press Enter for default (0,0.1,50): ").strip()
                if strain_input:
                    min_strain, max_strain, n_points = [float(x.strip()) for x in strain_input.split(',')]
                    strain_range = np.linspace(min_strain, max_strain, int(n_points))
                else:
                    strain_range = None
                
                # Make prediction
                print("\n🔄 Making prediction...")
                result = predict_stress_strain(starts, ends, Vf, strain_range)
                
                # Show results
                print(f"\n✅ Prediction complete!")
                print(f"  - Strain range: {result['strain'][0]:.3f} to {result['strain'][-1]:.3f}")
                print(f"  - Stress range: {result['predicted_stress'][0]:.1f} to {result['predicted_stress'][-1]:.1f} MPa")
                print(f"  - Modulus: {result['modulus']:.1f} MPa")
                print(f"  - Yield stress: {result['yield_stress']:.1f} MPa")
                print(f"  - Area under curve: {result['area_under_curve']:.1f}")
                
                # Plot result
                plot_prediction_result(result, f"Custom Microstructure Prediction")
                
            except ValueError as e:
                print(f"❌ Invalid input: {e}")
            except Exception as e:
                print(f"❌ Error: {e}")
        
        elif choice == '3':
            # Show available regions
            print(f"\n📊 Available regions: {sorted(dataset.region_ids)}")
            print(f"Total regions: {len(dataset.region_ids)}")
            
        elif choice == '4':
            # GNN consistency test
            print(f"\nAvailable regions: {sorted(dataset.region_ids)}")
            try:
                region_id = int(input("Enter RegionID to test GNN consistency: "))
                test_gnn_consistency(region_id)
            except ValueError:
                print("❌ Please enter a valid RegionID number")
            except Exception as e:
                print(f"❌ Error: {e}")
        
        elif choice == '5':
            # Input feature consistency test
            print(f"\nAvailable regions: {sorted(dataset.region_ids)}")
            try:
                region_id = int(input("Enter RegionID to test input feature consistency: "))
                test_input_consistency(region_id)
            except ValueError:
                print("❌ Please enter a valid RegionID number")
            except Exception as e:
                print(f"❌ Error: {e}")
        
        elif choice == '6':
            # Test prediction with training approach
            print(f"\nAvailable regions: {sorted(dataset.region_ids)}")
            try:
                region_id = int(input("Enter RegionID to test prediction with training approach: "))
                
                # Get fiber data for this region
                region_df = fiber_df[fiber_df["RegionID"] == region_id]
                starts = parse_point_str(region_df["FiberStartPoint"].iloc[0])
                ends = parse_point_str(region_df["FiberEndPoint"].iloc[0])
                Vf = region_df["Vf"].iloc[0]
                
                print(f"\nTesting Region {region_id} with coordinates:")
                print(f"Starts: {starts}")
                print(f"Ends: {ends}")
                print(f"Vf: {Vf}")
                
                # Get the training strain range first
                idx = list(dataset.region_ids).index(region_id)
                x_train, y_true = dataset[idx]
                training_strain = x_train[:, 0].cpu().numpy()
                
                # Test both approaches with SAME strain range
                print("\n1. Standard prediction approach:")
                try:
                    result1 = predict_stress_strain(
                        starts, ends, Vf,
                        strain_range=training_strain,
                        use_training_approach=False,
                        disable_calibration=True,
                    )
                    print(f"   Stress range: {result1['predicted_stress'][0]:.1f} to {result1['predicted_stress'][-1]:.1f} MPa")
                    print(f"   Strain range: {result1['strain'][0]:.3f} to {result1['strain'][-1]:.3f}")
                    print(f"   Number of points: {len(result1['predicted_stress'])}")
                except Exception as e:
                    print(f"   Error in standard approach: {e}")
                    result1 = None
                
                print("\n2. Training approach:")
                try:
                    result2 = predict_stress_strain(starts, ends, Vf, strain_range=training_strain, use_training_approach=True)
                    print(f"   Stress range: {result2['predicted_stress'][0]:.1f} to {result2['predicted_stress'][-1]:.1f} MPa")
                    print(f"   Strain range: {result2['strain'][0]:.3f} to {result2['strain'][-1]:.3f}")
                    print(f"   Number of points: {len(result2['predicted_stress'])}")
                except Exception as e:
                    print(f"   Error in training approach: {e}")
                    result2 = None
                
                # Compare results if both succeeded
                if result1 is not None and result2 is not None:
                    # Check if arrays have same length
                    if len(result1['predicted_stress']) == len(result2['predicted_stress']):
                        diff = np.mean(np.abs(result1['predicted_stress'] - result2['predicted_stress']))
                        print(f"\nMean absolute difference: {diff:.3f} MPa")
                        print(f"Results are {'IDENTICAL' if diff < 1e-6 else 'DIFFERENT'}")
                        
                        # Plot comparison
                        plt.figure(figsize=(12, 5))
                        
                        plt.subplot(1, 2, 1)
                        plt.plot(result1['strain'], result1['predicted_stress'], 'b-', label='Standard Prediction')
                        plt.plot(result2['strain'], result2['predicted_stress'], 'r--', label='Training Approach')
                        plt.xlabel('Strain')
                        plt.ylabel('Stress (MPa)')
                        plt.title(f'Region {region_id} Prediction Comparison')
                        plt.legend()
                        plt.grid(True, alpha=0.3)
                        
                        plt.subplot(1, 2, 2)
                        plt.scatter(result1['predicted_stress'], result2['predicted_stress'], alpha=0.6)
                        max_val = max(max(result1['predicted_stress']), max(result2['predicted_stress']))
                        plt.plot([0, max_val], [0, max_val], 'r--', label='Perfect Agreement')
                        plt.xlabel('Standard Prediction (MPa)')
                        plt.ylabel('Training Approach (MPa)')
                        plt.title(f'Prediction Agreement\nMean diff: {diff:.3f} MPa')
                        plt.legend()
                        plt.grid(True, alpha=0.3)
                        
                        plt.tight_layout()
                        plt.show()
                    else:
                        print(f"\nCannot compare: Different array lengths")
                        print(f"Standard: {len(result1['predicted_stress'])} points")
                        print(f"Training: {len(result2['predicted_stress'])} points")
                else:
                    print("\nCannot compare: One or both approaches failed")
                
            except ValueError:
                print("❌ Please enter a valid RegionID number")
            except Exception as e:
                print(f"❌ Error: {e}")
        
        elif choice == '7':
            # Universal prediction (training-like by default, auto-detect existing region)
            print("\n🔍 TESTING SIMILAR REGION APPROACH")
            print("This will test prediction for microstructures. If the input matches an existing region,"
                  " it will use the exact training path. Otherwise it uses the learned regressor (training-like).")
            
            try:
                # Get number of fibers
                n_fibers = int(input("Number of fibers: "))
                
                # Get fiber coordinates
                starts, ends = [], []
                print(f"\nEnter coordinates for {n_fibers} fibers:")
                for i in range(n_fibers):
                    print(f"\nFiber {i+1}:")
                    start_coords = input(f"  Start point (x,y,z): ").strip()
                    end_coords = input(f"  End point (x,y,z): ").strip()
                    
                    # Parse coordinates
                    start = [float(x.strip()) for x in start_coords.split(',')]
                    end = [float(x.strip()) for x in end_coords.split(',')]
                    starts.append(start)
                    ends.append(end)
                
                # Get volume fraction
                Vf = float(input("\nVolume fraction (Vf): "))
                
                print(f"\nTesting with coordinates:")
                print(f"Starts: {starts}")
                print(f"Ends: {ends}")
                print(f"Vf: {Vf}")
                
                # If this microstructure matches an existing region, use training path
                matched_region = None
                for rid in dataset.region_ids:
                    rdf = fiber_df[fiber_df['RegionID']==rid]
                    rs = parse_point_str(rdf['FiberStartPoint'].iloc[0])
                    re = parse_point_str(rdf['FiberEndPoint'].iloc[0])
                    rv = float(rdf['Vf'].iloc[0])
                    try:
                        same_starts = np.allclose(rs, np.array(starts), rtol=1e-6)
                        same_ends = np.allclose(re, np.array(ends), rtol=1e-6)
                        same_vf = abs(rv - Vf) < 1e-6
                        if same_starts and same_ends and same_vf:
                            matched_region = rid
                            break
                    except Exception:
                        continue

                if matched_region is not None:
                    print(f"\nExisting region detected (Region {matched_region}). Using exact training approach.")
                    idx = list(dataset.region_ids).index(matched_region)
                    x_train, y_true = dataset[idx]
                    training_strain = x_train[:,0].cpu().numpy()
                    # Convert lists to numpy arrays to avoid list arithmetic issues downstream
                    starts_np = np.array(starts, dtype=float)
                    ends_np = np.array(ends, dtype=float)
                    result = predict_stress_strain(starts_np, ends_np, Vf, strain_range=training_strain, use_training_approach=True)
                    plot_prediction_result(result, f"Region {matched_region} (Training Path)")
                else:
                    print("\nNew microstructure detected. Using training-like regressor + calibration.")
                    result = predict_stress_strain(starts, ends, Vf)
                    plot_prediction_result(result, "New Microstructure Prediction (Training-like)")
                
            except ValueError as e:
                print(f"❌ Invalid input: {e}")
            except Exception as e:
                print(f"❌ Error: {e}")
        
        elif choice == '8':
            print("\n👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice. Please enter 1, 2, 3, 4, 5, 6, 7, or 8.")

# ---------------------------
# Run Interactive Menu
# ---------------------------
print("\n" + "="*60)
print("🚀 SYSTEM READY!")
print("="*60)
print("Model trained and ready for validation and prediction.")
print(f"Debug: Dataset has {len(dataset.region_ids)} regions")
print(f"Debug: RegionIDs are: {sorted(dataset.region_ids)}")
print(f"Debug: RegionID 1 in dataset: {1 in dataset.region_ids}")
print("Starting interactive menu...")

# Run the interactive menu
interactive_menu()

# ---------------------------
# Simple GUI (Tkinter)
# ---------------------------
try:
    import tkinter as tk
    from tkinter import ttk, messagebox
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
except Exception:
    tk = None

def launch_gui():
    if tk is None:
        print("Tkinter not available.")
        return

    root = tk.Tk()
    root.title("GNN-LSTM Composite Stress Predictor")
    root.geometry("1200x800")

    nb = ttk.Notebook(root)
    nb.pack(fill=tk.BOTH, expand=True)

    # Tab 1: Training/Validation + Global discrepancy
    tab1 = ttk.Frame(nb); nb.add(tab1, text="Training Summary")
    frame1 = ttk.Frame(tab1); frame1.pack(fill=tk.BOTH, expand=True)
    fig1, ax1 = plt.subplots(1,2, figsize=(10,4))
    # Loss curves
    try:
        t_losses = np.load(os.path.join("checkpoints","train_losses.npy"))
        v_losses = np.load(os.path.join("checkpoints","val_losses.npy"))
    except Exception:
        t_losses, v_losses = np.array(train_losses), np.array(val_losses)
    ax1[0].plot(t_losses, label="Train Loss"); ax1[0].plot(v_losses, label="Val Loss")
    ax1[0].set_title("Training/Validation Loss"); ax1[0].set_xlabel("Epoch"); ax1[0].set_ylabel("Loss"); ax1[0].legend()
    # Global discrepancy scatter
    try:
        all_true, all_pred = [], []
        for idx in range(len(dataset)):
            x,y_true = dataset[idx]; x=x.unsqueeze(0)
            with torch.no_grad(): y_pred = best_model(x).squeeze().cpu().numpy()
            all_true.extend(y_true.squeeze().cpu().numpy().tolist())
            all_pred.extend(y_pred.tolist())
        ax1[1].scatter(all_true, all_pred, alpha=0.5, edgecolor='k')
        max_val = max(max(all_true), max(all_pred))
        ax1[1].plot([0,max_val],[0,max_val],'r--')
        ax1[1].set_title("Predicted vs Actual Stress"); ax1[1].set_xlabel("Actual"); ax1[1].set_ylabel("Predicted")
    except Exception as e:
        ax1[1].text(0.1,0.5, f"Error building scatter: {e}", transform=ax1[1].transAxes)
    canvas1 = FigureCanvasTkAgg(fig1, master=frame1); canvas1.draw(); canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # Tab 2: Region visualization & validation
    tab2 = ttk.Frame(nb); nb.add(tab2, text="Region Validation")
    top2 = ttk.Frame(tab2); top2.pack(side=tk.TOP, fill=tk.X)
    ttk.Label(top2, text="RegionID:").pack(side=tk.LEFT)
    rid_var = tk.StringVar(value=str(sorted(dataset.region_ids)[0]))
    ttk.Entry(top2, textvariable=rid_var, width=8).pack(side=tk.LEFT, padx=5)
    fig2 = plt.Figure(figsize=(10,4)); canvas2 = FigureCanvasTkAgg(fig2, master=tab2)
    canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def draw_region():
        fig2.clf()
        try:
            rid = int(rid_var.get())
            region_df = fiber_df[fiber_df["RegionID"]==rid]
            if region_df.empty:
                messagebox.showerror("Error", f"Region {rid} not found"); return
            idx = list(dataset.region_ids).index(rid)
            x, y_true = dataset[idx]; xnp = x.detach().cpu().numpy(); x1 = x.unsqueeze(0)
            with torch.no_grad(): y_pred = best_model(x1).squeeze().cpu().numpy()
            strain = xnp[:,0]
            # build adjacency
            data, adj_matrix, starts, ends, fiber_vfs = build_region_graph_with_adj(region_df)
            n_f = data.x.shape[0]
            centers = (starts[:n_f] + ends[:n_f]) / 2.0
            # layout
            axs = [fig2.add_subplot(131, projection='3d'), fig2.add_subplot(132), fig2.add_subplot(133)]
            # 3D
            colors = plt.cm.tab10.colors
            for i in range(n_f):
                c = colors[i % len(colors)]
                axs[0].plot([starts[i,0], ends[i,0]],[starts[i,1], ends[i,1]],[starts[i,2], ends[i,2]],'-o', color=c)
            for i,j in combinations(range(n_f),2):
                w = adj_matrix[i,j]
                if w>0:
                    axs[0].plot([centers[i,0],centers[j,0]],[centers[i,1],centers[j,1]],[centers[i,2],centers[j,2]], 'k-', alpha=0.6, linewidth=1+3*w)
            axs[0].set_title(f"Region {rid} (Vf={region_df['Vf'].iloc[0]:.3f})")
            # Heatmap
            cax = axs[1].matshow(adj_matrix, cmap='viridis'); fig2.colorbar(cax, ax=axs[1])
            axs[1].set_title("Adjacency Matrix"); axs[1].set_xlabel("Fiber"); axs[1].set_ylabel("Fiber")
            for (i,j), val in np.ndenumerate(adj_matrix):
                if i!=j: axs[1].text(j,i,f"{val:.2f}",ha='center',va='center', color='white' if val>0.5 else 'black', fontsize=7)
            # Stress curves
            axs[2].plot(strain, y_true.squeeze().cpu().numpy(),'k-', label='True')
            axs[2].plot(strain, y_pred,'r--', label='Predicted')
            from sklearn.metrics import r2_score as _r2
            r2v = _r2(y_true.squeeze().cpu().numpy(), y_pred)
            axs[2].set_title(f"Stress–Strain (R²={r2v:.3f})")
            axs[2].set_xlabel("Strain"); axs[2].set_ylabel("Stress (MPa)"); axs[2].legend()
            canvas2.draw()
        except Exception as e:
            messagebox.showerror("Error", str(e))
    ttk.Button(top2, text="Show", command=draw_region).pack(side=tk.LEFT, padx=5)

    # Tab 3: Custom prediction
    tab3 = ttk.Frame(nb); nb.add(tab3, text="Custom Prediction")
    top3 = ttk.Frame(tab3); top3.pack(side=tk.TOP, fill=tk.X)
    ttk.Label(top3, text="Num Fibers:").pack(side=tk.LEFT)
    nvar = tk.StringVar(value="2"); ttk.Entry(top3, textvariable=nvar, width=6).pack(side=tk.LEFT)
    ttk.Label(top3, text="Vf:").pack(side=tk.LEFT)
    vfvar = tk.StringVar(value="0.2"); ttk.Entry(top3, textvariable=vfvar, width=8).pack(side=tk.LEFT)
    # Dynamic fiber entry grid
    form3 = ttk.Frame(tab3); form3.pack(fill=tk.X, pady=6)
    header = ttk.Frame(form3); header.pack(fill=tk.X)
    ttk.Label(header, text="#", width=3).grid(row=0, column=0)
    ttk.Label(header, text="Start (x,y,z)", width=20).grid(row=0, column=1, padx=2)
    ttk.Label(header, text="End (x,y,z)", width=20).grid(row=0, column=2, padx=2)
    rows_frame = ttk.Frame(form3); rows_frame.pack(fill=tk.X)
    fiber_vars = []  # list of (start_str, end_str) StringVars

    def build_fiber_form():
        # clear previous
        for w in rows_frame.winfo_children():
            w.destroy()
        fiber_vars.clear()
        try:
            n = max(1, int(nvar.get()))
        except Exception:
            n = 1
        for i in range(n):
            ttk.Label(rows_frame, text=str(i+1), width=3).grid(row=i, column=0)
            v_start = tk.StringVar(); v_end = tk.StringVar()
            e1 = ttk.Entry(rows_frame, textvariable=v_start, width=28)
            e2 = ttk.Entry(rows_frame, textvariable=v_end, width=28)
            e1.grid(row=i, column=1, padx=2, pady=1); e2.grid(row=i, column=2, padx=2, pady=1)
            fiber_vars.append((v_start, v_end))

    build_fiber_form()
    ttk.Button(top3, text="Build", command=build_fiber_form).pack(side=tk.LEFT, padx=5)
    fig3 = plt.Figure(figsize=(10,4)); canvas3 = FigureCanvasTkAgg(fig3, master=tab3)
    canvas3.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def run_custom():
        fig3.clf()
        try:
            n = int(nvar.get()); Vf = float(vfvar.get())
            starts, ends = [], []
            if len(fiber_vars) < n:
                raise ValueError("Click Build to create fiber input rows.")
            for i in range(n):
                s_str = fiber_vars[i][0].get().strip(); e_str = fiber_vars[i][1].get().strip()
                if not s_str or not e_str:
                    raise ValueError(f"Fill both start and end for fiber {i+1}.")
                try:
                    sx,sy,sz = [float(x) for x in s_str.split(',')]
                    ex,ey,ez = [float(x) for x in e_str.split(',')]
                except Exception:
                    raise ValueError(f"Use 'x,y,z' format for fiber {i+1}.")
                starts.append([sx,sy,sz]); ends.append([ex,ey,ez])
            # Auto-detect if matches existing region
            s_arr, e_arr = np.array(starts, dtype=float), np.array(ends, dtype=float)
            matched_region = None
            for rid in dataset.region_ids:
                rdf = fiber_df[fiber_df['RegionID']==rid]
                rs = parse_point_str(rdf['FiberStartPoint'].iloc[0])
                re = parse_point_str(rdf['FiberEndPoint'].iloc[0])
                rv = float(rdf['Vf'].iloc[0])
                try:
                    if np.allclose(rs, s_arr, rtol=1e-6) and np.allclose(re, e_arr, rtol=1e-6) and abs(rv - Vf) < 1e-6:
                        matched_region = rid
                        break
                except Exception:
                    continue

            # Build adjacency/3D from provided microstructure
            fake_df = {'RegionID':[0],'Vf':[Vf],'FiberStartPoint':[str(starts)],'FiberEndPoint':[str(ends)]}
            rdf = pd.DataFrame(fake_df)
            data, adj_matrix, s_np, e_np, fiber_vfs = build_region_graph_with_adj(rdf)
            n_f = data.x.shape[0]; centers = (s_np[:n_f]+e_np[:n_f])/2
            axs = [fig3.add_subplot(131, projection='3d'), fig3.add_subplot(132), fig3.add_subplot(133)]
            colors = plt.cm.tab10.colors
            for i in range(n_f):
                c = colors[i % len(colors)]
                axs[0].plot([s_np[i,0], e_np[i,0]],[s_np[i,1], e_np[i,1]],[s_np[i,2], e_np[i,2]],'-o', color=c)
            for i,j in combinations(range(n_f),2):
                w = adj_matrix[i,j]
                if w>0:
                    axs[0].plot([centers[i,0],centers[j,0]],[centers[i,1],centers[j,1]],[centers[i,2],centers[j,2]],'k-',alpha=0.6,linewidth=1+3*w)
            cax = axs[1].matshow(adj_matrix, cmap='viridis'); fig3.colorbar(cax, ax=axs[1])
            for (i,j), val in np.ndenumerate(adj_matrix):
                if i!=j: axs[1].text(j,i,f"{val:.2f}",ha='center',va='center', color='white' if val>0.5 else 'black', fontsize=7)
            axs[1].set_title("Adjacency Matrix")

            if matched_region is not None:
                # Exact training path with True vs Predicted
                idx = list(dataset.region_ids).index(matched_region)
                x_train, y_true = dataset[idx]
                with torch.no_grad():
                    y_pred = best_model(x_train.unsqueeze(0)).squeeze().cpu().numpy()
                strain = x_train[:,0].cpu().numpy()
                from sklearn.metrics import r2_score as _r2
                r2v = _r2(y_true.squeeze().cpu().numpy(), y_pred)
                axs[2].plot(strain, y_true.squeeze().cpu().numpy(),'k-', label='True')
                axs[2].plot(strain, y_pred,'r--', label='Predicted')
                axs[2].set_title(f"Stress–Strain (R²={r2v:.3f})")
                axs[2].legend()
            else:
                # Training-like regressor path
                res = predict_stress_strain(starts, ends, Vf)
                axs[2].plot(res['strain'], res['predicted_stress'],'b-')
                axs[2].set_title("Predicted Stress–Strain")
            axs[2].set_xlabel("Strain"); axs[2].set_ylabel("Stress (MPa)")
            canvas3.draw()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    ttk.Button(top3, text="Predict", command=run_custom).pack(side=tk.LEFT, padx=5)

    root.mainloop()

if __name__ == "__main__":
    try:
        launch_gui()
    except Exception:
        pass