"""
==============================================================
  IGTabularExplainer  —  Visual Demo
==============================================================

  This demo trains a real heart disease risk MLP from scratch,
  then uses IGTabularExplainer to explain a specific patient's
  prediction. All 5 visualisations + the full dashboard are
  saved to user_saves/.

  Run from your project root:
      python demo_ig_tabular.py

  Output folder:
      user_saves/
          tabular_dashboard.png       ← full 5-panel dashboard
          tabular_bar.png             ← signed attribution bar chart
          tabular_force.png           ← force plot (baseline → prediction story)
          tabular_waterfall.png       ← cumulative attribution build-up
          tabular_distribution.png    ← patient vs training population
          tabular_summary.png         ← ranked feature importance table

  The model:
      Binary classifier — predicts heart disease risk (0=low, 1=high)
      Features: age, cholesterol, blood_pressure, max_heart_rate,
                blood_sugar, chest_pain_type, exercise_angina
      Architecture: Linear(7→64) → ReLU → Linear(64→32) → ReLU → Linear(32→2)
      Training: 300 samples, 150 epochs, Adam optimizer
==============================================================
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from EXACT.explainers.ig_tabular_explainer import IGTabularExplainer

# ── Output folder ─────────────────────────────────────────────────────────────
SAVE_DIR = "user_saves"
os.makedirs(SAVE_DIR, exist_ok=True)

# ── Feature and class names ───────────────────────────────────────────────────
FEATURE_NAMES = [
    "age", "cholesterol", "blood_pressure",
    "max_heart_rate", "blood_sugar",
    "chest_pain_type", "exercise_angina",
]
CLASS_NAMES = ["low risk", "high risk"]


# =============================================================================
# Step 1 — Synthetic heart disease dataset
# =============================================================================
# We build a dataset where the label (high risk / low risk) is determined
# by a known rule: a patient is high risk if age + cholesterol + blood_pressure
# is above average. This means those 3 features SHOULD dominate the attributions.
# When we see the IG explanation, we can verify it matches this known rule.

def make_dataset(n=400, seed=42):
    """
    Create a synthetic heart disease dataset with a known causal structure.

    High risk rule (ground truth):
        label = 1  if  age + cholesterol + blood_pressure  > 0
        label = 0  otherwise

    Features are normalised (mean≈0, std≈1) so a zero baseline is valid.
    The mean baseline computed from X_train will also work well.
    """
    rng = np.random.RandomState(seed)

    # 7 features, all roughly N(0, 1)
    X = rng.randn(n, 7).astype(np.float32)

    # Label based on the first 3 features (the "real" risk factors)
    # This is the ground truth — IG should reflect this
    risk_score = X[:, 0] + X[:, 1] + X[:, 2]   # age + cholesterol + bp
    y = (risk_score > 0).astype(np.int64)

    split = int(n * 0.75)
    return X[:split], y[:split], X[split:], y[split:]


X_train, y_train, X_test, y_test = make_dataset()
print(f"\n[Dataset]  Train: {X_train.shape}  Test: {X_test.shape}")
print(f"           Class balance — train: {y_train.mean():.1%} high-risk")


# =============================================================================
# Step 2 — Define and train the model
# =============================================================================

class HeartRiskMLP(nn.Module):
    """
    MLP for heart disease binary classification.
    Output: [batch, 2] logits for [low_risk, high_risk].
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(7, 64),  nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.net(x)


def train_model(X_tr, y_tr, epochs=150, lr=1e-3, seed=0):
    """Train HeartRiskMLP and return the trained model."""
    torch.manual_seed(seed)
    model     = HeartRiskMLP()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    X_t = torch.tensor(X_tr)
    y_t = torch.tensor(y_tr)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(X_t), y_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(X_t).argmax(dim=1).numpy()
    acc = (preds == y_tr).mean()
    return model, acc


print("\n[Training] HeartRiskMLP  (150 epochs, Adam, lr=0.001)...")
model, train_acc = train_model(X_train, y_train)
print(f"           Training accuracy: {train_acc:.1%}")

# Evaluate on test set
model.eval()
with torch.no_grad():
    test_preds = model(torch.tensor(X_test)).argmax(dim=1).numpy()
test_acc = (test_preds == y_test).mean()
print(f"           Test accuracy:     {test_acc:.1%}")


# =============================================================================
# Step 3 — Choose a patient to explain
# =============================================================================
# We pick a test patient who is confidently predicted as HIGH RISK.
# We want a clear, decisive prediction so the attribution story is easy to read.

model.eval()
with torch.no_grad():
    test_tensor = torch.tensor(X_test)
    probs       = torch.softmax(model(test_tensor), dim=1).numpy()

# Find the test sample with highest confidence for high_risk (class 1)
high_risk_conf = probs[:, 1]
patient_idx    = int(np.argmax(high_risk_conf))
patient_tensor = torch.tensor(X_test[patient_idx]).unsqueeze(0)  # [1, 7]
patient_prob   = high_risk_conf[patient_idx]

print(f"\n[Patient]  Chose test sample #{patient_idx}")
print(f"           Predicted: HIGH RISK  ({patient_prob:.1%} confidence)")
print(f"           True label: {'HIGH RISK' if y_test[patient_idx] == 1 else 'low risk'}")
print(f"\n           Feature values vs population mean:")
pop_mean = X_train.mean(axis=0)
for i, name in enumerate(FEATURE_NAMES):
    val  = X_test[patient_idx, i]
    diff = val - pop_mean[i]
    bar  = "▲" if diff > 0.3 else ("▼" if diff < -0.3 else "≈")
    print(f"           {name:<22} = {val:+.3f}  (mean={pop_mean[i]:+.3f})  {bar}")


# =============================================================================
# Step 4 — Run IGTabularExplainer
# =============================================================================

print("\n[IG]  Running IGTabularExplainer...")
print(f"      Steps: 200  |  Baseline: training mean  |  Explaining: class 1 (high risk)")

explainer = IGTabularExplainer(model, feature_names=FEATURE_NAMES)

results = explainer.explain(
    input_tensor  = patient_tensor,
    training_data = X_train,        # → mean baseline computed automatically
    target_class  = 1,              # explain "high risk" class
    steps         = 200,
    batch_size    = 64,
)

delta      = results["convergence_delta"]
attr       = results["attributions"]
quality    = "EXCELLENT" if delta < 0.05 else "OK" if delta < 0.15 else "[!!] increase steps"

print(f"\n      Convergence delta = {delta:.5f}  [{quality}]")
print(f"      Explained class   = {results['target_class']} ({CLASS_NAMES[results['target_class']]})")

# Print the attribution table so the user sees it in the terminal too
print(f"\n      Attribution ranking (most → least important):")
ranked = sorted(enumerate(attr), key=lambda x: abs(x[1]), reverse=True)
for rank, (feat_i, val) in enumerate(ranked, 1):
    direction = "→ supports HIGH RISK" if val > 0 else "→ suppresses HIGH RISK"
    bar = "█" * max(1, int(abs(val) / max(abs(attr)) * 20))
    print(f"      #{rank}  {FEATURE_NAMES[feat_i]:<22} {val:+.4f}  {bar}  {direction}")


# =============================================================================
# Step 5 — Save all 5 individual charts
# =============================================================================

print(f"\n[Saving]  Writing visualisations to '{SAVE_DIR}/'...")

chart_files = {
    "tabular_bar.png":          results["chart_bar"],
    "tabular_force.png":        results["chart_force"],
    "tabular_waterfall.png":    results["chart_waterfall"],
    "tabular_distribution.png": results["chart_distribution"],
    "tabular_summary.png":      results["chart_summary"],
}

for filename, fig in chart_files.items():
    path = os.path.join(SAVE_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"         Saved: {path}")
    plt.close(fig)

# Full dashboard
dashboard_path = os.path.join(SAVE_DIR, "tabular_dashboard.png")
explainer.save_dashboard(
    results,
    save_path  = dashboard_path,
    class_name = f"High Risk  ({patient_prob:.1%} confidence)",
    dpi        = 150,
)
print(f"         Saved: {dashboard_path}  ← full 5-panel dashboard")


# =============================================================================
# Step 6 — Verify explanation matches known ground truth
# =============================================================================
# Our dataset was built so that age, cholesterol, blood_pressure determine the label.
# IG should reflect this: those 3 features must dominate the attributions.

print(f"\n[Verification]  Does the explanation match the known data rule?")
print(f"                Rule: label = 1 if age + cholesterol + blood_pressure > 0")

ground_truth_features = {0, 1, 2}   # indices of age, cholesterol, blood_pressure
top3_indices = set([feat_i for feat_i, _ in ranked[:3]])

matches = len(ground_truth_features & top3_indices)
print(f"\n                Top 3 features by IG:   "
      f"{[FEATURE_NAMES[i] for i,_ in ranked[:3]]}")
print(f"                Ground truth causes:    "
      f"{[FEATURE_NAMES[i] for i in sorted(ground_truth_features)]}")
print(f"                Overlap: {matches}/3 ground truth features in top 3")

if matches >= 2:
    print(f"\n                ✓ EXPLANATION IS CORRECT — IG found the real causes")
else:
    print(f"\n                ~ Partial match — model may have learned indirect patterns")


# =============================================================================
# Summary
# =============================================================================

print()
print("=" * 62)
print("  DONE")
print("=" * 62)
print(f"  Model          : HeartRiskMLP  (train acc={train_acc:.1%}  test acc={test_acc:.1%})")
print(f"  Patient        : test sample #{patient_idx}")
print(f"  Prediction     : HIGH RISK  ({patient_prob:.1%})")
print(f"  Delta          : {delta:.5f}  [{quality}]")
print(f"  Outputs saved  : {SAVE_DIR}/")
print()
print("  Files saved:")
print("  ┌─ tabular_dashboard.png    ← open this first (all 5 panels)")
print("  ├─ tabular_bar.png          ← which features push toward/away from HIGH RISK")
print("  ├─ tabular_force.png        ← how each feature shifted the score")
print("  ├─ tabular_waterfall.png    ← cumulative attribution build-up")
print("  ├─ tabular_distribution.png ← this patient vs the full population")
print("  └─ tabular_summary.png      ← ranked table with importance bars")
print()
print("  How to read:")
print("  GREEN attribution = this feature is pushing toward HIGH RISK")
print("  RED   attribution = this feature is pulling away from HIGH RISK")
print("  The largest absolute value = the most influential feature for this patient")
print("=" * 62)
print()