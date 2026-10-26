import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Code for sharing
"""
CBR PREDICTION USING MGO-TRAINED ANN (6 INPUTS)
================================================
Architecture  : 6 → 6 (tanh) → 1 (linear)
Training      : Mountain Gazelle Optimizer (MGO)
Normalization : Fixed min–max (manuscript Table 2), mapped to [−1, 1]
 (CBR_ANN_Figure.png)
Reported      : RMSE = 1.6427 | MAE = 1.2337 | R² = 0.7828
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================================
# 1. LOAD ACTUAL DATASET
# ==========================================================
print("=" * 80)
print("CBR PREDICTION USING MGO-TRAINED ANN (6 INPUTS)")
print("Normalization: fixed limits (Table 2) → [−1, 1]")
print("Note: MDD values are interpreted as kN/m³")
print("=" * 80)

FILE_PATH = "SPSS2.xlsx"
if not os.path.exists(FILE_PATH):
    raise FileNotFoundError(f"Cannot find {FILE_PATH} — place it next to this script.")

df_raw = pd.read_excel(FILE_PATH, sheet_name=0)
df_raw.columns = [str(c).strip() for c in df_raw.columns]

feature_columns = ["G (X1)", "S (X2)", "F (X3)", "PI (X4)", "MDD (X7)", "OMC (X8)"]
short_names     = ["G", "S", "F", "PI", "MDD", "OMC"]
target_column   = "CBR (Y1)"

df = df_raw[feature_columns + [target_column]].copy()
for c in df.columns:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna().reset_index(drop=True)

if df.empty:
    raise ValueError(
        "Dataset became empty after dropna(). The columns F (X3), PI (X4), "
        "MDD (X7), OMC (X8) are external formula references whose cached "
        "values are missing. Open SPSS2.xlsx in Excel once, save, and re-run."
    )

df_short = df[feature_columns].copy()
df_short.columns = short_names

print(f"Dataset shape: {df.shape}")
print("First 5 rows:\n", df_short.head())

# ==========================================================
# 2. FIXED NORMALIZATION LIMITS (manuscript Table 2)
# ==========================================================
#                   G      S      F      PI     MDD    OMC
X_min = np.array([ 0.00,  4.22, 19.09,  0.00,  1.29,  6.74], dtype=float)
X_max = np.array([25.32, 80.00, 95.18, 54.48,  2.23, 36.65], dtype=float)
CBR_min, CBR_max = 2.90, 19.04

print("\nTable 2 — fixed normalisation ranges")
print(pd.DataFrame({
    "Feature": short_names,
    "X_min":   X_min,
    "X_max":   X_max,
}).to_string(index=False))
print(f"  CBR range: [{CBR_min}, {CBR_max}]")

# ==========================================================
# 3. MGO-TRAINED ANN PARAMETERS (Table 3)
# ==========================================================
# Input weights βij  (6 inputs × 6 hidden neurons)
W1 = np.array([
    [ 0.3831,  2.8539,  1.5929,  2.7800,  1.4101, -0.9469],
    [ 0.3300,  0.3549, -0.9612,  0.7649,  0.1369, -1.2791],
    [ 1.1108,  0.9930,  1.6415, -2.3228,  0.9685, -1.6878],
    [ 0.2868,  0.3434,  2.0227, -1.1512, -0.1960,  0.3644],
    [ 1.2822,  0.2249,  0.1821,  1.1641, -1.7104, -0.7723],
    [-0.1783, -1.2058,  0.6978,  0.9184,  0.6792, -2.1814],
])

# Hidden biases Uj
b1 = np.array([-1.0537, 0.7188, 0.4557, 0.9836, -0.0753, 2.4713])

# Output weights Wjk
W2 = np.array([ 1.2662,  0.1863, -1.1532, -0.0046,  0.6515, -0.3489])

# Output bias Uk
b2 = 1.2032

print(f"\nANN Architecture: {len(feature_columns)} inputs → 6 hidden neurons (tanh) → 1 output")
print(f"Table 3: W1 (6×6), Uj (6), Wjk (6), Uk = {b2:.4f}\n")

# ==========================================================
# 4. PREDICTION FUNCTION  (fixed min–max + tanh hidden)
# ==========================================================
def predict_cbr(features):
    features = np.asarray(features, dtype=float).ravel()

    # STEP 1 — fixed min–max normalisation to [−1, +1]
    X_norm = 2.0 * (features - X_min) / (X_max - X_min) - 1.0

    # STEP 2 — hidden layer pre-activation
    alpha = X_norm @ W1 + b1

    # STEP 3 — hidden layer tanh output
    h = np.tanh(alpha)

    # STEP 4 — output neuron (normalised CBR, in [−1, +1] scale)
    CBR_norm = float(h @ W2 + b2)

    # STEP 5 — de-normalise back to original CBR units
    CBR_pred = 0.5 * (CBR_norm + 1.0) * (CBR_max - CBR_min) + CBR_min

    return CBR_pred, X_norm, alpha, h, CBR_norm

# ==========================================================
# 5. FORWARD PASS FOR FIRST SAMPLE (console output)
# ==========================================================
X_raw = df_short.iloc[0].to_numpy(dtype=float)
CBR_pred, X_norm, alpha, h, CBR_norm = predict_cbr(X_raw)

print(f"Predicted CBR for first sample: {CBR_pred:.4f}")

print("\nSTEP 1: INPUT NORMALISATION (MIN–MAX → [−1, +1], Table 2)")
norm_df = pd.DataFrame({
    "Feature":    short_names,
    "Raw Value":  X_raw,
    "X_min":      X_min,
    "X_max":      X_max,
    "Normalized": X_norm,
})
print(norm_df.to_string(index=False))
print("  (Note: MDD raw value is in kN/m³)")

print("\nSTEP 2: HIDDEN LAYER PRE-ACTIVATION (α values)")
for i, a in enumerate(alpha):
    print(f"α{i+1}: {a:.6f}")

print("\nSTEP 3: HIDDEN LAYER TANH OUTPUTS (h values)")
for i, val in enumerate(h):
    print(f"h{i+1}: {val:.6f}")

print("\nSTEP 4: OUTPUT NEURON")
print(f"Normalized Output CBR': {CBR_norm:.6f}   (in [−1, +1] scale)")

print("\nSTEP 5: FINAL DE-NORMALISED OUTPUT")
print(f"Final Predicted CBR: {CBR_pred:.4f}")

# ==========================================================
# 6. VISUALISATION (4-panel figure)
# ==========================================================
teal   = "#008080"
orange = "#FF6F00"
gold   = "#FFD700"

plt.style.use('default')
sns.set_palette([teal, orange, gold])

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 14))

# --- 1. Input Feature Values ---
bars1 = ax1.bar(short_names, X_raw, color=teal)
ax1.set_title('Input Feature Values', fontsize=16, fontweight='bold')
ax1.grid(axis='y', linestyle='--', alpha=0.7)
ax1.text(4, X_raw[4] * 0.9, 'kN/m³', ha='center', va='top',
         color='white', fontweight='bold')
for bar, val in zip(bars1, X_raw):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
             f'{val:.1f}', ha='center', va='bottom', fontsize=12)

# --- 2. Hidden Layer Tanh Outputs ---
bars2 = ax2.bar([f'h{i+1}' for i in range(len(h))], h, color=orange)
ax2.set_title('Hidden Layer Tanh Outputs', fontsize=16, fontweight='bold')
ax2.axhline(0, color='black', linewidth=1)
ax2.set_ylim(-1.1, 1.1)
ax2.grid(axis='y', linestyle='--', alpha=0.7)
for bar, val in zip(bars2, h):
    y = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width() / 2,
             y + (0.03 if y >= 0 else -0.03),
             f'{val:.2f}', ha='center',
             va='bottom' if y >= 0 else 'top', fontsize=11)

# --- 3. Hidden Neuron Contributions ---
contributions = h * W2
bars3 = ax3.bar([f'h{i+1}' for i in range(len(h))], contributions, color=teal)
ax3.axhline(0, color='black')
ax3.set_title('Hidden Neuron Contributions (hⱼ · Wⱼₖ)',
              fontsize=16, fontweight='bold')
ax3.grid(axis='y', linestyle='--', alpha=0.7)
for bar, val in zip(bars3, contributions):
    y = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width() / 2,
             y + (0.02 if y >= 0 else -0.02),
             f'{val:.3f}', ha='center',
             va='bottom' if y >= 0 else 'top', fontsize=10)

# --- 4. Final Prediction ---
bars4 = ax4.bar(['Predicted CBR'], [CBR_pred], color=gold)
ax4.set_title('Final Predicted CBR', fontsize=16, fontweight='bold')
ax4.grid(axis='y', linestyle='--', alpha=0.7)
ax4.text(0, CBR_pred / 2, f'{CBR_pred:.3f}',
         fontsize=20, ha='center', color='black')

plt.tight_layout()
plt.savefig("CBR_ANN_Figure.png", dpi=150)
plt.show()

print("\n✅ Prediction complete — MGO-trained ANN (6–6–1, tanh hidden)")
print("   Weights: W1=βij, b1=Uj, W2=Wjk, b2=Uk (Table 3)")
print("   Normalization: Table 2 fixed limits → [−1, +1]")