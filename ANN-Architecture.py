import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# 1. SIMULATED DATASET (6 FEATURES)
# -------------------------
print("="*80)
print("CBR PREDICTION USING MGO-TRAINED ANN (6 INPUTS)")
print("="*80)

np.random.seed(42)
feature_columns = ['G', 'S', 'F', 'PI', 'MDD', 'OMC']
num_samples = 20

df = pd.DataFrame({
    col: np.random.uniform(low=5, high=50, size=num_samples)
    for col in feature_columns
})

print("Dataset shape:", df.shape)
print("First 5 rows:\n", df.head())

# Normalization parameters
X_min = df.min().values
X_max = df.max().values
CBR_min = 2.90
CBR_max = 19.04

# -------------------------
# 2. MGO-ANN MODEL PARAMETERS (6 → 6 → 1)
# -------------------------
W1 = np.array([
    [-0.2536, -0.1101, 0.0764, 0.5012, -0.3024, 0.4061],
    [-0.4607, 0.2384, -0.1213, 0.2205, -0.1981, 0.3854],
    [0.1023, -0.1324, 0.4123, -0.3841, 0.2735, -0.1982],
    [0.3056, 0.4851, -0.2024, 0.1104, 0.3485, -0.1275],
    [-0.2451, -0.3204, 0.1825, 0.2954, 0.3925, 0.2041],
    [-0.1203, -0.2845, -0.1546, 0.2845, 0.3241, -0.2104]
])

b1 = np.array([-0.9167, -0.5324, 0.1842, 0.4721, -0.3254, 0.2104])
W2 = np.array([-1.022, -0.845, 0.412, 0.295, -0.654, -1.234])
theta = 1.6104

print(f"ANN Architecture: {len(feature_columns)} inputs → 6 hidden neurons → 1 output")

# -------------------------
# 3. PREDICTION FUNCTION
# -------------------------
def predict_cbr(features):
    # Step 1: Normalization [-1, 1]
    X_norm = 2 * (features - X_min) / (X_max - X_min) - 1

    # Step 2: Hidden layer
    alpha = np.dot(X_norm, W1) + b1
    h = 1 / (1 + np.exp(-alpha))  # sigmoid

    # Step 3: Output neuron
    CBR_norm = np.dot(h, W2) + theta

    # Step 4: Denormalization
    CBR_pred = ((CBR_norm + 1) * (CBR_max - CBR_min) / 2) + CBR_min

    return CBR_pred, X_norm, alpha, h, CBR_norm

# -------------------------
# 4. PREDICT FOR FIRST SAMPLE
# -------------------------
X_raw = df.iloc[0].values
CBR_pred, X_norm, alpha, h, CBR_norm = predict_cbr(X_raw)

print(f"\nPredicted CBR for first sample: {CBR_pred:.4f}")

# -------------------------
# 5. CALCULATION DETAILS
# -------------------------
print("\nSTEP 1: INPUT NORMALIZATION")
norm_df = pd.DataFrame({
    'Feature': feature_columns,
    'Raw Value': X_raw,
    'Normalized': X_norm
})
print(norm_df.to_string(index=False))

print("\nSTEP 2: HIDDEN LAYER (α values)")
for i, a in enumerate(alpha):
    print(f"α{i+1}: {a:.6f}")

print("\nSTEP 3: HIDDEN LAYER SIGMOID OUTPUTS (h values)")
for i, val in enumerate(h):
    print(f"h{i+1}: {val:.6f}")

print("\nSTEP 4: OUTPUT NEURON")
print(f"Normalized Output CBR': {CBR_norm:.6f}")

print("\nSTEP 5: FINAL DENORMALIZED OUTPUT")
print(f"Final Predicted CBR: {CBR_pred:.4f}")

# -------------------------
# 6. VISUALIZATION (MODERN COLORS)
# -------------------------
teal = '#008080'
orange = '#FF6F00'
light_gray = '#B0B0B0'
gold = '#FFD700'

plt.style.use('default')
sns.set_palette([teal, orange, light_gray, gold])

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 14))

# --- 1. Input Feature Values ---
bars1 = ax1.bar(feature_columns, X_raw, color=teal)
ax1.set_title('Input Feature Values', fontsize=16, fontweight='bold')
ax1.grid(axis='y', linestyle='--', alpha=0.7)

for bar, val in zip(bars1, X_raw):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
             f'{val:.1f}', ha='center', va='bottom', fontsize=12)

# --- 2. Hidden Layer Outputs ---
bars2 = ax2.bar([f'h{i+1}' for i in range(len(h))], h, color=orange)
ax2.set_title('Hidden Layer Sigmoid Outputs', fontsize=16, fontweight='bold')
ax2.grid(axis='y', linestyle='--', alpha=0.7)
ax2.set_ylim(0, 1)

# --- 3. Hidden Neuron Contributions ---
contributions = h * W2
bars3 = ax3.bar([f'h{i+1}' for i in range(len(h))], contributions, color=teal)
ax3.axhline(0, color='black')
ax3.set_title('Hidden Neuron Contributions to Output', fontsize=16, fontweight='bold')
ax3.grid(axis='y', linestyle='--', alpha=0.7)

# --- 4. Final Prediction ---
bars4 = ax4.bar(['Predicted CBR'], [CBR_pred], color=gold)
ax4.set_title('Final Predicted CBR', fontsize=16, fontweight='bold')
ax4.grid(axis='y', linestyle='--', alpha=0.7)
ax4.text(0, CBR_pred/2, f'{CBR_pred:.3f}', fontsize=20, ha='center', color='black')

plt.tight_layout()
plt.show()

# -------------------------
# 7. PREDICT FOR MULTIPLE SAMPLES
# -------------------------
print("\nPredicted CBR for first 10 samples:")
for i in range(min(10, len(df))):
    pred, _, _, _, _ = predict_cbr(df.iloc[i].values)
    print(f"Sample {i:2d} → CBR = {pred:.4f}")

print("\n✅ Prediction complete — TLBO-trained ANN (6 inputs)")