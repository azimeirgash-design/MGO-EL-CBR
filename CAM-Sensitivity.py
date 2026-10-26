import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

# ------------------- GLOBAL STYLE SETTINGS -------------------
plt.style.use('default')
mpl.rcParams['font.size'] = 14
mpl.rcParams['font.weight'] = 'bold'
mpl.rcParams['axes.edgecolor'] = '#DDDDDD'
mpl.rcParams['axes.linewidth'] = 0.8


# ------------------- LOAD SPSS2 EXCEL DATA -------------------
def load_spss2_data(file_path='/content/SPSS2.xlsx'):
    """
    Load CBR data from SPSS2 Excel file.
    """
    try:
        df = pd.read_excel(file_path)
        df.columns = df.columns.str.strip()

        print(f"✅ File loaded successfully: {file_path}")
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print("\nFirst 5 rows:")
        print(df.head())

        expected_columns = ['CBR (Y1)', 'G (X1)', 'S (X2)', 'F (X3)',
                            'PI (X4)', 'MDD (X7)', 'OMC (X8)']

        missing_cols = [col for col in expected_columns if col not in df.columns]
        if missing_cols:
            print(f"⚠️ Warning: Missing columns: {missing_cols}")
            print(f"Available columns: {list(df.columns)}")

        return df

    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None


# ------------------- PREPARE DATA FROM SPSS2 -------------------
def prepare_spss2_data(df, min_valid_frac=0.5):
    """
    Prepare features and target from SPSS2 data.

    Drops columns with no cached values (Excel external-link formulas produce
    NaN when read by pandas) and drops constant columns (which make
    np.corrcoef return NaN and poison the CAM score vector).
    """
    feature_mapping = {
        'G (X1)':   'Gravel Content (G)',
        'S (X2)':   'Sand Content (S)',
        'F (X3)':   'Fines',
        'PI (X4)':  'Plasticity Index (PI)',
        'MDD (X7)': 'Maximum Dry Density (MDD)',
        'OMC (X8)': 'Optimum Moisture Content (OMC)',
    }
    target_column = 'CBR (Y1)'

    if target_column not in df.columns:
        print(f"❌ Target column '{target_column}' not found!")
        return None, None, None

    feature_columns = [c for c in feature_mapping if c in df.columns]
    if not feature_columns:
        print("❌ No feature columns found in the dataset!")
        return None, None, None

    num = df[feature_columns + [target_column]].apply(pd.to_numeric, errors='coerce')

    # ---- report columns with no real values (external-link formulas) ----
    valid_frac = num[feature_columns].notna().mean()
    usable = valid_frac[valid_frac >= min_valid_frac].index.tolist()
    unusable = [c for c in feature_columns if c not in usable]

    if unusable:
        print("\n⚠️  Dropping columns with no cached values "
              "(Excel external-link formulas):")
        for c in unusable:
            print(f"   - {c:10s}  valid values: {valid_frac[c]*100:5.1f}%")

    if not usable:
        print("❌ No usable feature columns remain. "
              "Break the external links in Excel and re-save.")
        return None, None, None

    # ---- drop rows with missing target ----
    n0 = len(num)
    num = num.dropna(subset=[target_column])
    if len(num) < n0:
        print(f"ℹ️  Dropped {n0 - len(num)} rows with missing target.")

    X_df = num[usable].copy()

    # ---- drop constant columns (these cause NaN correlations) ----
    const_cols = [c for c in X_df.columns if X_df[c].nunique(dropna=True) <= 1]
    if const_cols:
        print(f"ℹ️  Dropping constant columns: {const_cols}")
        X_df = X_df.drop(columns=const_cols)

    if X_df.shape[1] == 0:
        print("❌ All features are constant — nothing to analyse.")
        return None, None, None

    # ---- median-impute only genuine gaps ----
    X_df = X_df.fillna(X_df.median(numeric_only=True))

    X = X_df.values
    y = num[target_column].values
    feature_names = [feature_mapping[c] for c in X_df.columns]

    print(f"\n📊 Data Summary:")
    print(f"Number of samples:  {len(X)}")
    print(f"Number of features: {len(feature_names)}")
    print(f"Features: {feature_names}")
    print(f"Target range: {y.min():.2f} to {y.max():.2f}")

    print("\n📈 Feature Statistics:")
    print(pd.DataFrame(X, columns=feature_names).describe().round(2))

    return X, y, feature_names


# ------------------- SINGLE-SEED CAM -------------------
def _run_single_seed_cam(X, y, feature_names, n_components, hidden_layer_size, seed):
    """
    Fit one MLP and return (cam_scores, correlation_with_cbr, model, pca).
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    y_flat = y.ravel() if len(y.shape) > 1 else y

    # NaN-safe correlation with CBR (constant column -> 0)
    correlation_with_cbr = []
    for i in range(X_scaled.shape[1]):
        col = X_scaled[:, i]
        if np.std(col) < 1e-12:
            correlation_with_cbr.append(0.0)
        else:
            correlation_with_cbr.append(float(np.corrcoef(col, y_flat)[0, 1]))
    correlation_with_cbr = np.array(correlation_with_cbr)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_flat, test_size=0.2, random_state=seed
    )

    model = MLPRegressor(
        hidden_layer_sizes=(hidden_layer_size,),
        activation='tanh',
        solver='adam',
        max_iter=1000,
        random_state=seed,
        early_stopping=True,
        validation_fraction=0.1
    )
    model.fit(X_train, y_train)

    # ------------------- CAM -------------------
    hidden_activations = np.tanh(X_train @ model.coefs_[0] + model.intercepts_[0])

    n_comp = min(n_components, hidden_activations.shape[1],
                 hidden_activations.shape[0])
    pca = PCA(n_components=n_comp)
    components = pca.fit_transform(hidden_activations)

    n_features = len(feature_names)
    activation_patterns = np.zeros((n_comp, n_features))
    for comp_idx in range(n_comp):
        for feat_idx in range(n_features):
            col = X_train[:, feat_idx]
            if np.std(col) < 1e-12 or np.std(components[:, comp_idx]) < 1e-12:
                activation_patterns[comp_idx, feat_idx] = 0.0
            else:
                activation_patterns[comp_idx, feat_idx] = abs(
                    np.corrcoef(col, components[:, comp_idx])[0, 1]
                )

    cam_scores = np.mean(activation_patterns, axis=0)
    s = cam_scores.sum()
    if s > 0:
        cam_scores = cam_scores / s

    return cam_scores, correlation_with_cbr, model, pca, activation_patterns


# ------------------- SEED-AVERAGED CAM ANALYSIS -------------------
def perform_cam_analysis(X, y, feature_names, n_components=3,
                         hidden_layer_size=10, n_seeds=30, seed_base=0):
    """
    Run CAM over multiple random seeds and average the importances.
    A single split on ~260 samples is not reproducible; averaging over seeds
    gives stable rankings and standard deviations.
    """
    cam_runs = []
    corr_runs = []
    pattern_runs = []
    last_model = None
    last_pca = None

    print(f"\nTraining {n_seeds} neural networks (seeds {seed_base}.."
          f"{seed_base + n_seeds - 1}) ...")

    for s in range(n_seeds):
        seed = seed_base + s
        cam, corr, model, pca, patterns = _run_single_seed_cam(
            X, y, feature_names, n_components, hidden_layer_size, seed
        )
        cam_runs.append(cam)
        corr_runs.append(corr)
        pattern_runs.append(patterns)
        last_model, last_pca = model, pca

    cam_runs = np.vstack(cam_runs)          # (n_seeds, n_features)
    corr_runs = np.vstack(corr_runs)        # (n_seeds, n_features)

    cam_mean = cam_runs.mean(axis=0)
    cam_std = cam_runs.std(axis=0, ddof=1) if n_seeds > 1 else np.zeros_like(cam_mean)
    corr_mean = corr_runs.mean(axis=0)

    if cam_mean.sum() > 0:
        cam_mean = cam_mean / cam_mean.sum()

    # Average activation-pattern matrix across seeds
    activation_patterns = np.mean(pattern_runs, axis=0)

    combined_results = pd.DataFrame({
        'Feature': feature_names,
        'CAM_Score': cam_mean,
        'CAM_Std': cam_std,
        'Correlation_with_CBR': corr_mean,
    }).sort_values('CAM_Score', ascending=False).reset_index(drop=True)

    return combined_results, activation_patterns, last_model, last_pca


# ------------------- VISUALIZATION FUNCTIONS -------------------
def plot_cam_ranking(combined_results):
    """Single horizontal bar chart of CAM importance."""
    fig, ax = plt.subplots(figsize=(10, 6))

    df = combined_results.sort_values('CAM_Score', ascending=True)
    colors = plt.cm.YlOrRd(np.linspace(0.4, 0.9, len(df)))

    bars = ax.barh(df['Feature'], df['CAM_Score'],
                   xerr=df['CAM_Std'],
                   color=colors, edgecolor='white', height=0.6,
                   alpha=0.9, error_kw=dict(ecolor='#333333', lw=1))

    ax.set_xlabel('CAM Importance Score', fontsize=12, fontweight='bold')
    ax.set_title('CAM Feature Importance', fontsize=14, fontweight='bold')
    ax.grid(axis='x', linestyle='--', alpha=0.3)

    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.004, bar.get_y() + bar.get_height() / 2,
                f'{width:.3f}', ha='left', va='center',
                fontweight='bold', fontsize=10)

    plt.tight_layout()
    plt.show()


def plot_activation_patterns(activation_patterns, feature_names):
    """Heat-map of |correlation| between features and each PCA component."""
    fig, ax = plt.subplots(figsize=(12, 6))

    im = ax.imshow(activation_patterns, aspect='auto', cmap='YlOrRd')

    ax.set_xticks(np.arange(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=45, ha='right')
    ax.set_yticks(np.arange(activation_patterns.shape[0]))
    ax.set_yticklabels([f'PCA Component {i+1}'
                        for i in range(activation_patterns.shape[0])])

    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel('|Correlation|', rotation=-90, va="bottom")

    vmax = activation_patterns.max() if activation_patterns.size else 1.0
    for i in range(activation_patterns.shape[0]):
        for j in range(activation_patterns.shape[1]):
            val = activation_patterns[i, j]
            ax.text(j, i, f'{val:.3f}',
                    ha="center", va="center",
                    color="white" if val > 0.6 * vmax else "black",
                    fontsize=9, fontweight='bold')

    ax.set_title('CAM Activation Patterns: '
                 'Feature → PCA Component Correlation',
                 fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()


def plot_correlation_analysis(combined_results):
    """CAM score vs correlation with CBR."""
    fig, ax = plt.subplots(figsize=(9, 7))

    ax.scatter(combined_results['Correlation_with_CBR'],
               combined_results['CAM_Score'],
               s=combined_results['CAM_Score'] * 1000 + 50,
               c=combined_results['CAM_Score'],
               cmap='YlOrRd', alpha=0.85, edgecolors='black')

    ax.set_xlabel('Correlation with CBR', fontsize=12, fontweight='bold')
    ax.set_ylabel('CAM Score', fontsize=12, fontweight='bold')
    ax.set_title('CAM Score vs Correlation with CBR',
                 fontsize=14, fontweight='bold')
    ax.axhline(y=combined_results['CAM_Score'].mean(),
               color='gray', linestyle='--', alpha=0.5,
               label='Mean CAM')
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    ax.grid(True, alpha=0.3)
    ax.legend()

    for _, row in combined_results.iterrows():
        ax.annotate(row['Feature'].split('(')[0].strip(),
                    xy=(row['Correlation_with_CBR'], row['CAM_Score']),
                    xytext=(6, 6), textcoords='offset points',
                    fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.show()


def plot_final_summary(combined_results, activation_patterns, feature_names):
    """Compact 2x2 summary: ranking, correlation, activation heat-map, cumulative."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    axes = axes.flatten()

    # 1. CAM ranking
    df_sorted = combined_results.sort_values('CAM_Score', ascending=True)
    colors = plt.cm.YlOrRd(np.linspace(0.4, 0.9, len(df_sorted)))
    bars = axes[0].barh(df_sorted['Feature'], df_sorted['CAM_Score'],
                        xerr=df_sorted['CAM_Std'],
                        color=colors, edgecolor='white', height=0.6,
                        alpha=0.9,
                        error_kw=dict(ecolor='#333333', lw=1))
    axes[0].set_xlabel('CAM Importance Score', fontsize=11, fontweight='bold')
    axes[0].set_title('CAM Feature Ranking', fontsize=13, fontweight='bold')
    axes[0].grid(axis='x', alpha=0.3, linestyle='--')
    for bar in bars:
        w = bar.get_width()
        axes[0].text(w + 0.004, bar.get_y() + bar.get_height() / 2,
                     f'{w:.3f}', ha='left', va='center',
                     fontweight='bold', fontsize=9)

    # 2. CAM score vs correlation
    axes[1].scatter(combined_results['Correlation_with_CBR'],
                    combined_results['CAM_Score'],
                    s=combined_results['CAM_Score'] * 1000 + 50,
                    c=combined_results['CAM_Score'],
                    cmap='YlOrRd', alpha=0.85, edgecolors='black')
    axes[1].set_xlabel('Correlation with CBR', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('CAM Score', fontsize=11, fontweight='bold')
    axes[1].set_title('CAM vs Correlation', fontsize=13, fontweight='bold')
    axes[1].axhline(y=combined_results['CAM_Score'].mean(),
                    color='gray', linestyle='--', alpha=0.5)
    axes[1].axvline(x=0, color='black', linestyle='-', alpha=0.3)
    axes[1].grid(True, alpha=0.3)
    for _, row in combined_results.iterrows():
        axes[1].annotate(row['Feature'].split('(')[0].strip(),
                         xy=(row['Correlation_with_CBR'], row['CAM_Score']),
                         xytext=(6, 6), textcoords='offset points',
                         fontsize=9, fontweight='bold')

    # 3. Activation pattern heat-map
    im = axes[2].imshow(activation_patterns, aspect='auto', cmap='YlOrRd')
    axes[2].set_xticks(np.arange(len(feature_names)))
    axes[2].set_xticklabels([f.split('(')[0].strip() for f in feature_names],
                            rotation=45, ha='right', fontsize=9)
    axes[2].set_yticks(np.arange(activation_patterns.shape[0]))
    axes[2].set_yticklabels([f'PC {i+1}'
                             for i in range(activation_patterns.shape[0])],
                            fontsize=9)
    axes[2].set_title('Activation Patterns (|corr|)',
                      fontsize=13, fontweight='bold')
    fig.colorbar(im, ax=axes[2], shrink=0.8)

    # 4. Cumulative importance
    df_cum = combined_results.sort_values('CAM_Score', ascending=False).copy()
    df_cum['Cumulative'] = df_cum['CAM_Score'].cumsum()
    axes[3].plot(range(1, len(df_cum) + 1), df_cum['Cumulative'],
                 marker='o', color='#2E86AB', linewidth=2, markersize=9)
    axes[3].axhline(y=0.8, color='red', linestyle='--', alpha=0.6,
                    label='80% threshold')
    axes[3].set_xticks(range(1, len(df_cum) + 1))
    axes[3].set_xticklabels([f.split('(')[0].strip() for f in df_cum['Feature']],
                            rotation=45, ha='right', fontsize=9)
    axes[3].set_xlabel('Number of Features', fontsize=11, fontweight='bold')
    axes[3].set_ylabel('Cumulative CAM Importance',
                       fontsize=11, fontweight='bold')
    axes[3].set_title('Cumulative Importance',
                      fontsize=13, fontweight='bold')
    axes[3].set_ylim(0, 1.05)
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()

    plt.suptitle('CAM Feature Importance Analysis',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()


# ------------------- MAIN EXECUTION -------------------
def main():
    print("🔬 CBR Prediction - CAM Feature Importance Analysis (SPSS2 Data)")
    print("=" * 60)

    file_path = 'SPSS2.xlsx'   # <-- update if needed
    df = load_spss2_data(file_path)
    if df is None:
        print("❌ Failed to load data. Please check the file path.")
        return None, None

    X, y, feature_names = prepare_spss2_data(df)
    if X is None or y is None:
        print("❌ Failed to prepare data for analysis.")
        return None, None

    if len(feature_names) < 2:
        print("❌ Need at least 2 usable features. "
              "Break the Excel external links and re-save SPSS2.xlsx.")
        return None, None

    print("\n" + "=" * 60)
    print("🤖 Performing seed-averaged CAM analysis ...")
    print("=" * 60)

    n_comp = min(3, len(feature_names))
    hidden = max(4, min(10, 2 * len(feature_names)))

    combined_results, activation_patterns, model, pca = perform_cam_analysis(
        X, y, feature_names,
        n_components=n_comp,
        hidden_layer_size=hidden,
        n_seeds=30,
        seed_base=0,
    )

    # Visualisations
    plot_cam_ranking(combined_results)
    plot_activation_patterns(activation_patterns, feature_names)
    plot_correlation_analysis(combined_results)
    plot_final_summary(combined_results, activation_patterns, feature_names)

    # Results table
    print("\n" + "=" * 60)
    print("📊 CAM FEATURE IMPORTANCE RESULTS (mean over 30 seeds)")
    print("=" * 60)

    combined_results['CAM_Rank'] = combined_results['CAM_Score'].rank(
        ascending=False).astype(int)

    display_df = combined_results.sort_values(
        'CAM_Score', ascending=False).reset_index(drop=True)

    print("\nFeature Importance Rankings:")
    print("-" * 80)
    cols = ['Feature', 'CAM_Score', 'CAM_Std', 'CAM_Rank',
            'Correlation_with_CBR']
    print(display_df[cols].round(4).to_string(index=False))

    # Cumulative importance
    display_df['Cumulative'] = display_df['CAM_Score'].cumsum()

    print("\n" + "=" * 60)
    print("📈 CUMULATIVE IMPORTANCE")
    print("=" * 60)
    for i, (_, row) in enumerate(display_df.iterrows(), 1):
        print(f"Top {i:2d}: {row['Feature']:35s} "
              f"CAM = {row['CAM_Score']:.3f}  "
              f"Cumulative = {row['Cumulative']:.3f}")

    # Minimum feature set for 80% cumulative importance
    n_80 = int((display_df['Cumulative'] < 0.80).sum()) + 1
    print(f"\n🎯 Minimum feature set for ≥80% cumulative CAM importance: "
          f"{n_80} feature(s)")

    print("\n" + "=" * 60)
    print("🎯 FINAL RECOMMENDATIONS")
    print("=" * 60)

    top_features = display_df.head(3)
    for i, (_, row) in enumerate(top_features.iterrows(), 1):
        print(f"{i}. {row['Feature']}: "
              f"CAM = {row['CAM_Score']:.3f} "
              f"(±{row['CAM_Std']:.3f}), "
              f"r(CBR) = {row['Correlation_with_CBR']:+.3f}")

    return combined_results, model


# ------------------- EXECUTE -------------------
if __name__ == "__main__":
    results, model = main()