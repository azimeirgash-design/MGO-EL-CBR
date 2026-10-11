MGO-based models
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import warnings
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

from scipy import stats

warnings.filterwarnings('ignore')
np.random.seed(42)
random.seed(42)


# ---------------------- MGO (Mountain Gazelle Optimization) ----------------------
class MountainGazelleOptimization:
    def __init__(self, n_gazelles=30, n_iterations=100, lb=-10, ub=10, dim=None):
        self.n_gazelles = n_gazelles
        self.n_iterations = n_iterations
        self.lb = lb
        self.ub = ub
        self.dim = dim
        self.positions = None
        self.fitness = None
        self.best_position = None
        self.best_fitness = float('inf')

    def initialize_population(self, dim):
        self.dim = dim
        self.positions = np.random.uniform(self.lb, self.ub, (self.n_gazelles, dim))
        self.fitness = np.full(self.n_gazelles, float('inf'))

    def evaluate_fitness(self, fitness_function):
        for i in range(self.n_gazelles):
            self.fitness[i] = fitness_function(self.positions[i])
            if self.fitness[i] < self.best_fitness:
                self.best_fitness = self.fitness[i]
                self.best_position = self.positions[i].copy()

    def update_positions(self, iteration):
        a = 2 * iteration * (2 / max(1, self.n_iterations))

        for i in range(self.n_gazelles):
            r1 = np.random.random()
            r2 = np.random.random()
            r3 = np.random.random()

            if r1 < 0.5:
                D_predator = np.abs(r2 * self.best_position - self.positions[i])
                self.positions[i] = self.best_position - a * D_predator
            else:
                random_gazelle = self.positions[np.random.randint(0, self.n_gazelles)]
                D_random = np.abs(r3 * random_gazelle - self.positions[i])
                self.positions[i] = random_gazelle - a * D_random

            self.positions[i] = np.clip(self.positions[i], self.lb, self.ub)


# ---------------------- Hybrid model using ANN, Ridge, XGBoost ----------------------
class MGOHybridModel:
    def __init__(self, ann_units=[64, 32], ridge_alpha=1.0, xgb_params=None,
                 n_gazelles=20, n_iterations=50):
        self.ann_units = ann_units
        self.ridge_alpha = ridge_alpha
        self.xgb_params = xgb_params or {'max_depth': 6, 'learning_rate': 0.1, 'n_estimators': 100}
        self.n_gazelles = n_gazelles
        self.n_iterations = n_iterations

        self.scaler = StandardScaler()
        self.ann_model = None
        self.ridge_model = None
        self.xgb_model = None
        self.optimal_weights = None
        self.mgo = None

    def create_ann_model(self, input_dim, units):
        model = Sequential()
        model.add(Dense(units[0], activation='relu', input_dim=input_dim))
        model.add(Dropout(0.2))
        for units_count in units[1:]:
            model.add(Dense(units_count, activation='relu'))
            model.add(Dropout(0.2))
        model.add(Dense(1, activation='linear'))
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model

    def train_base_models(self, X_train, y_train, X_val_internal, y_val_internal, epochs=100):
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val_internal)

        print("Training ANN...")
        self.ann_model = self.create_ann_model(X_train.shape[1], self.ann_units)
        self.ann_model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val_internal),
            epochs=epochs, batch_size=32, verbose=0
        )

        print("Training Ridge Regression...")
        self.ridge_model = Ridge(alpha=self.ridge_alpha)
        self.ridge_model.fit(X_train_scaled, y_train)

        print("Training XGBoost...")
        self.xgb_model = xgb.XGBRegressor(**self.xgb_params, verbosity=0)
        self.xgb_model.fit(X_train_scaled, y_train)

    def get_base_predictions(self, X):
        X_scaled = self.scaler.transform(X)
        ann_pred = self.ann_model.predict(X_scaled).flatten()
        ridge_pred = self.ridge_model.predict(X_scaled)
        xgb_pred = self.xgb_model.predict(X_scaled)
        return np.column_stack([ann_pred, ridge_pred, xgb_pred])

    def weighted_ensemble_fitness(self, weights, X, y_true):
        weights = np.abs(weights)
        s = np.sum(weights)
        if s == 0:
            weights = np.ones_like(weights) / len(weights)
        else:
            weights = weights / s
        base_preds = self.get_base_predictions(X)
        ensemble_pred = np.sum(base_preds * weights, axis=1)
        return np.sqrt(mean_squared_error(y_true, ensemble_pred))

    def optimize_weights(self, X_val_internal, y_val_internal):
        print("Optimizing ensemble weights using Mountain Gazelle Optimization...")

        self.mgo = MountainGazelleOptimization(
            n_gazelles=self.n_gazelles,
            n_iterations=self.n_iterations,
            lb=0.1,
            ub=10.0
        )

        def fitness_function(weights):
            return self.weighted_ensemble_fitness(weights, X_val_internal, y_val_internal)

        self.mgo.initialize_population(dim=3)

        for iteration in range(self.n_iterations):
            self.mgo.evaluate_fitness(fitness_function)
            self.mgo.update_positions(iteration)

            if (iteration + 1) % 10 == 0 or iteration == 0 or iteration == self.n_iterations - 1:
                print(f"Iteration {iteration + 1}/{self.n_iterations}, Best RMSE: {self.mgo.best_fitness:.6f}")

        self.optimal_weights = np.abs(self.mgo.best_position)
        self.optimal_weights = self.optimal_weights / np.sum(self.optimal_weights)

        print(f"\nOptimal weights found: ANN={self.optimal_weights[0]:.3f}, "
              f"Ridge={self.optimal_weights[1]:.3f}, XGBoost={self.optimal_weights[2]:.3f}")

        return self.optimal_weights

    def predict(self, X):
        base_preds = self.get_base_predictions(X)
        ensemble_pred = np.sum(base_preds * self.optimal_weights, axis=1)
        return ensemble_pred

    def evaluate(self, X, y_true):
        predictions = self.predict(X)
        rmse = np.sqrt(mean_squared_error(y_true, predictions))
        mae = mean_absolute_error(y_true, predictions)
        r2 = r2_score(y_true, predictions)
        return {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'Predictions': predictions}


def descriptive_stats_series(arr):
    s = pd.Series(arr)
    mode_val = s.mode()
    mode_val = float(mode_val.iloc[0]) if not mode_val.empty else np.nan

    stats_dict = {
        'Mean': float(s.mean()),
        'Median': float(s.median()),
        'Mode': mode_val,
        'SD': float(s.std(ddof=1)),
        'Kurtosis': float(s.kurtosis()),
        'Skewness': float(s.skew()),
        'Range': float(s.max() - s.min()),
        'Min': float(s.min()),
        'Max': float(s.max()),
        'Count': int(s.count())
    }
    return stats_dict


def load_and_preprocess_data(file_path):
    df = pd.read_excel(file_path, sheet_name=0)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df_clean = df.dropna().reset_index(drop=True)
    print(f"Original data shape: {df.shape}, Cleaned: {df_clean.shape}")
    return df_clean


def main():
    file_path = "SPSS2.xlsx"
    use_sample = False

    if not os.path.exists(file_path):
        print(f"File '{file_path}' not found. Using sample synthetic data for demo.")
        use_sample = True

    if use_sample:
        from sklearn.datasets import make_regression
        X, y = make_regression(n_samples=400, n_features=6, noise=10.0, random_state=42)
        cols = [f'X{i+1}' for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=cols)
        df.insert(0, 'CBR', y)
    else:
        df = load_and_preprocess_data(file_path)

    target_col = df.columns[0]
    feature_columns = [c for c in df.columns if c != target_col]

    X = df[feature_columns].values
    y = df[target_col].values

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train, X_val_internal, y_train, y_val_internal = train_test_split(
        X_train_full, y_train_full, test_size=0.125, random_state=42
    )

    print(f"Data splits (train / internal-val / test): "
          f"{X_train.shape[0]} / {X_val_internal.shape[0]} / {X_test.shape[0]} "
          f"| Features: {X_train.shape[1]}")

    hybrid_model = MGOHybridModel(
        ann_units=[128, 64, 32],
        ridge_alpha=0.5,
        xgb_params={'max_depth': 6, 'learning_rate': 0.05, 'n_estimators': 150,
                    'subsample': 0.8, 'colsample_bytree': 0.8},
        n_gazelles=30,
        n_iterations=80
    )

    print("\n=== TRAINING PHASE (on training set only) ===")
    hybrid_model.train_base_models(
        X_train, y_train,
        X_val_internal, y_val_internal,
        epochs=120
    )

    optimal_weights = hybrid_model.optimize_weights(X_val_internal, y_val_internal)

    print("\n=== TEST PHASE (on unseen test set) ===")
    X_test_scaled = hybrid_model.scaler.transform(X_test)

    ann_pred = hybrid_model.ann_model.predict(X_test_scaled).flatten()
    ridge_pred = hybrid_model.ridge_model.predict(X_test_scaled)
    xgb_pred = hybrid_model.xgb_model.predict(X_test_scaled)
    simple_avg_pred = (ann_pred + ridge_pred + xgb_pred) / 3.0

    ensemble_results