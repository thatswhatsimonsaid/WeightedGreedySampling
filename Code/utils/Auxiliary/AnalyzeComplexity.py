
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
# ================= CONFIGURATION =================
DATA_DIR = r"C:\vscode\WeightedGreedySampling\Data\processed"

# List of datasets
DATASETS = [
    "beer", "bodyfat", "burbidge_correct", "burbidge_low_noise", 
    "burbidge_misspecified", "concrete_4", "concrete_cs", "concrete_flow", 
    "concrete_slump", "cps_wage", "housing", "mpg", "no2", "pm10", 
    "qsar", "wine_red", "wine_white", "yacht"
]

def load_data_from_df(path):
    """Loads a dataframe pickle and splits it into X (features) and y (target)."""
    try:
        with open(path, 'rb') as f:
            df = pickle.load(f)
        
        if not isinstance(df, pd.DataFrame):
            return None, None
            
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        
        return X, y
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None, None

print(f"{'DATASET':<25} | {'Feats':<5} | {'R2 (Lin)':<8} | {'R2 (RF)':<8} | {'GAP':<8} | {'COMPLEXITY CLASS'}")
print("-" * 105)

for name in DATASETS:
    # Construct path
    path = os.path.join(DATA_DIR, f"{name}.pkl")
    
    if not os.path.exists(path):
        print(f"{name:<25} | FILE NOT FOUND")
        continue

    X, y = load_data_from_df(path)
    if X is None:
        print(f"{name:<25} |  Not a DataFrame?")
        continue

    # Preprocessing
    scaler = StandardScaler()
    try:
        X_scaled = scaler.fit_transform(X)
        
        # 1. Linear Baseline
        #    Ridge instead of pure LinearRegression to handle multicollinearity 
        #    better
        lin_model = Ridge(alpha=1.0)
        lin_scores = cross_val_score(lin_model, X_scaled, y, cv=5, scoring='r2')
        # Clip negative R2 to 0.0 for cleaner table reading
        r2_lin = np.maximum(np.mean(lin_scores), 0)

        # 2. Non-Linear Baseline (Random Forest)
        #    Depth=10 allows capturing interactions without massive overfitting
        rf_model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
        rf_scores = cross_val_score(rf_model, X_scaled, y, cv=5, scoring='r2')
        r2_rf = np.maximum(np.mean(rf_scores), 0)
        
        # 3. The Complexity Gap
        gap = r2_rf - r2_lin
        
        # 4. Categorize Formulaically
        #    If Gap > 0.10, the linear model misses massive signal -> High Complexity
        #    If Gap < 0.03, the linear model is nearly perfect -> Low Complexity
        if gap > 0.10:
            category = "High (Non-Linear)"
        elif gap > 0.03:
            category = "Moderate"
        else:
            category = "Low (Linear)"

        print(f"{name:<25} | {X.shape[1]:<5} | {r2_lin:.3f}    | {r2_rf:.3f}    | {gap:.3f}    | {category}")

    except Exception as e:
        print(f"{name:<25} | ERROR: {e}")