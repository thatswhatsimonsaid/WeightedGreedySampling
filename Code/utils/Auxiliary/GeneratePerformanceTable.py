
import os
import pickle
import numpy as np
from scipy.stats import wilcoxon

# ================= CONFIGURATION =================
# Active datasets to compare
DATASETS = ["dgp_two_regime", "dgp_three_regime"]
METRIC = "RMSE"
ALPHA = 0.05
MODERATE_THRESHOLD = 1.0

# Robust Path Finder: Finds 'Results' folder automatically
base_path = "Results/simulation_results/aggregated"
possible_paths = [base_path, os.path.join("..", base_path), os.path.join("WeightedGreedySampling", base_path)]
RESULTS_DIR = next((p for p in possible_paths if os.path.exists(p)), None)

if not RESULTS_DIR:
    print("CRITICAL ERROR: Could not find 'Results' directory.")
    exit()

def load_data(dataset):
    path = os.path.join(RESULTS_DIR, dataset, "full_pool_metrics", f"{METRIC}.pkl")
    if not os.path.exists(path): return None
    
    try:
        with open(path, 'rb') as f: return pickle.load(f)
    except: return None

def compare(baseline_data, challenger_data):
    """
    Returns (Category, Improvement %).
    Positive % means Challenger is BETTER (Lower Error) than Baseline.
    """
    if baseline_data is None or challenger_data is None:
        return "N/A", 0.0

    # Collapse to mean over iterations (Result: Array of 20 seeds)
    # Check shape to ensure we average the right axis
    if len(np.array(baseline_data).shape) > 1:
        base_means = np.mean(baseline_data, axis=1)
        chal_means = np.mean(challenger_data, axis=1)
    else:
        base_means = np.array(baseline_data)
        chal_means = np.array(challenger_data)

    # 1. Statistical Significance (Wilcoxon)
    try:
        _, p_val = wilcoxon(base_means, chal_means)
    except ValueError: 
        p_val = 1.0 
        
    # 2. % Improvement
    diff_pct = ((np.mean(base_means) - np.mean(chal_means)) / np.mean(base_means)) * 100
    
    # 3. Categorize
    if p_val < ALPHA:
        if diff_pct > MODERATE_THRESHOLD: category = "Signif. (+)"  # Significantly Better
        elif diff_pct < -MODERATE_THRESHOLD: category = "Signif. (-)" # Significantly Worse
        else: category = "Mixed"
    else:
        category = "Same"
        
    return category, diff_pct

# ================= EXECUTION =================

# Header
print(f"{'DATASET':<20} | {'WiGS vs iGS':<15} | {'QBC vs iGS':<15} | {'WiGS vs QBC':<15}")
print("-" * 75)

for name in DATASETS:
    data = load_data(name)
    
    if data is None:
        print(f"{name:<20} | [File Not Found]")
        continue

    # Extract specific keys
    igs  = data.get('iGS')
    wigs = data.get('WiGS (SAC)')
    qbc  = data.get('QBC')

    # 1. WiGS vs iGS (Baseline)
    w_i_cat, w_i_imp = compare(igs, wigs)
    
    # 2. QBC vs iGS (Baseline)
    q_i_cat, q_i_imp = compare(igs, qbc)

    # 3. WiGS vs QBC (QBC is Baseline)
    # Positive % here means WiGS is better than QBC
    w_q_cat, w_q_imp = compare(qbc, wigs)
    
    # Print Row
    print(f"{name:<20} | "
          f"{w_i_cat:<10} {w_i_imp:>+4.1f}% | "
          f"{q_i_cat:<10} {q_i_imp:>+4.1f}% | "
          f"{w_q_cat:<10} {w_q_imp:>+4.1f}%")