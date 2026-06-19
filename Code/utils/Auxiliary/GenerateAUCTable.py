import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
import traceback

### CONFIGURATION ###
METRIC = 'RMSE'  # Options: 'RMSE', 'MAE', 'R2', 'CC'
TARGET_BASELINES = [
    'Passive Learning', 
    'iGS', 
    'QBC', 
    'Uncertainty Sampling', 
    'EGAL', 
    'EMCM', 
    #'WiGS (SAC)', 
    None
]

OUTPUT_FILENAME_BASE = 'AUC_Performance_Heatmap'

def load_and_calculate_auc_from_dirs(data_dir):
    auc_records = []
    
    try:
        dataset_folders = sorted([
            f for f in os.listdir(data_dir) 
            if os.path.isdir(os.path.join(data_dir, f))
        ])
    except FileNotFoundError:
        print(f"Error: Data directory not found at {data_dir}")
        sys.exit(1)

    print(f"Found {len(dataset_folders)} dataset folders. Processing...")

    for dataset_name in dataset_folders:
        if dataset_name == 'dgp_new':
            print(f"  [Excluded] Skipping {dataset_name} per user configuration.")
            continue
            
        metric_path = os.path.join(data_dir, dataset_name, 'full_pool_metrics', f'{METRIC}.pkl')
        
        if not os.path.exists(metric_path):
            print(f"  [Skipping] {dataset_name}: {METRIC}.pkl not found.")
            continue
            
        try:
            data = pd.read_pickle(metric_path)
            if not isinstance(data, dict):
                 continue

            selector_means = {}
            for selector, val in data.items():
                if isinstance(val, pd.DataFrame):
                    safe_df = pd.DataFrame()
                    for col in val.columns:
                        safe_df[col] = pd.to_numeric(val[col], errors='coerce')
                        
                    if any(isinstance(idx, str) and 'Sim_' in idx for idx in val.index):
                        safe_df = safe_df.T
                        
                    safe_df = safe_df.dropna(axis=1, how='all').dropna(axis=0, how='all')

                    if 'mean' in safe_df.columns:
                        selector_means[selector] = safe_df['mean'].values
                    elif 'Mean' in safe_df.columns:
                        selector_means[selector] = safe_df['Mean'].values
                    else:
                        selector_means[selector] = safe_df.mean(axis=1, skipna=True).values
                        
                elif isinstance(val, pd.Series):
                    selector_means[selector] = pd.to_numeric(val, errors='coerce').values
            
            if not selector_means:
                continue
                
            df = pd.DataFrame(selector_means)
            df.index = np.arange(len(df))
            x = df.index.values
            
            for selector in df.columns:
                y = df[selector].values
                if np.isnan(y).any():
                    y = pd.Series(y).interpolate().bfill().ffill().values
                
                if hasattr(np, 'trapezoid'):
                     auc = np.trapezoid(y, x)
                else:
                     auc = np.trapz(y, x)
                
                auc_records.append({
                    'Dataset': dataset_name,
                    'Selector': selector,
                    'AUC': auc
                })
                
        except Exception as e:
            print(f"  [Error] {dataset_name}: Failed to calculate AUC.")
            traceback.print_exc()

    return pd.DataFrame(auc_records)

def generate_heatmap(auc_df, output_dir, baseline_method):
    if auc_df.empty:
        print("Error: No AUC data calculated.")
        return

    pivot_df = auc_df.pivot(index='Dataset', columns='Selector', values='AUC')
    plt.figure(figsize=(24, 12))     
    
    if baseline_method is None:
        print(f"\n--- Generating Heatmap (Absolute AUC) ---")
        plot_data = pivot_df.T
        plot_data = plot_data.sort_index(axis=1) 
        cmap = "viridis_r" 
        
        sns.heatmap(plot_data, annot=True, fmt=".1f", cmap=cmap, linewidths=.5,
                    cbar_kws={'label': f'Absolute Total AUC ({METRIC})', 'pad': 0.01, 'shrink': 0.8})
        
        filename = f"{OUTPUT_FILENAME_BASE}_Absolute.png"
    else:
        print(f"\n--- Generating Heatmap vs {baseline_method} ---")
        if baseline_method not in pivot_df.columns:
            return

        ratio_df = pivot_df.div(pivot_df[baseline_method], axis=0)        
        plot_data = ratio_df.T
        plot_data = plot_data.sort_index(axis=1)
        
        selectors = plot_data.index.tolist()
        if baseline_method in selectors:
            selectors.remove(baseline_method)
            selectors.sort()
            new_order = [baseline_method] + selectors
            plot_data = plot_data.reindex(new_order)

        cmap = sns.diverging_palette(240, 10, as_cmap=True, center='light')
        sns.heatmap(plot_data, annot=True, fmt=".3f", cmap=cmap, center=1.0, 
                    vmin=0.90, vmax=1.10, linewidths=.5,
                    cbar_kws={'label': f'Relative AUC of {METRIC} vs {baseline_method}'})

        safe_baseline = baseline_method.replace(" ", "_").replace("(", "").replace(")", "")
        filename = f"{OUTPUT_FILENAME_BASE}_vs_{safe_baseline}.png"

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    save_path_png = os.path.join(output_dir, filename)
    plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved to: {save_path_png}")

def main():
    try:
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
    except NameError:
        PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))
        
    DATA_DIR = os.path.join(PROJECT_ROOT, 'Results', 'test_split_run', 'simulation_results', 'aggregated')
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'Results', 'test_split_run', 'images', 'manuscript', 'AUC_Tables')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Scanning {DATA_DIR}...")
    
    auc_df = load_and_calculate_auc_from_dirs(DATA_DIR)   

    # This filters out any row where the 'Selector' name contains 'SAC'
    auc_df = auc_df[~auc_df['Selector'].str.contains('SAC', na=False)]
     
    for baseline in TARGET_BASELINES:
        generate_heatmap(auc_df, OUTPUT_DIR, baseline)
    
    print("\nAll heatmaps generated.")

if __name__ == "__main__":
    main()