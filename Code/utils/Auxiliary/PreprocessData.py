### Import Packages ###
import os
import sys
import pickle
import kagglehub
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

### My DGP ###
### Import Packages ###
import numpy as np
import pandas as pd

def generate_two_regime_data(n_samples=1000, seed=None):
    """
    Generates a synthetic dataset with two distinct regimes to test the
    adaptability of active learning strategies.

    - Regime 1 (x < 0.5): A complex sine wave with low noise.
      Requires EXPLORATION to learn its shape.
    - Regime 2 (x >= 0.5): A simple linear function with a small region
      of very high noise. Requires EXPLOITATION to reduce uncertainty.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate uniformly distributed inputs
    x = np.random.uniform(low=0, high=1, size=n_samples)
    
    # Initialize the target and noise vectors
    y = np.zeros(n_samples)
    noise = np.zeros(n_samples)
    
    # --- Define the two regimes ---
    
    # Regime 1: Exploration is key
    exploration_mask = x < 0.5
    y[exploration_mask] = np.sin(x[exploration_mask] * 10 * np.pi)
    noise[exploration_mask] = np.random.normal(0, 0.1, size=np.sum(exploration_mask))
    
    # Regime 2: Exploitation is key
    exploitation_mask = x >= 0.5
    y[exploitation_mask] = 2 * x[exploitation_mask] - 1
    noise[exploitation_mask] = np.random.normal(0, 0.1, size=np.sum(exploitation_mask))
    
    # Add a "trap" of high noise in a small part of the exploitation regime
    noise_trap_mask = (x > 0.8) & (x < 0.9)
    noise[noise_trap_mask] = np.random.normal(0, 1.0, size=np.sum(noise_trap_mask))
    
    # Combine the function and the noise
    final_y = y + noise
    
    # Create and return the DataFrame
    df = pd.DataFrame({'X1': x, 'Y': final_y})
    return df

### Burbridge DGP ###
def GenerateBurbridgeData(n_samples=500, delta=0.0, sigma_epsilon=0.3, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    ### Generate inputs from N(\mu=0.2, \sd^2=0.4^2) ###
    x = np.random.normal(loc=0.2, scale=0.4, size=n_samples)
    
    ### Calculate z and r(x) ###
    z = (x - 0.2) / 0.4
    r_x = (z**3 - 3 * z) / np.sqrt(6)
    
    ### Calculate the deterministic part of the function ###
    f_x = 1 - x + x**2 + delta * r_x
    
    ### Add noise ###
    epsilon = np.random.normal(loc=0, scale=sigma_epsilon, size=n_samples)
    y = f_x + epsilon
    
    ### Create and return the final DataFrame ###
    df = pd.DataFrame({'X1': x, 'Y': y})
    return df

### Helper Function for Preprocessing ###
def _preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Helper function to apply the full preprocessing pipeline to a dataframe.
    """
    if 'Y' not in df.columns:
        raise ValueError("DataFrame must have a target column named 'Y'.")
    
    X = df.drop(columns=['Y'])
    y = df['Y']
    
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = X[col].astype('category')

    # UPDATED: Added drop_first=True to handle binary variables correctly
    X_encoded = pd.get_dummies(X, drop_first=True)
    
    # Standardize all features
    scaler = StandardScaler()
    X_scaled_array = scaler.fit_transform(X_encoded)
    
    # Convert scaled features back to a DataFrame
    X_scaled = pd.DataFrame(X_scaled_array, columns=X_encoded.columns, index=X_encoded.index)
    
    # Recombine and return
    return pd.concat([y, X_scaled], axis=1)

### Main Preprocessing and Saving Function ###
def preprocess_and_save_all():
    """
    Loads all datasets, applies the full preprocessing pipeline, and saves them.
    """
    print("--- Starting Data Preprocessing")
    
    # --- Setup Paths ---
    try:
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
    except NameError:
        PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))
        
    save_path = os.path.join(PROJECT_ROOT, 'Data', 'processed')
    os.makedirs(save_path, exist_ok=True)
    
    datasets_to_save = {}

    print("Loading and processing all datasets...")

    # --- Load, Process, and Collect All Datasets ---
    # Each dataset is loaded raw, then passed to the helper function
    
    # 1. Concrete Compressive Strength
    concrete_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls'
    df_raw = pd.read_excel(concrete_url).rename(columns={'Concrete compressive strength(MPa, megapascals) ': 'Y'})
    datasets_to_save['concrete_4'] = _preprocess_dataframe(df_raw)
    print("  > Processed: concrete_4")

    # 2 - 4. Concrete Slump (CS, Flow, Slump)
    slump_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/slump/slump_test.data'
    df_slump_base = pd.read_csv(slump_url)
    df_cs_raw = df_slump_base.drop(columns=['FLOW(cm)', 'SLUMP(cm)']).rename(columns={'Compressive Strength (28-day)(Mpa)': 'Y'})
    df_flow_raw = df_slump_base.drop(columns=['Compressive Strength (28-day)(Mpa)', 'SLUMP(cm)']).rename(columns={'FLOW(cm)': 'Y'})
    df_slump_raw = df_slump_base.drop(columns=['Compressive Strength (28-day)(Mpa)', 'FLOW(cm)']).rename(columns={'SLUMP(cm)': 'Y'})
    df_cs_raw = df_cs_raw.drop('No', axis=1)
    df_flow_raw = df_flow_raw.drop('No', axis=1)
    df_slump_raw = df_slump_raw.drop('No', axis=1)
    datasets_to_save['concrete_cs'] = _preprocess_dataframe(df_cs_raw)
    print("  > Processed: concrete_cs")
    datasets_to_save['concrete_flow'] = _preprocess_dataframe(df_flow_raw)
    print("  > Processed: concrete_flow")
    datasets_to_save['concrete_slump'] = _preprocess_dataframe(df_slump_raw)
    print("  > Processed: concrete_slump")

    # 5. Yacht Hydrodynamics
    yacht_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data'
    yacht_columns = ['longitudinal_pos', 'prismatic_coeff', 'length_displacement_ratio', 'beam_draught_ratio', 'length_beam_ratio', 'froude_number', 'Y']
    df_raw = pd.read_csv(yacht_url, sep=r'\s+', header=None, names=yacht_columns)
    datasets_to_save['yacht'] = _preprocess_dataframe(df_raw)
    print("  > Processed: yacht")

    # 6. Housing
    housing_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data'
    housing_columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    df_raw = pd.read_csv(housing_url, sep=r'\s+', header=None, names=housing_columns)
    df_raw = df_raw.rename(columns={'MEDV': 'Y'})
    datasets_to_save['housing'] = _preprocess_dataframe(df_raw)
    print("  > Processed: housing")
    
    # 7. Auto MPG
    mpg_column_names = ['Y', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin', 'car_name']
    mpg_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
    df_raw = pd.read_csv(mpg_url, sep=r'\s+', header=None, names=mpg_column_names, na_values='?')
    del df_raw['car_name']
    df_raw.dropna(inplace=True)
    df_raw['origin'] = df_raw['origin'].astype('category')
    datasets_to_save['mpg'] = _preprocess_dataframe(df_raw)
    print("  > Processed: mpg")

    # 8 - 9. Wine (Red and White)
    url_red = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
    df_red_raw = pd.read_csv(url_red, sep=';').rename(columns={'quality': 'Y'})
    datasets_to_save['wine_red'] = _preprocess_dataframe(df_red_raw)
    print("  > Processed: wine_red")
    url_white = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'
    df_white_raw = pd.read_csv(url_white, sep=';').rename(columns={'quality': 'Y'})
    datasets_to_save['wine_white'] = _preprocess_dataframe(df_white_raw)
    print("  > Processed: wine_white")

    # # 10 - 11. pmlb datasets (NO2 and PM10)
    # df_pm10_raw = fetch_data('522_pm10', return_X_y=False).rename(columns={'target': 'Y'})
    # datasets_to_save['pm10'] = _preprocess_dataframe(df_pm10_raw)
    # print("  > Processed: pm10")
    # df_no2_raw = fetch_data('547_no2', return_X_y=False).rename(columns={'target': 'Y'})
    # datasets_to_save['no2'] = _preprocess_dataframe(df_no2_raw)
    # print("  > Processed: no2")

    # 12. QSAR Aquatic Toxicity
    qsar_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00505/qsar_aquatic_toxicity.csv'
    df_raw = pd.read_csv(qsar_url, sep=';', header=None)
    df_raw.columns = ['TPSA', 'SAacc', 'H050', 'MLOGP', 'RDCHI','GATS1p', 'nN', 'C040', 'Y']
    datasets_to_save['qsar'] = _preprocess_dataframe(df_raw)
    print("  > Processed: qsar")

    # Kaggle datasets
    try:

        # 13. Body Fat
        download_path_bf = kagglehub.dataset_download("fedesoriano/body-fat-prediction-dataset")
        df_raw = pd.read_csv(os.path.join(download_path_bf, 'bodyfat.csv'))
        if 'Density' in df_raw.columns: del df_raw['Density']
        df_raw = df_raw.rename(columns={'BodyFat': 'Y'})
        datasets_to_save['bodyfat'] = _preprocess_dataframe(df_raw)
        print("  > Processed: bodyfat")

        # 14. Beer
        download_path_beer = kagglehub.dataset_download("dongeorge/beer-consumption-sao-paulo")
        df_raw = pd.read_csv(os.path.join(download_path_beer, 'Consumo_cerveja.csv'), decimal=',')
        df_raw.dropna(inplace=True) 
        df_raw.columns = ['Date', 'Temp_Avg_C', 'Temp_Min_C', 'Temp_Max_C', 'Precipitation_mm', 'Weekend', 'Y'] 
        del df_raw['Date']
        for col in df_raw.columns:
            df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')
        df_raw.dropna(inplace=True)
        datasets_to_save['beer'] = _preprocess_dataframe(df_raw)
        print("  > Processed: beer")

        # 15. CPS
        download_path_cps = kagglehub.dataset_download("avikdas2021/determinants-of-wages-data-cps-1985")
        csv_file = [f for f in os.listdir(download_path_cps) if f.endswith('.csv')][0]
        df_raw = pd.read_csv(os.path.join(download_path_cps, csv_file))

        # Rename wage to Y (target variable)
        df_raw = df_raw.rename(columns={'wage': 'Y'})

        # Drop the unnecessary index column
        if "Unnamed: 0" in df_raw.columns:
            df_raw = df_raw.drop("Unnamed: 0", axis=1)

        # Explicitly define which columns are categorical by changing their type
        categorical_cols = [
            'ethnicity', 'region', 'occupation', 
            'sector', 'union', 'married'
        ]
        for col in categorical_cols:
            if col in df_raw.columns:
                df_raw[col] = df_raw[col].astype('category')

        # The helper function will now correctly handle these
        datasets_to_save['cps_wage'] = _preprocess_dataframe(df_raw)
        print("  > Processed: cps_wage")

    except Exception as e:
        print(f"\n--- KAGGLE ERROR: {e} ---")

    # 16 - 18. Burbridge Dataset
    df_dgp_correct = GenerateBurbridgeData(delta=0.0, sigma_epsilon=0.3, seed=42)
    datasets_to_save['dgp_correct'] = _preprocess_dataframe(df_dgp_correct)
    print("  > Processed: dgp_correct")
    df_dgp_misspecified = GenerateBurbridgeData(delta=0.05, sigma_epsilon=0.3, seed=42)
    datasets_to_save['dgp_misspecified'] = _preprocess_dataframe(df_dgp_misspecified)
    print("  > Processed: dgp_misspecified")
    df_dgp_low_noise = GenerateBurbridgeData(delta=0.05, sigma_epsilon=0.1, seed=42)
    datasets_to_save['dgp_low_noise'] = _preprocess_dataframe(df_dgp_low_noise)
    print("  > Processed: dgp_low_noise")

    # 19. Two-Regime Dataset
    df_dgp_two_regime = generate_two_regime_data(seed=42)
    datasets_to_save['dgp_two_regime'] = _preprocess_dataframe(df_dgp_two_regime)
    print("  > Processed: dgp_two_regime")

    # --- Save all successfully processed datasets ---
    print("\nSaving all processed datasets...")
    for name, dataframe in datasets_to_save.items():
        file_path = os.path.join(save_path, f"{name}.pkl")
        with open(file_path, 'wb') as file:
            pickle.dump(dataframe, file)
        
    print("\n--- Data Preprocessing Complete ---")
    return(datasets_to_save)

if __name__ == "__main__":
    preprocess_and_save_all()