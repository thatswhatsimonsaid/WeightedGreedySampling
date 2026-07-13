### Import libraries ###
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from utils.Auxiliary.DataFrameUtils import get_features_and_target

### Function ###
def TestSetErrorFunction(InputModel, df_Test: pd.DataFrame) -> dict:
    """
    Calculates performance metrics on the held-out test set.

    The model (trained on the current training set) predicts on the test
    features, and those predictions are scored against the test set's true
    labels. Unlike the hybrid full-pool method, this is a straight predict-and-score
    on data the model never saw during training or candidate selection.

    Args:
        InputModel (object): A trained model object with a .predict() method.
        df_Test (pd.DataFrame): The held-out test dataset.

    Returns:
        dict: A dictionary containing the calculated metrics: 'RMSE', 'MAE', 'R2', and 'CC'.
    """
    # 1. Features and true labels from the test set.
    X_test, y_true = get_features_and_target(df_Test, "Y")

    # 2. Predict on the test features.
    y_pred = InputModel.predict(X_test)

    # 3. Calculate metrics.
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Handle the zero-variance edge case for the correlation coefficient.
    if np.std(y_pred) > 0 and np.std(y_true) > 0:
        cc = np.corrcoef(y_true, y_pred)[0, 1]
    else:
        cc = 1.0

    return {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'CC': cc}
