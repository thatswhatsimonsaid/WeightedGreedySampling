### Libraries ###
from sklearn.model_selection import train_test_split

### Function ###
def TrainCandidateSplit(df, CandidateProportion, TestProportion=0.10):
    """
    Splits the original dataframe df into THREE sets: test, training, and candidate sets.

    Args:
        - df: The original dataframe.
        - CandidateProportion: Proportion of the LEARNING data that is "unseen" candidate data.
        - TestProportion: Proportion of the TOTAL data explicitly held out for testing.
    
    Returns:
        - df_Train: The initial training set.
        - df_Candidate: The candidate set that is queried during active learning.
        - df_Test: The held-out test set evaluated at every iteration.
    """

    # 1. Hold-out Test Split FIRST #
    # Slices off 10% (by default) for the true held-out test set
    df_LearningPool, df_Test = train_test_split(df, test_size=TestProportion)

    # 2. Train/Candidate Split on the REMAINING data #
    X_Train, X_Candidate, y_Train, y_Candidate = train_test_split(
        df_LearningPool.loc[:, df_LearningPool.columns != "Y"], 
        df_LearningPool["Y"], 
        test_size=CandidateProportion)
    
    # Column names #
    df_Train = X_Train.copy()
    df_Train.insert(0, 'Y', y_Train)
    
    df_Candidate = X_Candidate.copy()
    df_Candidate.insert(0, 'Y', y_Candidate)

    return df_Train, df_Candidate, df_Test