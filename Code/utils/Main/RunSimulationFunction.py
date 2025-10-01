### Import ###
from utils.Main.OneIterationFunction import OneIterationFunction

### Function Definition ###
def RunSimulationFunction(DataFileInput,
                          Seed,
                          machine_learning_model,
                          test_proportion,
                          candidate_proportion):
    """
    Runs a single simulation iteration across multiple data selection strategies.

    Args:
        DataFileInput (str): The file path for the dataset to be loaded.
        Seed (int): Seed
        machine_learning_model (str): A string identifier for the machine learning model to be used.
            Expected values include 'LinearRegressionPredictor', 'RandomForestRegressorPredictor', or 'RidgeRegressionPredictor'.
        test_proportion (float): The fraction of the dataset to be reserved for the final, unseen test set.
        candidate_proportion (float): The fraction of the non-test data to be used as the unlabeled candidate pool.
    Returns:
        dict: A nested dictionary containing the simulation results for the single seed. 
            The keys are the string names of the strategies that were run (e.g., 'Passive Learning', 'iGS'), 
            and the values are the corresponding result objects returned by `OneIterationFunction`.
    """

    ### Set Up ###
    all_results_by_strategy = {}
    strategies_to_run = {
        'Passive Learning': {'SelectorType': 'PassiveLearningSelector'},
        'GSx': {'SelectorType': 'GreedySamplingSelector', 'strategy': 'GSx'},
        'GSy': {'SelectorType': 'GreedySamplingSelector', 'strategy': 'GSy'},
        'iGS': {'SelectorType': 'GreedySamplingSelector', 'strategy': 'iGS'},
        'WiGS (Static w_x=0.25)': {'SelectorType': 'WeightedGreedySamplingSelector',
                                   'weight_strategy': 'static',
                                   'w_x': 0.25},
        'WiGS (Static w_x=0.5)': {'SelectorType': 'WeightedGreedySamplingSelector',
                                  'weight_strategy': 'static',
                                  'w_x': 0.5},
        'WiGS (Static w_x=0.75)': {'SelectorType': 'WeightedGreedySamplingSelector',
                                   'weight_strategy': 'static',
                                   'w_x': 0.75},
        'WiGS (Time-Decay, Linear)': {'SelectorType': 'WeightedGreedySamplingSelector',
                                      'weight_strategy': 'time_decay',
                                      'decay_type': 'linear'},
        'WiGS (Time-Decay, Exponential)': {'SelectorType': 'WeightedGreedySamplingSelector',
                                           'weight_strategy': 'time_decay',
                                           'decay_type': 'exponential',
                                           'decay_constant': 5.0},
        'WiGS (MAB-UCB1, c=0.5)': {'SelectorType': 'WiGS_MAB_Selector', 'mab_c': 0.5},
        'WiGS (MAB-UCB1, c=2.0)': {'SelectorType': 'WiGS_MAB_Selector', 'mab_c': 2.0},
        'WiGS (MAB-UCB1, c=5.0)': {'SelectorType': 'WiGS_MAB_Selector', 'mab_c': 5.0},
        'WiGS (SAC)': {'SelectorType': 'WiGS_SAC_Selector'}
        # 'iRDM': {'SelectorType': 'IRDMSelector'},
        # 'IDEAL': {'SelectorType': 'IDEALSelector', 'delta': 5.0},
        }
    
    ### Loop Through Strategies ###
    for strategy_name, strategy_params in strategies_to_run.items():
        
        ## Base configuration ##
        SimulationConfigInput = {
            'DataFileInput': DataFileInput, 
            'Seed': Seed,
            'TestProportion': test_proportion, 
            'CandidateProportion': candidate_proportion,
            'ModelType': machine_learning_model,
            'regularization': 0.01 
        }
        SimulationConfigInput.update(strategy_params)
        
        # Run simulation #
        results = OneIterationFunction(SimulationConfigInput)

        # Store results #
        all_results_by_strategy[strategy_name] = results

    ### Return the nested dictionary for this single seed ###
    return all_results_by_strategy