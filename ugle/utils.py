from omegaconf import OmegaConf, open_dict, DictConfig
from optuna import Study
from typing import Tuple
import random
import torch
import numpy as np
from ugle.logger import log, ugle_path
import optuna
from optuna.trial import Trial
import os 
import pickle

neural_algorithms = ['daegc', 'dgi', 'dmon', 'grace', 'mvgrl', 'selfgnn', 'sublime', 'bgrl', 'vgaer', 'cagc']


def load_model_config(override_model: str = None, override_cfg: DictConfig = None) -> DictConfig:
    """
    Loads the base model config, model-specific config, followed by override_cfg.

    Args:
        override_model (str): name of the model to load config for
        override_cfg (DictConfig): Config to override options on the loaded config

    Returns:
        config (DictConfig): Config for running experiments
    """ 

    config = OmegaConf.load(f"{ugle_path}/ugle/configs/config.yaml")

    if override_model:
        config.model = override_model

    config_name = config.model
    model_name = config.model
    
    # model_path structure edit if testing
    if config_name.__contains__('_'):
        model_name, name_ext = config_name.split("_")
        config.model = model_name
        model_path = f"{ugle_path}/ugle/configs/models/{model_name}/{config_name}.yaml"
    else:
        model_path = f"{ugle_path}/ugle/configs/models/{model_name}/{model_name}.yaml"

    config = merge_yaml(config, model_path)

    # model_eval override
    if override_cfg:
        with open_dict(config):
            config = OmegaConf.merge(config, override_cfg)

    return config


def process_study_cfg_parameters(config: DictConfig) -> DictConfig:
    """
    Creates optimization directions for study.

    Args:
        config (DictConfig): Config object

    Returns:
        config (DictConfig): Processed config
    """
    opt_directions = []
    if len(config.trainer.valid_metrics) > 1:
        config.trainer.multi_objective_study = True
    else:
        config.trainer.multi_objective_study = False

    for metric in config.trainer.valid_metrics:
        if metric == 'nmi' or metric == 'f1' or metric == 'modularity':
            opt_directions.append('maximize')
        else:
            opt_directions.append('minimize')

    config.trainer.optimisation_directions = opt_directions
    return config


def extract_best_trials_info(study: Study, valid_metrics: list) -> Tuple[list, list]:
    """
    Extracts the best trial from a study for each given metric and associated trial.

    Args:
        study (Study): The study object
        valid_metrics (list): The validation metrics 

    Returns:
        best_values (list): The best values for each metric
        associated_trial (list): The associated trial with each best value
    """
    best_values = study.best_trials[0].values
    associated_trial = [study.best_trials[0].number] * len(valid_metrics)
    if len(study.best_trials) > 1:
        for a_best_trial in study.best_trials[1:]:
            for i, bval in enumerate(a_best_trial.values):
                if (bval > best_values[i] and study.directions[i].name == 'MAXIMIZE') or (
                        bval < best_values[i] and study.directions[i].name == 'MINIMIZE'):
                    best_values[i] = bval
                    associated_trial[i] = a_best_trial.number
                
    return best_values, associated_trial


def merge_yaml(config: DictConfig, yaml_str: str) -> DictConfig:
    """
    Merges config object with config specified in YAML path string.

    Args:
        config (DictConfig): Config object to be merged
        yaml_str (str): Path location of .yaml string to be merged

    Returns:
        config (DictConfig): Merged config
    """
    yaml_dict = OmegaConf.load(yaml_str)
    with open_dict(config):
        config = OmegaConf.merge(config, yaml_dict)
    return config


def set_random(random_seed: int):
    """
    Sets the random seed.
    
    Args:
        random_seed (int): Random seed to be set
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    return


def set_device(gpu: int):
    """
    Returns the correct device.

    Args:
        gpu (int): GPU index to specify the device
    """
    if gpu != -1 and torch.cuda.is_available():
        device = f'cuda:{gpu}'
    else:
        device = 'cpu'
    return device


def remove_last_line():
    """
    Removes the last line in the terminal.

    Note: This function prints ANSI escape sequences to move the cursor to the beginning of the last line.
    """
    print("\033[2F\033[", flush=True)
    return


def sample_hyperparameters(trial: optuna.trial.Trial, args: DictConfig, prune_params=None) -> DictConfig:
    """
    Iterates through the args configuration. If an item is a list, suggests a value based on
    the optuna trial instance.

    Args:
        trial (optuna.trial.Trial): Instance of trial for suggesting parameters
        args (DictConfig): Config dictionary where list
        prune_params: (optional) Parameters to prune from the sampled configuration

    Returns:
        DictConfig: New config with values replaced where list is given
    """
    vars_to_set = []
    vals_to_set = []
    for k, v in args.items_ex(resolve=False):
        if not OmegaConf.is_list(v):
            continue
        else:
            trial_sample = trial.suggest_categorical(k, OmegaConf.to_object(v))
            setattr(args, k, trial_sample)
            vars_to_set.append(k)
            vals_to_set.append(trial_sample)

    if prune_params:
        repeated = prune_params.check_params()
    for var, val in zip(vars_to_set, vals_to_set):
        log.info(f"args.{var}={val}")

    return args

def assign_test_params(config: DictConfig, best_params: dict) -> DictConfig:
    """
    Assigns the best parameters from the hyperparameter selection and assigns test config settings.

    Args:
        config (DictConfig): Original config for training
        best_params (dict): The best hyperparameters from training

    Returns:
        DictConfig: Config for training and then testing
    """
    cfg = config.copy()
    # assigns the best parameters from hp to config
    for key, value in cfg.args.items_ex(resolve=False):
        if not OmegaConf.is_list(value):
            continue
        else:
            cfg.args[key] = best_params[key]

    # assigns parameters for testing
    cfg.trainer.train_to_valid_split = 1.0
    cfg.trainer.n_valid_splits = 1
    cfg.trainer.currently_testing = True

    return cfg


def is_neural(model_name: str) -> bool:
    """
    Checks if an algorithm is neural or non-neural.

    Args:
        model_name (str): The model name string

    Returns:
        bool: True if neural, False if not
    """
    if model_name.__contains__('_'):
        adjusted_model_name, _ = model_name.split("_")
    else:
        adjusted_model_name = 'this is definitely not a name of an algorithm'

    if model_name in neural_algorithms or adjusted_model_name in neural_algorithms:
        return True
    else:
        raise ValueError(f"Algorithm {model_name} is not implemented")


def collate_study_results(average_results: dict, results: dict, val_idx: int) -> dict:
    """
    Converts study results from results.

    Args:
        average_results (dict): Dictionary of metrics with numpy arrays of results across training runs
        results (dict): Dictionary of results
        val_idx (int): Number of the training run

    Returns:
        average_results (dict): Dictionary of metrics with numpy arrays of results across training runs
    """
    for trial in results:
        for metric, value in trial["results"].items():
            average_results[metric][val_idx] = value

    return average_results


def create_study_tracker(k_splits: int, metrics: list) -> dict:
    """
    Creates a study tracker for multiple seeds.

    Args:
        k_splits (int): The number of seeds
        metrics (list): The metrics to create a tracker for

    Returns:
        results (object): The results tracker
    """
    results = {}
    for metric in metrics:
        results[metric] = np.zeros(k_splits)
    return results


def create_experiment_tracker(exp_cfg: DictConfig) -> list:
    """
    Creates the experiment tracker to track results.

    Args:
        exp_cfg (DictConfig): Experiment config

    Returns:
        experiment_tracker (list): List of results objects that store results from individual experiments
    """
    experiment_tracker = []
    if exp_cfg.dataset_algo_combinations:
        for dataset_algo in exp_cfg.dataset_algo_combinations:
            dataset, algorithm = dataset_algo.split('_', 1)
            experiment_tracker.append(OmegaConf.create(
                                {'dataset': dataset,
                                'algorithm': algorithm,
                                'seeds': exp_cfg.seeds,
                                'results': {}
                                }))
            
    elif exp_cfg.datasets and exp_cfg.algorithms:
        for dataset in exp_cfg.datasets:
            for algorithm in exp_cfg.algorithms:
                experiment_tracker.append(OmegaConf.create(
                                {'dataset': dataset,
                                'algorithm': algorithm,
                                'seeds': exp_cfg.seeds,
                                'results': {}
                                }))

    return experiment_tracker



def extract_result_values(exp_result, metrics=['f1', 'nmi', 'modularity', 'conductance']) -> str:
    """
    Extracts the results into a string ready to print for LaTeX display. If std values exist, then they are also used.

    Args:
        exp_result (dict): Dictionary of results
        metrics (list): List of metrics in results (default: ['f1', 'nmi', 'modularity', 'conductance'])

    Returns:
        metric_string (str): String to print given results
    """
    metric_string = ''
    for metric in metrics:
        metric_string += ' & '
        metric_mean = f'{metric}_mean'
        metric_val = exp_result[metric_mean]
        metric_string += str(metric_val)

        metric_std = f'{metric}_std'
        if metric_std in exp_result.keys():
            std = str(exp_result[metric_std])
            metric_string += f'\pm {std}'

    metric_string += ' \\\\'
    return metric_string


def save_experiment_tracker(result_tracker: list, results_path: str):
    """
    Pickle saves result tracker to results path.

    Args:
        result_tracker: List of experiment results
        results_path: The path to save results in
    """
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    experiment_identifier = log.name
    log.info(f'saving to {results_path}{experiment_identifier}')
    pickle.dump(result_tracker, open(f"{results_path}{experiment_identifier}.pkl", "wb"))

    return


def calc_average_results(results_tracker: dict) -> dict:
    """
    Calculates the average of the performance statistic to an appropriate significant figure.

    Args:
        results_tracker (dict): Contains results over seeds

    Returns:
        average_results (dict): The average of results over all seeds
    """
    average_results = {}
    for stat, values in results_tracker.items():
        if stat != 'memory':
            average_results[f'{stat}_mean'] = float(np.format_float_positional(np.mean(values, axis=0),
                                                                               precision=4,
                                                                               unique=False,
                                                                               fractional=False))
            if len(values) > 1:
                average_results[f'{stat}_std'] = float(np.format_float_positional(np.std(values, axis=0),
                                                                                  precision=2,
                                                                                  unique=False,
                                                                                  fractional=False))
        else:
            average_results[f'memory_max'] = float(np.format_float_positional(np.max(values, axis=0),
                                                                              precision=7,
                                                                              unique=False,
                                                                              fractional=False))
    return average_results