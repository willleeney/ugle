from omegaconf import OmegaConf, open_dict, DictConfig
from optuna import Study
from typing import Tuple
import random
import torch
import numpy as np
import optuna
import os
import pickle
from ugle.logger import log

neural_algorithms = ['daegc', 'dgi', 'dmon', 'grace', 'mvgrl', 'selfgnn', 'sublime', 'bgrl', 'vgaer', 'cagc', 'igo']


def load_model_config(override_model: str = None, override_cfg: DictConfig = None) -> DictConfig:
    """
    loads the base model config, model specific config, then override_cfg, then the umap processing
    investigation config options if relevant
    :param override_model: name of the model to load config for
    :param override_cfg: config to override options on loaded config
    :return config: config for running experiments
    """
    config = OmegaConf.load('ugle/configs/config.yaml')

    if override_model:
        config.model = override_model

    config_name = config.model
    model_name = config.model

    # model_path structure edit if testing
    if config_name.__contains__('_'):
        model_name, name_ext = config_name.split("_")
        config.model = model_name
        model_path = f'ugle/configs/models/{model_name}/{config_name}.yaml'
    else:
        model_path = f'ugle/configs/models/{model_name}/{model_name}.yaml'

    config = merge_yaml(config, model_path)

    # model_eval override
    if override_cfg:
        with open_dict(config):
            config = OmegaConf.merge(config, override_cfg)

    return config


def process_study_cfg_parameters(config: DictConfig) -> DictConfig:
    """
    creates optimisation directions for study
    :param config: config object
    :return config: process config
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
    extracts the best trial from a study for each given metric and associated trial
    :param study: the study object
    :param valid_metrics: the validation metrics 
    :return best_values: the best values for each metric
    :return associated_trail: the associated trial with each best value
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
    merges config object with config specified in yaml path str
    :param config: config object to be merged
    :param yaml_str: path location of .yaml string to be merged
    :return config: merged config
    """
    yaml_dict = OmegaConf.load(yaml_str)
    with open_dict(config):
        config = OmegaConf.merge(config, yaml_dict)
    return config


def set_random(random_seed: int):
    """
    sets the random seed
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    return


def set_device(gpu: int):
    """
    returns the correct device
    """
    if gpu != -1 and torch.cuda.is_available():
        device = f'cuda:{gpu}'
    else:
        device = 'cpu'
    return device


def remove_last_line():
    print(f"\033[2F\033[", end='\n')
    return


def sample_hyperparameters(trial: optuna.trial.Trial, args: DictConfig, prune_params=None) -> DictConfig:
    """
    iterates through the args configuration, if an item is a list then suggests a value based on
    the optuna trial instance
    :param trial: instance of trial for suggestion parameters
    :param args: config dictionary where list
    :return: new config with values replaced where list is given
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
    assigns the best params from the hyperparameter selection and assigns test config settings
    :param config: original config for training
    :param best_params: the best hyperparameters from training
    :return: config for training then testing
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
    checks if an algorithm is neural or non_neural
    :param model_name: the model name string
    :return True if neural, False if not
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
    converts study results from results
    :param average_results: dict of metrics with numpy arrays of results across training runs
    :param results: dictionary of results
    :param val_idx: no. training run
    :return:
    """
    for trial in results:
        for metric, value in trial["results"].items():
            average_results[metric][val_idx] = value

    return average_results


def create_study_tracker(k_splits: int, metrics: list) -> dict:
    """
    creates a study tracker for multiple seeds
    :param k_splits: the number of seeds
    :param metrics: the metrics to create a tracker for
    :return results: the results tracker
    """
    results = {}
    for metric in metrics:
        results[metric] = np.zeros(k_splits)
    return results


def create_experiment_tracker(exp_cfg: DictConfig) -> list:
    """
    creates the experiment tracker to track results
    :param exp_cfg: experiment config
    :experiment_tracker: list of results objects that store results from individual experiment
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


def display_saved_results(results_path: str = "./results"):
    """
    displays results saved in memory under given directory in latex form
    :param results_path: path where results are located
    """
    for subdir, dirs, files in os.walk(results_path):
        for file in files:
            log.info(f"{file}")
            exp_result_path = os.path.join(subdir, file)
            exp_result = pickle.load(open(exp_result_path, "rb"))
            for experiment in exp_result:
                if hasattr(experiment, 'algorithm'):
                    log.info(f"{experiment.dataset}")
                    try:
                        display_latex_results(experiment.algorithm_identifier, experiment.results)
                    except:
                        print(experiment.results)

    return


def display_evaluation_results(experiment_tracker: list):
    """
    displays the evaluation results in table latex form
    :param experiment_tracker: list of results from experiment
    """
    for experiment in experiment_tracker:
        try: 
            display_latex_results(experiment.algorithm, experiment.results, experiment.dataset)
        except:
            pass

    return


def display_latex_results(method: str, exp_result: dict, dataset):
    """
    prints out latex results of experiment
    :param method: method string
    :param exp_result: dictionary of results
    """
    display_str = f'{method} {dataset}'

    if not is_neural(method):
        n_clusters = int(exp_result['n_clusters_mean'])
        display_str += f' ({n_clusters}) '

    if 'memory_max' in exp_result.keys():
        memory = float(exp_result['memory_max'])
        display_str += f' ({memory}) '

    metric_string = extract_result_values(exp_result)
    display_str += metric_string
    print(display_str)

    return


def extract_result_values(exp_result, metrics=['f1', 'nmi', 'modularity', 'conductance']) -> str:
    """
    extracts the results into a strjing ready to print for latex display
    if std values exist then they are also used
    :param exp_result: dictionary of results
    :param metrics: list of metrics in results
    :return metric_string: string to print given results
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
    pickle saves result tracker to results path
    :param result_tracker: list of experiment results
    :param results_path: the path to save results in
    """
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    experiment_identifier = log.name
    log.info(f'saving to {results_path}{experiment_identifier}')
    pickle.dump(result_tracker, open(f"{results_path}{experiment_identifier}.pkl", "wb"))

    return


def calc_average_results(results_tracker: dict) -> dict:
    """
    calculates the average of the performance statistic to an appropriate s.f.
    :param results_tracker: contains results of over seeds
    :return average_results: the average of results over all seeds
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



def get_best_args(results, model_resolution_metric):
    best = -1.
    for seed_res in results: 
        for metric_res in seed_res['study_output']:
            if model_resolution_metric in metric_res['metrics'] and best < metric_res['results'][model_resolution_metric]: 
                best = metric_res['results'][model_resolution_metric]
                best_seed = seed_res['seed']
                args = metric_res['args']
    return args, best_seed
