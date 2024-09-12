from main import neural_run
from omegaconf import OmegaConf, DictConfig
import ugle
import argparse
from ugle.logger import log
import pickle
from os.path import exists
from os import makedirs
import pickle
import re

def identify_type(s):
    # Check for boolean
    if s.lower() in ('true', 'false'):
        return bool
    
    # Check for float
    try:
        float(s)
        return float
    except ValueError:
        pass

    # Check for integer
    try:
        int(s)
        return int
    except ValueError:
        pass

    # If none of the above, it's a string
    return str

def run_study(study_override_cfg: DictConfig, algorithm: str, dataset: str, seeds: list, check_save):
    """
    Runs a study of one algorithm on one dataset over multiple seeds.

    Args:
        study_override_cfg (DictConfig): study configuration object
        algorithm (str): name of the algorithm to test on 
        dataset (str): name of the dataset to test on 
        seeds (list): list of seeds to test over
    Returns:
        average_results (Dict): the results for each metric averaged over the seeds
    """
    study_cfg = study_override_cfg.copy()
    average_results = ugle.utils.create_study_tracker(len(seeds), study_cfg.trainer.test_metrics)
    if check_save is not None:
        study_results = OmegaConf.create({'dataset': dataset,
                                        'model': algorithm,
                                        'average_results': {},
                                        'results': check_save})
        for seed_res in check_save:
            for res in seed_res['study_output']:
                study_cfg.trainer.hps_found_so_far.append(res['args'])
    else: 
        study_results = OmegaConf.create({'dataset': dataset,
                                        'model': algorithm,
                                        'average_results': {},
                                        'results': []})
    study_cfg.previous_results = None
    # repeat training over all seeds
    for idx, seed in enumerate(seeds):
        study_cfg.args.random_seed = seed
        log.info(f'Study -- {algorithm}:{dataset}:Seed({seed})')

        # test results stores the results of one algorithm run
        if ugle.utils.is_neural(algorithm):
            results = neural_run(override_model=algorithm,
                                 override_dataset=dataset,
                                 override_cfg=study_cfg)

        # save study output
        study_results.results.append({'seed': seed, 'study_output': results})
        average_results = ugle.utils.collate_study_results(average_results, results, idx)

        # use first seed hyperparameters and train/test on remaining
        if idx == 0 and check_save is not None:
            study_cfg.previous_results = results
        
        # add all best hyperparameters if they don't exist to enque trial 
        if study_cfg.trainer.suggest_hps_from_previous_seed:
            for res in results:
                study_cfg.trainer.hps_found_so_far.append(res['args'])

        if study_cfg.trainer.use_hps_on_all_seeds:
            study_cfg.trainer.only_testing = True

        if not exists(study_cfg.trainer.results_path):
            makedirs(study_cfg.trainer.results_path)
        save_path = f"{study_cfg.trainer.results_path}{dataset}_{algorithm}"
        pickle.dump(study_results, open(f"{save_path}.pkl", "wb"))

    # average results stores the average calculation of statistics
    average_results = ugle.utils.calc_average_results(average_results)
    study_results.average_results = average_results

    # save the result of the study
    if not exists(study_cfg.trainer.results_path):
        makedirs(study_cfg.trainer.results_path)
    save_path = f"{study_cfg.trainer.results_path}{dataset}_{algorithm}"
    pickle.dump(study_results, open(f"{save_path}.pkl", "wb"))
        
    return average_results


def run_experiment(exp_cfg_name: str, 
                   dataset_algorithm_override: str=None, 
                   gpu_override: str=None,
                   test_run: bool=False,
                   check_save: bool=False):
    """
    Run experiments which consists of multiple models and datasets

    Args:
        exp_cfg_name: location of the yaml file containing experiment configuration
        dataset_algorithm_override: dataset_algorithm combination which can be used to run a single experiment if datasets
           and algorithms are both empty
    
    """
    # load experiment config
    log.info(f'loading experiment: {exp_cfg_name}')
    exp_cfg = OmegaConf.load('ugle/configs/experiments/exp_cfg_template.yaml')
    exp_cfg = ugle.utils.merge_yaml(exp_cfg, exp_cfg_name)
    if dataset_algorithm_override: 
        exp_cfg.dataset_algo_combinations = [dataset_algorithm_override]
        exp_cfg.datasets = []
        exp_cfg.algorithms = []
    if gpu_override:
        exp_cfg.study_override_cfg.trainer.gpu = gpu_override
    if test_run: 
        if not check_save :
            exp_cfg.seeds = [42, 69]
        exp_cfg.study_override_cfg.args.max_epoch = 1
        exp_cfg.study_override_cfg.trainer.n_trials_hyperopt = 2
        
    
    save_path = exp_cfg.study_override_cfg.trainer.results_path

    if check_save is not None: 
        
        with open(f"./check_save/{check_save}", 'r') as content_file:
            content = content_file.read()

        trial_splits = content.split("Trial 249 finished")[1:-1]
        dset = dataset_algorithm_override.split('_')[0]
        algo = dataset_algorithm_override.split('_')[1]

        loaded_results_dict = {'average_results': {},
                               'dataset': dset,
                               'model': algo,
                               'results': []}

        for sidx, str_content in enumerate(trial_splits):
            seed_dict = {'seed': exp_cfg.seeds[sidx], 'study_output': []}

            for match in re.finditer("Best hyperparameters for metric", str_content):
                start_metrics_idx = match.end() + 5

                end_of_potential_metrics_idx = str_content[start_metrics_idx:].find("\x1b")
                pot_metrics = str_content[start_metrics_idx:start_metrics_idx+end_of_potential_metrics_idx-1]
                pot_metrics = pot_metrics.split(",")
                pot_metrics = [metric[1:] if idx > 0 else metric for idx, metric in enumerate(pot_metrics)]

                for valid_metric in exp_cfg.study_override_cfg.trainer.valid_metrics:
                    if valid_metric in pot_metrics:
                        study_out_dict = {'args': {},
                                          'metrics': valid_metric,
                                          'results': {},
                                          'validation_results': {}}
                        metric_dict = {}
                        for test_metric in exp_cfg.study_override_cfg.trainer.test_metrics:
                            # find point where results start going metric=val
                            start_idx = str_content[start_metrics_idx:].find(f"{test_metric}=")
                            # find the end of the metric result
                            if test_metric != 'conductance':
                                stringtofind = ','
                            else:
                                stringtofind = "\x1b"
                            end_idx = str_content[start_metrics_idx+start_idx:].find(stringtofind)
                            end_idx = end_idx - len(test_metric) - 1
                            startincontent = start_metrics_idx+len(test_metric)+1+start_idx
                            metric_number = float(str_content[startincontent:startincontent+end_idx])
                            metric_dict[test_metric] = metric_number
                        
                        all_args_stuff = str_content[start_metrics_idx+end_of_potential_metrics_idx-1:startincontent+end_idx]
                        args_dict = {}
                        for match in re.finditer(" : ", all_args_stuff):
                            start_arg = all_args_stuff[:match.start()].rfind(']')
                            arg = all_args_stuff[start_arg+2:match.start()]

                            end_value = all_args_stuff[match.start():].find("\x1b")
                            value = all_args_stuff[match.end():match.start() + end_value]

                            # chcek if value is float, bool, int or str
                            if identify_type(value) == float:
                                args_dict[str(arg)] = float(value)

                            elif identify_type(value) == bool:
                                args_dict[str(arg)] = bool(value)
                            
                            elif identify_type(value) == int:
                                args_dict[str(arg)] = int(value)
                            
                            elif identify_type(value) == str:
                                args_dict[str(arg)] = str(value)
                        
                        study_out_dict['args'] = args_dict
                        study_out_dict['results'] = metric_dict

                        seed_dict['study_output'].append(study_out_dict)
            loaded_results_dict['results'].append(seed_dict)

            
    # creating experiment iterator
    experiment_tracker = ugle.utils.create_experiment_tracker(exp_cfg)
    experiments_cpu = []

    if check_save: 
        # change the seeds in exp_cfg
        exp_cfg.seeds = exp_cfg.seeds[sidx+1:]
        check_save = loaded_results_dict['results']
    else:
        check_save = None

    for experiment in experiment_tracker:
        log.debug(f'starting new experiment ... ...')
        log.debug(f'testing dataset: {experiment.dataset}')
        log.debug(f'testing algorithm: {experiment.algorithm}')
        
        try:
            # run experiment
            experiment_results = run_study(exp_cfg.study_override_cfg,
                                            experiment.algorithm,
                                            experiment.dataset,
                                            exp_cfg.seeds,
                                            check_save)
            # save result in experiment tracker
            experiment.results = experiment_results
            ugle.utils.save_experiment_tracker(experiment_tracker, save_path)

        # if breaks then tests on cpu
        except Exception as e:
            log.exception(str(e))
            log.info(f"adding to cpu fallback test")
            experiments_cpu.append(experiment)

        # run all experiments that didn't work on gpu
        if experiments_cpu and exp_cfg.run_cpu_fallback:
            log.info(f'launching cpu fallback experiments')
            exp_cfg.study_override_cfg.trainer.gpu = -1

            for experiment in experiments_cpu:
                log.debug(f'starting new experiment ...')
                log.debug(f'testing dataset: {experiment.dataset}')
                log.debug(f'testing algorithm: {experiment.algorithm}')

                # run experiment
                experiment_results = run_study(exp_cfg.study_override_cfg,
                                                experiment.algorithm,
                                                experiment.dataset,
                                                exp_cfg.seeds)
                # save result in experiment tracker
                experiment.results = experiment_results
                ugle.utils.save_experiment_tracker(experiment_tracker, save_path)
        else:
            if experiments_cpu:
                log.info('The following combinations lead to OOM')
                for experiment in experiments_cpu:
                    log.info(f'{experiment.dataset} : {experiment.algorithm}')
        
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='parsing the experiment to run')
    parser.add_argument('-ec', '--experiment_config', type=str, required=True,
                        help='the location of the experiment config')
    parser.add_argument('-da', '--dataset_algorithm_override', type=str, default=None,
                        help='dataset_algorithm override setting')
    parser.add_argument('--gpu', type=str, default=None)
    parser.add_argument('--test_run', action='store_true')
    parser.add_argument('-cs', type=str, default=None)
    parsed, unknown = parser.parse_known_args()
    run_experiment(parsed.experiment_config, parsed.dataset_algorithm_override, parsed.gpu, parsed.test_run, parsed.cs)