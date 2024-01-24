from main import neural_run
from omegaconf import OmegaConf, DictConfig
import ugle
import argparse
from ugle.logger import log
import pickle
from os.path import exists
from os import makedirs

def run_study(study_override_cfg: DictConfig, algorithm: str, dataset: str, seeds: list):
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
    study_results = OmegaConf.create({'dataset': dataset,
                                      'model': algorithm,
                                      'average_results': {},
                                      'results': []})
    if 'default' not in algorithm:
        assert not study_cfg.trainer.only_testing, "Attempting hyperparameter analyis config with only_testing param enabled"

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
        if idx == 0:
            study_cfg.previous_results = results
        
        # add all best hyperparameters if they don't exist to enque trial 
        if study_cfg.trainer.suggest_hps_from_previous_seed:
            for res in results:
                study_cfg.trainer.hps_found_so_far.append(res['args'])

        elif study_cfg.trainer.use_hps_on_all_seeds:
            study_cfg.trainer.only_testing = True
            study_cfg.trainer.load_existing_test: True

    # average results stores the average calculation of statistics
    average_results = ugle.utils.calc_average_results(average_results)
    study_results.average_results = average_results

    # save the result of the study
    if not exists(study_cfg.trainer.results_path):
        makedirs(study_cfg.trainer.results_path)
    save_path = f"{study_cfg.trainer.results_path}{dataset}_{algorithm}"
    pickle.dump(study_results, open(f"{save_path}.pkl", "wb"))
        
    return average_results


def run_experiment(exp_cfg_name: str, dataset_algorithm_override: str):
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
    

    save_path = exp_cfg.study_override_cfg.trainer.results_path

    # creating experiment iterator
    experiment_tracker = ugle.utils.create_experiment_tracker(exp_cfg)
    experiments_cpu = []

    for experiment in experiment_tracker:
        log.debug(f'starting new experiment ... ...')
        log.debug(f'testing dataset: {experiment.dataset}')
        log.debug(f'testing algorithm: {experiment.algorithm}')
        
        try:
            # run experiment
            experiment_results = run_study(exp_cfg.study_override_cfg,
                                            experiment.algorithm,
                                            experiment.dataset,
                                            exp_cfg.seeds)
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
    parsed = parser.parse_args()
    run_experiment(parsed.experiment_config, parsed.dataset_algorithm_override)
