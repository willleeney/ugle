from main import neural_run
from omegaconf import OmegaConf, DictConfig
import ugle
import argparse
from ugle.logger import log
import pickle
from os.path import exists
from os import makedirs
from copy import deepcopy


def run_study(study_override_cfg: DictConfig, algorithm: str, dataset: str, seeds: list):
    """
    runs a study of one algorithm on one dataset over multiple seeds
    :param study_override_cfg: study configuration object
    :param algorithm: string of algorithm to test on
    :param dataset: string of the dataset to test on
    :param seeds: list of seeds to test over
    :return average_results: the results for each metric averaged over the seeds
    """
    study_cfg = study_override_cfg.copy()
    average_results = ugle.utils.create_study_tracker(len(seeds), study_cfg.trainer.test_metrics)
    study_results = OmegaConf.create({'dataset': dataset,
                                      'model': algorithm,
                                      'average_results': {},
                                      'results': []})

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
            study_cfg.trainer.only_testing = True
            study_cfg.previous_results = results

    # average results stores the average calculation of statistics
    average_results = ugle.utils.calc_average_results(average_results)
    study_results.average_results = average_results

    # save the result of the study
    if not exists(study_cfg.trainer.results_path):
        makedirs(study_cfg.trainer.results_path)
    save_path = f"{study_cfg.trainer.results_path}{dataset}_{algorithm}"
    pickle.dump(study_results, open(f"{save_path}.pkl", "wb"))

    return average_results


def run_experiment(exp_cfg_name: str):
    """
    run experiments which consists of multiple models and datasets
    :param exp_cfg_name: location of the yaml file containing experiment configuration
    """
    # load experiment config
    log.info(f'loading experiment: {exp_cfg_name}')
    exp_cfg = OmegaConf.load('ugle/configs/experiments/exp_cfg_template.yaml')
    exp_cfg = ugle.utils.merge_yaml(exp_cfg, exp_cfg_name)

    # iterate
    if exp_cfg.special_training.split_addition_percentage:
        log.info('Special Experiment: split_addition_percentage')
        saved_cfg = deepcopy(exp_cfg)
        special_runs = deepcopy(exp_cfg.study_override_cfg.trainer.split_addition_percentage)
    elif exp_cfg.special_training.retrain_on_each_dataset: 
        log.info('Special Experiment: retrain_on_each_dataset')
        # figure out save model location 
        # overwrite model creation but not for first run on new seed

    else:
        special_runs = [-1]

    save_path = exp_cfg.study_override_cfg.trainer.results_path
    
    for special_var in special_runs:
        if special_var != -1:
            log.info(f'Experiment: {special_var}% of the entire dataset')
            exp_cfg = deepcopy(saved_cfg)
            exp_cfg.study_override_cfg.trainer.split_addition_percentage = special_var
            exp_cfg.study_override_cfg.trainer.results_path += str(special_var).replace('.', '') + '/'
            if not exists(exp_cfg.study_override_cfg.trainer.results_path):
                makedirs(exp_cfg.study_override_cfg.trainer.results_path)

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
        
        ugle.utils.display_evaluation_results(experiment_tracker)
        


    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='parsing the experiment to run')
    parser.add_argument('-ec', '--experiment_config', type=str, required=True,
                        help='the location of the experiment config')
    parsed = parser.parse_args()
    run_experiment(parsed.experiment_config)
