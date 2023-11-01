import ugle
import ugle.utils as utils
import ugle.datasets as datasets
from ugle.logger import log
import time
from omegaconf import OmegaConf, DictConfig, ListConfig
import numpy as np
import optuna
from optuna import Trial, Study
from optuna.samplers import TPESampler
from optuna.trial import TrialState
import time
from tqdm import trange
import copy
import torch
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import optuna
import warnings
from os.path import exists
from os import makedirs, devnull
from memory_profiler import memory_usage

from optuna.exceptions import ExperimentalWarning
warnings.filterwarnings("ignore", category=ExperimentalWarning)
optuna.logging.set_verbosity(optuna.logging.CRITICAL)


class StopWhenMaxTrialsHit:
    def __init__(self, max_n_trials: int, max_n_pruned: int):
        self.max_n_trials = max_n_trials
        self.max_pruned = max_n_pruned
        self.completed_trials = 0
        self.pruned_in_a_row = 0

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            self.completed_trials += 1
            self.pruned_in_a_row = 0
        elif trial.state == optuna.trial.TrialState.PRUNED:
            self.pruned_in_a_row += 1

        if self.completed_trials >= self.max_n_trials:
            log.info('Stopping Study for Reaching Max Number of Trials')
            study.stop()

        if self.pruned_in_a_row >= self.max_pruned:
            log.info('Stopping Study for Reaching Max Number of Pruned Trials in a Row')
            study.stop()


class ParamRepeatPruner:
    """Prunes repeated trials, which means trials with the same parameters won't waste time/resources."""

    def __init__(
        self,
        study: optuna.study.Study,
        repeats_max: int = 0,
        should_compare_states: List[TrialState] = [TrialState.COMPLETE],
        compare_unfinished: bool = True,
    ):
        """
        Args:
            study (optuna.study.Study): Study of the trials.
            repeats_max (int, optional): Instead of prunning all of them (not repeating trials at all, repeats_max=0) you can choose to repeat them up to a certain number of times, useful if your optimization function is not deterministic and gives slightly different results for the same params. Defaults to 0.
            should_compare_states (List[TrialState], optional): By default it only skips the trial if the paremeters are equal to existing COMPLETE trials, so it repeats possible existing FAILed and PRUNED trials. If you also want to skip these trials then use [TrialState.COMPLETE,TrialState.FAIL,TrialState.PRUNED] for example. Defaults to [TrialState.COMPLETE].
            compare_unfinished (bool, optional): Unfinished trials (e.g. `RUNNING`) are treated like COMPLETE ones, if you don't want this behavior change this to False. Defaults to True.
        """
        self.should_compare_states = should_compare_states
        self.repeats_max = repeats_max
        self.repeats: Dict[int, List[int]] = defaultdict(lambda: [], {})
        self.unfinished_repeats: Dict[int, List[int]] = defaultdict(lambda: [], {})
        self.compare_unfinished = compare_unfinished
        self.study = study

    @property
    def study(self) -> Optional[optuna.study.Study]:
        return self._study

    @study.setter
    def study(self, study):
        self._study = study
        if self.study is not None:
            self.register_existing_trials()

    def register_existing_trials(self):
        """In case of studies with existing trials, it counts existing repeats"""
        trials = self.study.trials
        trial_n = len(trials)
        for trial_idx, trial_past in enumerate(self.study.trials[1:]):
            self.check_params(trial_past, False, -trial_n + trial_idx)

    def prune(self):
        self.check_params()

    def should_compare(self, state):
        return any(state == state_comp for state_comp in self.should_compare_states)

    def clean_unfinised_trials(self):
        trials = self.study.trials
        finished = []
        for key, value in self.unfinished_repeats.items():
            if self.should_compare(trials[key].state):
                for t in value:
                    self.repeats[key].append(t)
                finished.append(key)

        for f in finished:
            del self.unfinished_repeats[f]

    def check_params(
        self,
        trial: Optional[optuna.trial.BaseTrial] = None,
        prune_existing=True,
        ignore_last_trial: Optional[int] = None,
    ):
        if self.study is None:
            return
        trials = self.study.trials
        if trial is None:
            trial = trials[-1]
            ignore_last_trial = -1

        self.clean_unfinised_trials()

        self.repeated_idx = -1
        self.repeated_number = -1
        for idx_p, trial_past in enumerate(trials[:ignore_last_trial]):
            should_compare = self.should_compare(trial_past.state)
            should_compare |= (
                self.compare_unfinished and not trial_past.state.is_finished()
            )
            if should_compare and trial.params == trial_past.params:
                if not trial_past.state.is_finished():
                    self.unfinished_repeats[trial_past.number].append(trial.number)
                    continue
                self.repeated_idx = idx_p
                self.repeated_number = trial_past.number
                break

        if self.repeated_number > -1:
            self.repeats[self.repeated_number].append(trial.number)
        if len(self.repeats[self.repeated_number]) > self.repeats_max:
            if prune_existing:
                log.info('Pruning Trial for Suggesting Duplicate Parameters')
                raise optuna.exceptions.TrialPruned()

        return self.repeated_number

    def get_value_of_repeats(
        self, repeated_number: int, func=lambda value_list: np.mean(value_list)
    ):
        if self.study is None:
            raise ValueError("No study registered.")
        trials = self.study.trials
        values = (
            trials[repeated_number].value,
            *(
                trials[tn].value
                for tn in self.repeats[repeated_number]
                if trials[tn].value is not None
            ),
        )
        return func(values)
    

def log_trial_result(trial: Trial, results: dict, valid_metrics: list, multi_objective_study: bool):
    """
    logs the results for a trial
    :param trial: trial object
    :param results: result dictionary for trial
    :param valid_metrics: validation metrics used
    :param multi_objective_study: boolean whether under multi-objective study or not
    """
    if not multi_objective_study:
        # log validation results
        trial_value = results[valid_metrics[0]]
        log.info(f'Trial {trial.number} finished. Validation performance: {valid_metrics[0]}={trial_value}')
        # log best trial comparison
        new_best_message = f'New best trial {trial.number}'
        if trial.number > 0:
            if trial_value > trial.study.best_value and trial.study.direction.name == 'MAXIMIZE':
                log.info(new_best_message)
            elif trial_value < trial.study.best_value and trial.study.direction.name == 'MINIMIZE':
                log.info(new_best_message)
            else:
                log.info(
                    f'Trial {trial.number} finished. Best trial is {trial.study.best_trial.number} with {valid_metrics[0]}={trial.study.best_value}')
        else:
            log.info(new_best_message)

    else:
        # log trial results
        right_order_results = [results[k] for k in valid_metrics]
        to_log_trial_values = ''.join(
            f'{metric}={right_order_results[i]}, ' for i, metric in enumerate(valid_metrics))
        log.info(f'Trial {trial.number} finished. Validation performance: {to_log_trial_values.rpartition(",")[0]}')
        # log best trial comparison
        if trial.number > 0:
            # get best values for each metric across the best trials
            best_values, associated_trial = ugle.utils.extract_best_trials_info(trial.study, valid_metrics)
            # compare best values for each trial with new values found
            improved_metrics = []
            for i, metric_result in enumerate(right_order_results):
                if (metric_result > best_values[i] and trial.study.directions[i].name == 'MAXIMIZE') or \
                        (metric_result < best_values[i] and trial.study.directions[i].name == 'MINIMIZE'):
                    best_values[i] = metric_result
                    improved_metrics.append(valid_metrics[i])
                    associated_trial[i] = trial.number

            # log best trial and value for each metric
            if improved_metrics:
                improved_metrics_str = ''.join(f'{metric}, ' for metric in improved_metrics)
                improved_metrics_str = improved_metrics_str[:improved_metrics_str.rfind(',')]
                log.info(f'New best trial for metrics: {improved_metrics_str}')
            else:
                log.info('Trial worse than existing across all metrics')

            best_so_far = ''.join(
                f'trial {associated_trial[i]} ({metric}={best_values[i]}), ' for i, metric in enumerate(valid_metrics))
            best_so_far = best_so_far[:best_so_far.rfind(',')]
            log.info(f'Best results so far: {best_so_far}')

    return


class ugleTrainer:
    def __init__(self, cfg: DictConfig):
        super(ugleTrainer, self).__init__()

        _device = ugle.utils.set_device(cfg.trainer.gpu)
        if _device == 'cpu':
            self.device_name = _device
        else:
            self.device_name = cfg.trainer.gpu
        self.device = torch.device(_device)

        utils.set_random(cfg.args.random_seed)
        if cfg.trainer.show_config:
            log.info(OmegaConf.to_yaml(cfg.args))

        cfg = ugle.utils.process_study_cfg_parameters(cfg)

        self.cfg = cfg
        if not exists(cfg.trainer.models_path):
            makedirs(cfg.trainer.models_path)

        self.progress_bar = None
        self.model = None
        self.loss_function = None
        self.scheduler = None
        self.optimizers = None
        self.patience_wait = 0
        self.best_loss = 1e9
        self.current_epoch = 0

    def load_database(self):
        log.info(f'Loading dataset {self.cfg.dataset}')
        if 'synth' not in self.cfg.dataset:
            train_loader, val_loader, test_loader = datasets.create_dataset_loader(self.cfg.dataset, 
                                                                                   self.cfg.trainer.max_nodes_in_dataloader, 
                                                                                   self.cfg.trainer.train_to_valid_split, 
                                                                                   self.cfg.trainer.training_to_testing_split)

        else:
            _, adj_type, feature_type, n_clusters = self.cfg.dataset.split("_")
            n_clusters = int(n_clusters)
            train_loader, val_loader, test_loader = datasets.create_synth_graph(n_nodes=1000, n_features=500, n_clusters=n_clusters, 
                                                                                num_val=self.cfg.trainer.train_to_valid_split,
                                                                                num_test=self.cfg.trainer.training_to_testing_split,
                                                                                adj_type=adj_type, feature_type=feature_type)

        # extract database relevant info
        dstats = datasets.extract_dataset_stats(train_loader)
    
        if not self.cfg.args.n_clusters:
            self.cfg.args.n_clusters = int(dstats['n_clusters'])
        self.cfg.args.n_nodes = int(dstats['n_nodes'])
        self.cfg.args.n_features = int(dstats['n_features'])
        self.cfg.args.n_edges = int(dstats['n_edges'])

        return train_loader, val_loader, test_loader
    
    def eval(self):
        # loads the database to train on
        train_loader, val_loader, test_loader = self.load_database()

        # creates store for range of hyperparameters optimised over
        self.cfg.hypersaved_args = copy.deepcopy(self.cfg.args)

        # process data for training
        train_loader = self.preprocess_data(train_loader)
        val_loader = self.preprocess_data(val_loader)
        test_loader = self.preprocess_data(test_loader)

        # if only testing then no need to optimise hyperparameters
        if not self.cfg.trainer.only_testing:
            optuna.logging.disable_default_handler()
            # creates the hpo study
            if self.cfg.trainer.multi_objective_study:
                study = optuna.create_study(study_name=f'{self.cfg.model}_{self.cfg.dataset}',
                                            directions=self.cfg.trainer.optimisation_directions,
                                            sampler=TPESampler(seed=self.cfg.args.random_seed, multivariate=True, group=True))
            else:
                study = optuna.create_study(study_name=f'{self.cfg.model}_{self.cfg.dataset}',
                                            direction=self.cfg.trainer.optimisation_directions[0],
                                            sampler=TPESampler(seed=self.cfg.args.random_seed))
            log.info(f"A new hyperparameter study created: {study.study_name}")

            # add study callbacks for stopping and repeat parameter suggestion pruner
            study_stop_cb = StopWhenMaxTrialsHit(self.cfg.trainer.n_trials_hyperopt, self.cfg.trainer.max_n_pruned)
            prune_params = ParamRepeatPruner(study)

            # add best hyperparameters from other seeds
            if self.cfg.trainer.suggest_hps_from_previous_seed:
                for hps in self.cfg.trainer.hps_found_so_far:
                    study.enqueue_trial(hps)

            # hpo
            study.optimize(lambda trial: self.train(trial, train_loader, val_loader,
                                                    prune_params=prune_params),
                                n_trials=self.cfg.trainer.n_trials_hyperopt,
                                callbacks=[study_stop_cb])

            # assigns test parameters found in the study
            if not self.cfg.trainer.multi_objective_study:
                params_to_assign = study.best_trial.params
                if params_to_assign != {}:
                    log.info('Best Hyperparameters found: ')
                    for hp_key, hp_val in params_to_assign.items():
                        log.info(f'{hp_key} : {hp_val}')
                self.cfg = utils.assign_test_params(self.cfg, params_to_assign)
        else:
            params_to_assign = 'default'


        if (not self.cfg.trainer.multi_objective_study) or self.cfg.trainer.only_testing:
            # retrain the model w/o hpo or w/ given hps
            self.model.train()
            validation_results = self.train(None, train_loader, val_loader)
            
            # get test results for all validation metrics 
            objective_results = []
            for opt_metric in self.cfg.trainer.valid_metrics:
                # evaluate model on test
                log.info(f'Evaluating {opt_metric} model on test split')
                self.model.load_state_dict(torch.load(f"{self.cfg.trainer.models_path}{self.cfg.model}_{self.device_name}_{opt_metric}.pt")['model'])
                self.model.to(self.device)
                self.model.eval()
                results = self.test(test_loader, self.cfg.trainer.test_metrics)

                # log test results
                right_order_results = [results[k] for k in self.cfg.trainer.test_metrics]
                to_log_trial_values = ''.join(f'{metric}={right_order_results[i]}, ' for i, metric in enumerate(self.cfg.trainer.test_metrics))
                log.info(f'Test results optimised for {opt_metric}: {to_log_trial_values.rpartition(",")[0]}')

                # save results to return
                objective_results.append({'metrics': opt_metric,
                                          'results': results,
                                          'args': params_to_assign})
                if self.cfg.trainer.save_validation:
                    objective_results[-1]['validation_results'] = validation_results

        else:
            # extract best hps found for each validation metric
            best_values, associated_trial = ugle.utils.extract_best_trials_info(study, self.cfg.trainer.valid_metrics)
            unique_trials = list(np.unique(np.array(associated_trial)))
            objective_results = []

            for idx, best_trial_id in enumerate(unique_trials):
                # find the metrics for which trial is best at
                best_at_metrics = [metric for i, metric in enumerate(self.cfg.trainer.valid_metrics) if
                                    associated_trial[i] == best_trial_id]
                best_hp_params = [trial for trial in study.best_trials if trial.number == best_trial_id][0].params

                # log the best hyperparameters and metrics at which they are best
                best_at_metrics_str = ''.join(f'{metric}, ' for metric in best_at_metrics)
                best_at_metrics_str = best_at_metrics_str[:best_at_metrics_str.rfind(',')]
                test_metrics = ''.join(f'{metric}_' for metric in best_at_metrics)
                test_metrics = test_metrics[:test_metrics.rfind(',')]
                log.info(f'Best hyperparameters for metric(s): {best_at_metrics_str} ')
                if not self.cfg.trainer.only_testing:
                    for hp_key, hp_val in best_hp_params.items():
                        log.info(f'{hp_key} : {hp_val}')

                # if finetuning or using same init of network then size of network has to be the same
                if self.cfg.trainer.finetuning_new_dataset or self.cfg.trainer.same_init_hpo:
                    args_to_overwrite = list(set(self.cfg.args.keys()).intersection(self.cfg.trainer.args_cant_finetune))
                    saved_args = OmegaConf.load(f'ugle/configs/models/{self.cfg.model}/{self.cfg.model}_default.yaml')
                    for k in args_to_overwrite:
                        best_hp_params[k] = saved_args.args[k]

                # assign hyperparameters for the metric optimised over
                self.cfg = utils.assign_test_params(self.cfg, best_hp_params)

                # retrain 
                log.debug('Retraining model')
                self.model.train()
                validation_results = self.train(None, train_loader, val_loader)
                
                # at this point, the self.train() loop should go have saved models for each validation metric 
                # then go through the best at metrics, and do a test for each of the best models 
                for opt_metric in best_at_metrics:
                    # if epoch is the same for all opt_metrics, there will be an unecessary forward pass but we move since unlikely 
                    log.debug(f'Evaluating {opt_metric} model on test split')
                    self.model.load_state_dict(torch.load(f"{self.cfg.trainer.models_path}{self.cfg.model}_{self.device_name}_{opt_metric}.pt")['model'])
                    self.model.to(self.device)
                    self.model.eval()
                    results = self.test(test_loader, self.cfg.trainer.test_metrics)
            
                    # log test results
                    right_order_results = [results[k] for k in self.cfg.trainer.test_metrics]
                    to_log_trial_values = ''.join(f'{metric}={right_order_results[i]}, ' for i, metric in enumerate(self.cfg.trainer.test_metrics))
                    log.info(f'Test results optimised for {opt_metric}: {to_log_trial_values.rpartition(",")[0]}')

                    # save the test results
                    objective_results.append({'metrics': opt_metric,
                                            'results': results,
                                            'args': best_hp_params})
                # if saving validation 
                if self.cfg.trainer.save_validation:
                    objective_results[-1]['validation_results'] = validation_results

                # re init the args object for assign_test params
                self.cfg.args = copy.deepcopy(self.cfg.hypersaved_args)

        return objective_results

    def train(self, trial: Trial, train_loader, val_loader, prune_params=None):
        
        timings = np.zeros(2)
        start = time.time()
        # configuration of args if hyperparameter optimisation phase
        if trial is not None:
            log.info(f'Launching Trial {trial.number}')
            # if finetuning or reusing init procedure then just use default architecture sizes
            if self.cfg.trainer.finetuning_new_dataset or self.cfg.trainer.same_init_hpo:
                args_to_overwrite = list(set(args.keys()).intersection(self.cfg.trainer.args_cant_finetune))
                saved_args = OmegaConf.load(f'ugle/configs/models/{self.cfg.model}/{self.cfg.model}_default.yaml')
                for k in args_to_overwrite:
                    self.cfg.args[k] = saved_args.args[k]
                    args = saved_args.args[k]
            # this will prune the trial if the args have appeared 
            self.cfg.args = copy.deepcopy(self.cfg.hypersaved_args)
            self.cfg.args = ugle.utils.sample_hyperparameters(trial, self.cfg.args, prune_params)

        # create the model for each trial
        self.training_preprocessing(self.cfg.args, train_loader)
        self.model.train()

        if self.cfg.trainer.finetuning_new_dataset:
            log.info('Loading pretrained model')
            self.model.load_state_dict(torch.load(f"{self.cfg.trainer.models_path}{self.cfg.model}.pt")['model'])
            self.model.to(self.device)

        if trial is not None:
            if self.cfg.trainer.same_init_hpo and trial.number == 0:
                log.info("Saving initilisation of parameters")
                torch.save(self.model.state_dict(), f"{self.cfg.trainer.models_path}{self.cfg.model}_{self.device_name}_init.pt")

            if self.cfg.trainer.same_init_hpo: 
                log.info("Loading same initilisation of parameters")
                self.model.load_state_dict(torch.load(f"{self.cfg.trainer.models_path}{self.cfg.model}_{self.device_name}_init.pt"))
                self.model.to(self.device)

        # create training loop
        self.progress_bar = trange(self.cfg.args.max_epoch, desc='Training...', leave=True, position=0, bar_format='{l_bar}{bar:15}{r_bar}{bar:-15b}', file=open(devnull, 'w'))
        best_so_far = np.zeros(len((self.cfg.trainer.valid_metrics))) - 0.01
        best_epochs = np.zeros(len((self.cfg.trainer.valid_metrics)), dtype=int)
        best_so_far[self.cfg.trainer.valid_metrics.index('conductance')] = 1.01
        best_so_far[self.cfg.trainer.valid_metrics.index('modularity')] = -1.01
        patience_waiting = np.zeros((len(self.cfg.trainer.valid_metrics)), dtype=int)

        # train the model 
        for self.current_epoch in self.progress_bar:
            # progress bar update
            if self.current_epoch % self.cfg.trainer.log_interval == 0:
                if self.current_epoch != 0:
                    if not (self.current_epoch - self.cfg.trainer.log_interval == 0 and self.cfg.trainer.calc_memory):
                        nlength_term = utils.remove_last_line()
                        if len_prev_line > nlength_term:
                            _ = utils.remove_last_line(nlength_term)
                    log.info(str(self.progress_bar))
                    len_prev_line = len(log.name) + 19 + len(str(self.progress_bar))
                else:
                    log.info(str(self.progress_bar))
                    len_prev_line = len(log.name) + 19 + len(str(self.progress_bar))

            # add time to train time, reset for val
            timings[0] += time.time() - start
            start = time.time()
            # check if validation time
            if self.current_epoch % self.cfg.trainer.validate_every_nepochs == 0:
                # compute training iteration
                if self.cfg.trainer.calc_memory and self.current_epoch == 0: 
                    mem_usage, results = memory_usage((self.test, (val_loader, self.cfg.trainer.valid_metrics)), 
                                                       interval=.005, retval=True) 
                    log.info(f"Max memory usage by testing_loop(): {max(mem_usage):.2f}MB")
                else:
                    results = self.test(val_loader, self.cfg.trainer.valid_metrics)

                # record best model state over this train/trial for each metric
                for m, metric in enumerate(self.cfg.trainer.valid_metrics):
                    if (results[metric] > best_so_far[m] and metric != 'conductance') or (results[metric] < best_so_far[m] and metric == 'conductance'):
                        best_so_far[m] = results[metric]
                        best_epochs[m] = self.current_epoch
                        # save model for this metric
                        torch.save({"model": self.model.state_dict(),
                                    "args": self.cfg.args},
                                    f"{self.cfg.trainer.models_path}{self.cfg.model}_{self.device_name}_{metric}.pt")
                        patience_waiting[m] = 0
                    else:
                        patience_waiting[m] += 1
            else: 
                patience_waiting += 1
            
            # add time to val time, reset for train
            timings[1] += time.time() - start
            start = time.time()
            # do one epoch of training
            self.model.train()
            loss = self.training_epoch_iter(train_loader)

            # update progress bar
            self.progress_bar.set_description(f'Training: m-{self.cfg.model}, d-{self.cfg.dataset}, s-{self.cfg.args.random_seed} -- Loss={loss.item():.4f}', refresh=True)
            # if all metrics hit patience then end
            if patience_waiting.all() >= self.cfg.args.patience:
                log.info(f'Early stopping at epoch {self.current_epoch}!')
                break
        
        # finalise progress bar logging
        nlength_term = utils.remove_last_line()
        if len_prev_line > nlength_term:
            _ = utils.remove_last_line(nlength_term)
        log.info(str(self.progress_bar))
        # finished training, record time taken
        timings[0] += time.time() - start
        start = time.time()    
        # compute final validation 
        return_results = dict(zip(self.cfg.trainer.valid_metrics, best_so_far))
        # if saving validation results then record the result on the test metrics
        if self.cfg.trainer.save_validation:
            valid_results = {}
        # save a model for each validation metric
        for m, metric in enumerate(self.cfg.trainer.valid_metrics):
            log.info(f'Best model for {metric} at epoch {best_epochs[m]}')
            # if validating then save the model
            if self.cfg.trainer.save_validation and trial is None:
                self.model.load_state_dict(torch.load(f"{self.cfg.trainer.models_path}{self.cfg.model}_{self.device_name}_{metric}.pt")['model'])
                self.model.to(self.device)
                self.model.eval()
                results = self.test(val_loader, self.cfg.trainer.test_metrics)
                valid_results[metric] = results

        # final validation record the time
        timings[1] += time.time() - start
        start = time.time()
        if self.cfg.trainer.calc_time:
            log.info(f"Time training {round(timings[0], 3)}s")
            log.info(f"Time validating {round(timings[1], 3)}s")

        # return values and log
        if trial is None:
            if not self.cfg.trainer.save_validation:
                return
            else:
                return valid_results
        else:
            log_trial_result(trial, return_results, self.cfg.trainer.valid_metrics, self.cfg.trainer.multi_objective_study)

            if not self.cfg.trainer.multi_objective_study:
                return return_results[self.cfg.trainer.valid_metrics[0]]
            else:
                right_order_results = [return_results[k] for k in self.cfg.trainer.valid_metrics]
                return tuple(right_order_results)
            

    def preprocess_data(self, loader):
        log.error('NO PROCESSING STEP IMPLEMENTED FOR MODEL')
        pass

    def training_preprocessing(self, args, train_loader):
        log.error('NO TRAINING PREPROCESSING PROCEDURE')
        pass

    def training_epoch_iter(self, train_loader):
        log.error('NO TRAINING ITERATION LOOP')
        pass

    def test(self, test_loader):
        log.error('NO TESTING STEP IMPLEMENTED')
        pass

