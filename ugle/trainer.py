import ugle
import ugle.utils as utils
import ugle.datasets as datasets
from ugle.logger import log
from ugle.hpo import ParamRepeatPruner, StopWhenMaxTrialsHit
import time
from omegaconf import OmegaConf, DictConfig, ListConfig
import numpy as np
import optuna
from optuna import Trial
from optuna.samplers import TPESampler
import time
from tqdm import trange
import copy
import torch
from typing import Dict, List, Optional, Tuple
import optuna
import warnings
from os import makedirs, devnull, get_terminal_size
from memory_profiler import memory_usage
import pickle
from torch_geometric.utils import to_undirected
import types

from optuna.exceptions import ExperimentalWarning
warnings.filterwarnings("ignore", category=ExperimentalWarning)
optuna.logging.set_verbosity(optuna.logging.CRITICAL)


def log_trial_result(trial: Trial, results: dict, valid_metrics: list, multi_objective_study: bool):
    """
        Logs the results for a trial.

        Args:
            trial (Trial): trial object
            results (dict): result dictionary for the trial
            valid_metrics (list): Validation metrics used
            multi_objective_study (boolean): indicating whether under multi-objective study or not

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
    def __init__(self, model_name: str, cfg=None):
        super(ugleTrainer, self).__init__()

        # get the required functions from the other debugger
        model_trainer = getattr(getattr(ugle.models, model_name), f"{model_name}_trainer")
        self.preprocess_data = types.MethodType(model_trainer.preprocess_data, self)
        self.training_preprocessing = types.MethodType(model_trainer.training_preprocessing, self)
        self.training_epoch_iter = types.MethodType(model_trainer.training_epoch_iter, self)
        self.test = types.MethodType(model_trainer.test, self)

        # if cfg is none then loads the default
        if not cfg:
            cfg = utils.load_model_config(override_model=model_name)
            cfg.trainer.only_testing = False

        # set the correct device for gpu:id / cpu 
        _device = ugle.utils.set_device(cfg.trainer.gpu)
        if _device == 'cpu':
            self.device_name = _device
        else:
            self.device_name = cfg.trainer.gpu
        self.device = torch.device(_device)

        # set the random seed
        utils.set_random(cfg.args.random_seed)
        if cfg.trainer.show_config:
            log.info(OmegaConf.to_yaml(cfg.args))

        # parses the config options for hpo study
        cfg = ugle.utils.process_study_cfg_parameters(cfg)
        self.cfg = cfg

        makedirs(cfg.trainer.models_path, exist_ok=True)
        makedirs(cfg.trainer.results_path, exist_ok=True)

        self.progress_bar = None
        self.model = None
        self.loss_function = None
        self.scheduler = None
        self.optimizers = None
        self.patience_wait = 0
        self.best_loss = 1e9
        self.current_epoch = 0
     

    def load_database(self, dataset: dict=None):
        """
        Function to load the dataset and set the attributes of the dataset to cfg.

        Args:
            dataset (dict): contains an in memory dataset

        Returns: 
            features (np.ndarray): feature matrix 
            label (np.ndarray): labels for each node
            train_adjacency (np.ndarray): train/validation split of the adjacency matrix
            test_adjacency (np.ndarray): test split of the adjacency matrix
        """
        if dataset:
            self.cfg.dataset = 'In_Memory'
        log.info(f'Loading dataset {self.cfg.dataset}')
        
        # if no dataset has been passed to trainer
        if not dataset:
            if 'synth' not in self.cfg.dataset:
                # loads and splits the dataset
                features, label, train_adjacency, test_adjacency = datasets.load_real_graph_data(
                    self.cfg.dataset,
                    self.cfg.trainer.training_to_testing_split,
                    self.cfg.trainer.split_scheme)
            else:
                _, adj_type, feature_type, n_clusters = self.cfg.dataset.split("_")
                n_clusters = int(n_clusters)
                adjacency, features, label = datasets.create_synth_graph(n_nodes=1000, n_features=500, n_clusters=n_clusters, 
                                                                        adj_type=adj_type, feature_type=feature_type)
                train_adjacency, test_adjacency = datasets.split_adj(adjacency,
                                                                     self.cfg.trainer.training_to_testing_split,
                                                                     self.cfg.trainer.split_scheme)
                
        # if an in-memory dataset has been passed
        else:
            assert 'adjacency' in dataset, "dataset did not contain adjacency info"
            assert 'features' in dataset, "dataset did not contain features info"
            assert 'adjacency' in dataset, "dataset did not contain label info"
            assert np.shape(dataset['features'])[0] == np.shape(dataset['adjacency'])[0], "feature and adjacency node dimension mismatch"
            assert np.shape(dataset['features'])[0] == np.shape(dataset['adjacency'])[0], "feature and adjacency node dimension mismatch"
            assert np.shape(dataset['features'])[0] == np.shape(dataset['label'])[0], "label node dimension mismatch with input data"

            log.info('Converting adjacency matrix connections to unweighted')
            dataset['adjacency'] = np.where(dataset['adjacency'] != 0, 1, dataset['adjacency'])

            log.info('Converting adjacency to undirected')
            dataset['adjacency'] = datasets.to_dense_adj(to_undirected(datasets.to_edge_index(dataset['adjacency']))).numpy().squeeze(0)

            train_adjacency, test_adjacency = datasets.split_adj(dataset['adjacency'],
                                                        self.cfg.trainer.training_to_testing_split,
                                                        self.cfg.trainer.split_scheme)
            
            features = dataset['features']
            label = dataset['label']

        # extract database relevant info
        if not self.cfg.args.n_clusters:
            self.cfg.args.n_clusters = np.unique(label).shape[0]
        self.cfg.args.n_nodes = features.shape[0]
        self.cfg.args.n_features = features.shape[1]

        return features, label, train_adjacency, test_adjacency
    
    def eval(self, dataset: dict=None):
        """
        Function to load database, split adjacency, preprocess dataset correctly for the method
        run a HPO if required, find the best model based on the validation data and then test 
        the best model. 

        Args: 
            dataset (dict): optional dataset dictionary that contains an In-memory dataset

        Returns:
            objective_results (dict): performance dictionary that contains results 
                along with associated best args and the metrics that they were best at

        """

        # loads the database to train on
        features, label, validation_adjacency, test_adjacency = self.load_database(dataset)
            
        # creates store for range of hyperparameters optimised over
        self.cfg.hypersaved_args = copy.deepcopy(self.cfg.args)

        log.debug('splitting dataset into train/validation')
        train_adjacency, validation_adjacency = datasets.split_adj(validation_adjacency, 
                                                                   self.cfg.trainer.train_to_valid_split, 
                                                                   self.cfg.trainer.split_scheme)

        # process data for training
        processed_data = self.preprocess_data(features, train_adjacency)
        processed_valid_data = self.preprocess_data(features, validation_adjacency)

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
            # create the trial pruners that remove repeat suggested parameters
            study_stop_cb = StopWhenMaxTrialsHit(self.cfg.trainer.n_trials_hyperopt, self.cfg.trainer.max_n_pruned)
            prune_params = ParamRepeatPruner(study)
            # adds the best hps from previous seeds to speed up optimisation
            if self.cfg.trainer.suggest_hps_from_previous_seed:
                for hps in self.cfg.trainer.hps_found_so_far:
                    study.enqueue_trial(hps)
            # perform the hpo
            study.optimize(lambda trial: self.train(trial,
                                                    label,
                                                    processed_data,
                                                    validation_adjacency,
                                                    processed_valid_data,
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
            study = None

        processed_test_data = self.preprocess_data(features, test_adjacency)
        if self.cfg.trainer.only_testing:
            validation_results = self.train(None, label, processed_data, validation_adjacency, processed_valid_data)
        elif not self.cfg.trainer.multi_objective_study:
            validation_results = study.best_trial.user_attrs['validation_results']
        # if you aren't optimising for more than one metric or are only evaluating hps 
        if (not self.cfg.trainer.multi_objective_study) or self.cfg.trainer.only_testing:                         
            objective_results = []
            for opt_metric in self.cfg.trainer.valid_metrics:
                log.info(f'Evaluating {opt_metric} model on test split')
                self.model.load_state_dict(torch.load(f"{self.cfg.trainer.models_path}{self.cfg.model}_{self.device_name}_{opt_metric}.pt")['model'])
                self.model.to(self.device)
                results = self.testing_loop(label, test_adjacency, processed_test_data, self.cfg.trainer.test_metrics)
                # log test results
                right_order_results = [results[k] for k in self.cfg.trainer.test_metrics]
                to_log_trial_values = ''.join(f'{metric}={right_order_results[i]}, ' for i, metric in enumerate(self.cfg.trainer.test_metrics))
                log.info(f'Test results optimised for {opt_metric}: {to_log_trial_values.rpartition(",")[0]}')
                objective_results.append({'metrics': opt_metric, 'results': results, 
                                          'args': params_to_assign, "validation_results": validation_results})

        # if you have optimised for more than one metric then there are potentially more than one 
        # metrics that need to be optimised for 
        else:
            # extract best hps found for each metrics
            best_values, associated_trial = ugle.utils.extract_best_trials_info(study, self.cfg.trainer.valid_metrics)
            unique_trials = list(np.unique(np.array(associated_trial)))
            objective_results = []

            for idx, best_trial_id in enumerate(unique_trials):
                # find the metrics for which trial is best at
                best_at_metrics = [metric for i, metric in enumerate(self.cfg.trainer.valid_metrics) if associated_trial[i] == best_trial_id]
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
                
                # assign best hyperparameters to config
                self.cfg = utils.assign_test_params(self.cfg, best_hp_params)

                # at this point, the self.train() loop should go have saved models for each validation metric 
                # then go through the best at metrics, and do a test for each of the best models 
                for opt_metric in best_at_metrics:
                    # if epoch is the same for all opt_metrics, there will be an unecessary forward pass but we move since unlikely 
                    log.debug(f'Evaluating {opt_metric} model on test split')
                    self.model.load_state_dict(torch.load(f"{self.cfg.trainer.models_path}{self.cfg.model}_{self.device_name}_{opt_metric}.pt")['model'])
                    self.model.to(self.device)
                    results = self.testing_loop(label, test_adjacency, processed_test_data, self.cfg.trainer.test_metrics)
                    
                    # log test results
                    right_order_results = [results[k] for k in self.cfg.trainer.test_metrics]
                    to_log_trial_values = ''.join(f'{metric}={right_order_results[i]}, ' for i, metric in enumerate(self.cfg.trainer.test_metrics))
                    log.info(f'Test results optimised for {opt_metric}: {to_log_trial_values.rpartition(",")[0]}')
                    objective_results.append({'metrics': opt_metric, 'results': results, 
                                              'args': best_hp_params, "validation_results": study.trials[best_trial_id].user_attrs['validation_results']})
                                            
                # re init the args object for assign_test params
                self.cfg.args = copy.deepcopy(self.cfg.hypersaved_args)

                # this isnt the best code but we move
                if self.cfg.trainer.save_model:
                    log.info(f'Saving best version of model for {best_at_metrics}')
                    torch.save({"model": self.model.state_dict(),
                                "args": best_hp_params},
                               f"{self.cfg.trainer.models_path}{self.cfg.model}_{test_metrics}.pt")

        if study and self.cfg.trainer.save_hpo_study:
            pickle.dump(study, open(f"{self.cfg.trainer.results_path}study_{self.cfg.args.random_seed}_{self.cfg.dataset}_{self.cfg.model}.pkl","wb"))

        return objective_results

    def train(self, trial: Trial, label: np.ndarray, processed_data: tuple, validation_adjacency: np.ndarray, processed_valid_data: tuple, prune_params=None):
        """
        Training function for training a GNN under hpo suggested or specified parameters. Contains functions 
        for switching to active devices, progress bar tracking and validation selection. 

        Args:
            trial (Trial): the trial object from the optuna hpo (can be None) 
            label (np.ndarray): the ground-truth for the dataset
            processed_data (tuple): the processed data necessary for the algorithm specified
            validation_adjacency (np.ndarray): the validation adjacency matrix
            processed_valid_data (tuple): tthe processed validation data necessary for the algorithm specified
            prune_params (object): the object for pruning parameters when suggested twice

        Returns: 
            if trial is None: returns validation results
            else: returns validation performance results as tuple for the trial in hpo
        """


        timings = np.zeros(2)
        start = time.time()
        # configuration of args if hyperparameter optimisation phase
        if trial is not None:
            log.info(f'Launching Trial {trial.number}')
            self.cfg.args = copy.deepcopy(self.cfg.hypersaved_args)
            # this will prune the trial if the args have appeared 
            self.cfg.args = ugle.utils.sample_hyperparameters(trial, self.cfg.args, prune_params)

        # process model creation
        processed_data = self.move_to_activedevice(processed_data)
        self.training_preprocessing(self.cfg.args, processed_data)
        self.model.train()

        # create training loop
        self.progress_bar = trange(self.cfg.args.max_epoch, desc='Training...', leave=True, position=0, bar_format='{l_bar}{bar:15}{r_bar}{bar:-15b}', file=open(devnull, 'w'))
        best_so_far = np.zeros(len((self.cfg.trainer.valid_metrics))) - 0.01
        best_epochs = np.zeros(len((self.cfg.trainer.valid_metrics)), dtype=int)
        if 'conductance' in self.cfg.trainer.valid_metrics:
            best_so_far[self.cfg.trainer.valid_metrics.index('conductance')] = 1.01
        patience_waiting = np.zeros((len(self.cfg.trainer.valid_metrics)), dtype=int)

        for self.current_epoch in self.progress_bar:
            if self.current_epoch % self.cfg.trainer.log_interval == 0:
                if self.current_epoch != 0:
                    if not (self.current_epoch - self.cfg.trainer.log_interval == 0 and self.cfg.trainer.calc_memory):
                        try: 
                            nlength_terminal = get_terminal_size().columns
                        except: 
                           nlength_terminal = 80
                        for _ in range((len_prev_line // nlength_terminal) + 1):
                            utils.remove_last_line() 
                    log.info(str(self.progress_bar))
                    len_prev_line = len(log.name) + 19 + len(str(self.progress_bar))
                else:
                    log.info(str(self.progress_bar))
                    len_prev_line = len(log.name) + 19 + len(str(self.progress_bar))

            timings[0] += time.time() - start
            start = time.time()
            # check if validation time
            if self.current_epoch % self.cfg.trainer.validate_every_nepochs == 0:
                # put in validation mode 
                processed_data = self.move_to_cpudevice(processed_data)

                # compute training iteration
                if self.cfg.trainer.calc_memory and self.current_epoch == 0: 
                    mem_usage, results = memory_usage((self.testing_loop, (label, validation_adjacency, processed_valid_data, self.cfg.trainer.valid_metrics)), 
                                                      interval=.005, retval=True) 
                    log.info(f"Max memory usage by testing_loop(): {max(mem_usage):.2f}MB")
                else:
                    results = self.testing_loop(label, validation_adjacency, processed_valid_data, self.cfg.trainer.valid_metrics)
                
                # put data back into training mode
                processed_data = self.move_to_activedevice(processed_data)

                # record best model state over this train/trial for each metric
                for m, metric in enumerate(self.cfg.trainer.valid_metrics):
                    if (results[metric] > best_so_far[m] and metric != 'conductance') or (results[metric] < best_so_far[m] and metric == 'conductance'):
                        best_so_far[m] = results[metric]
                        best_epochs[m] = self.current_epoch
                        # save model for this metric
                        torch.save({"model": self.model.state_dict(), "args": self.cfg.args},
                                    f"{self.cfg.trainer.models_path}{self.cfg.model}_{self.device_name}_{metric}.pt")
                        patience_waiting[m] = 0
                    else:
                        patience_waiting[m] += 1
            else: 
                patience_waiting += 1
            
            timings[1] += time.time() - start
            start = time.time()

            if self.cfg.trainer.calc_memory and self.current_epoch == 0:
                mem_usage, retvals = memory_usage((self.training_epoch_iter, (self.cfg.args, processed_data)), interval=.005, retval=True) 
                loss, data_returned = retvals
                log.info(f"Max memory usage by training_epoch_iter(): {max(mem_usage):.2f}MB")
            else:
                loss, data_returned = self.training_epoch_iter(self.cfg.args, processed_data)
            if data_returned:
                processed_data = data_returned

            # optimise
            for opt in self.optimizers:
                opt.zero_grad()

            if self.cfg.trainer.calc_memory and self.current_epoch == 0: 
                mem_usage = memory_usage((loss.backward), interval=.005)
                log.info(f"Max memory usage by loss.backward(): {max(mem_usage):.2f}MB")
            else:
                loss.backward()

            for opt in self.optimizers:
                opt.step()
            if self.scheduler:
                self.scheduler.step()

            # update progress bar
            self.progress_bar.set_description(f'Training: m-{self.cfg.model}, d-{self.cfg.dataset}, s-{self.cfg.args.random_seed} -- Loss={loss.item():.4f}', refresh=True)
            # if all metrics hit patience then end
            if patience_waiting.all() >= self.cfg.args.patience:
                log.info(f'Early stopping at epoch {self.current_epoch}!')
                break
        
        len_prev_line = len(log.name) + 19 + len(str(self.progress_bar))
        try: 
            nlength_terminal = get_terminal_size().columns
        except: 
            nlength_terminal = 80
        for _ in range((len_prev_line // nlength_terminal) + 1):
            utils.remove_last_line() 
        log.info(str(self.progress_bar))
        # finished training, record time taken
        timings[0] += time.time() - start
        start = time.time()    
        # compute final validation 
        return_results = dict(zip(self.cfg.trainer.valid_metrics, best_so_far))
        processed_data = self.move_to_cpudevice(processed_data)
        #  then record the result on the validation metrics
        valid_results = {}
        # print best performance per validation set
        for m, metric in enumerate(self.cfg.trainer.valid_metrics):
            log.info(f'Best model by validation {metric} ({best_so_far[m]}) at epoch {best_epochs[m]}')
            if trial is None:
                self.model.load_state_dict(torch.load(f"{self.cfg.trainer.models_path}{self.cfg.model}_{self.device_name}_{metric}.pt")['model'])
                self.model.to(self.device)
                results = self.testing_loop(label, validation_adjacency, processed_valid_data, self.cfg.trainer.test_metrics)
                valid_results[metric] = results
            

        timings[1] += time.time() - start
        start = time.time()
        if self.cfg.trainer.calc_time:
            log.info(f"Time training {round(timings[0], 3)}s")
            log.info(f"Time validating {round(timings[1], 3)}s")

        # log trial result in context of the hpo + return info for optimisation
        if trial is None:
            return valid_results
        else:
            trial.set_user_attr("validation_results", valid_results)
            log_trial_result(trial, return_results, self.cfg.trainer.valid_metrics, self.cfg.trainer.multi_objective_study)
            if not self.cfg.trainer.multi_objective_study:
                return return_results[self.cfg.trainer.valid_metrics[0]]
            else:
                right_order_results = [return_results[k] for k in self.cfg.trainer.valid_metrics]
                return tuple(right_order_results)
            
    def testing_loop(self, label: np.ndarray, adjacency: np.ndarray, processed_data: tuple, eval_metrics: ListConfig):
        """
        Process the testing data through the model to get predictions then evaluates those predictions.

        Args:
            label (np.ndarray): the ground-truth for the dataset
            adjacency (np.ndarray): the testing adjacency matrix - necessary for conductance and modularity
            processed_data (tuple): the processed test data ready for the model
            eval_metrics (ListConfig): a list of metrics to evaluate.

        Returns: 
            results (dict): performance results for each of the metrics
        """
        self.model.eval()
        processed_data = self.move_to_activedevice(processed_data)
        preds = self.test(processed_data)
        processed_data = self.move_to_cpudevice(processed_data)
        results, _ = ugle.process.preds_eval(label, preds, sf=4, adj=adjacency, metrics=eval_metrics)
        self.model.train()
        return results

    def preprocess_data(self, features: np.ndarray, adjacency: np.ndarray) -> tuple:
        log.error('NO PROCESSING STEP IMPLEMENTED FOR MODEL')
        pass

    def test(self, processed_data: tuple) -> np.ndarray:
        log.error('NO TESTING STEP IMPLEMENTED')
        pass

    def training_preprocessing(self, args: DictConfig, processed_data: tuple):
        log.error('NO TRAINING PREPROCESSING PROCEDURE')
        pass

    def training_epoch_iter(self, args: DictConfig, processed_data: tuple):
        log.error('NO TRAINING ITERATION LOOP')
        pass

    def move_to_cpudevice(self, data: torch.Tensor):
        return tuple(databite.to(torch.device("cpu"), non_blocking=True) if torch.is_tensor(databite) else databite for databite in data)

    def move_to_activedevice(self, data: torch.Tensor):
        return tuple(databite.to(self.device, non_blocking=True)  if torch.is_tensor(databite) else databite for databite in data)


