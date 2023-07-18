import ugle
import ugle.utils as utils
import ugle.datasets as datasets
from ugle.logger import log

from omegaconf import OmegaConf, DictConfig, ListConfig
import numpy as np
import optuna
from optuna import Trial
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
import threading
import time
from alive_progress import alive_it
import copy

# https://stackoverflow.com/questions/9850995/tracking-maximum-memory-usage-by-a-python-function

class StoppableThread(threading.Thread):
    def __init__(self):
        super(StoppableThread, self).__init__()
        self.daemon = True
        self.__monitor = threading.Event()
        self.__monitor.set()
        self.__has_shutdown = False

    def run(self):
        '''Overloads the threading.Thread.run'''
        # Call the User's Startup functions
        self.startup()

        # Loop until the thread is stopped
        while self.isRunning():
            self.mainloop()

        # Clean up
        self.cleanup()

        # Flag to the outside world that the thread has exited
        # AND that the cleanup is complete
        self.__has_shutdown = True

    def stop(self):
        self.__monitor.clear()

    def isRunning(self):
        return self.__monitor.isSet()

    def isShutdown(self):
        return self.__has_shutdown

    ###############################
    ### User Defined Functions ####
    ###############################

    def mainloop(self):
        '''
        Expected to be overwritten in a subclass!!
        Note that Stoppable while(1) is handled in the built in "run".
        '''
        pass

    def startup(self):
        '''Expected to be overwritten in a subclass!!'''
        pass

    def cleanup(self):
        '''Expected to be overwritten in a subclass!!'''
        pass


class MyLibrarySniffingClass(StoppableThread):
    def __init__(self, target_lib_call):
        super(MyLibrarySniffingClass, self).__init__()
        self.target_function = target_lib_call
        self.results = None

    def startup(self):
        # Overload the startup function
        log.info("Starting Memory Calculation")

    def cleanup(self):
        # Overload the cleanup function
        log.info("Ending Memory Tracking")

    def mainloop(self):
        # Start the library Call
        self.results = self.target_function()

        # Kill the thread when complete
        self.stop()


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
        log.info(f'Trial {trial.number} finished. Validation result || {valid_metrics[0]}: {trial_value} ||')
        # log best trial comparison
        new_best_message = f'New best trial {trial.number}'
        if trial.number > 0:
            if trial_value > trial.study.best_value and trial.study.direction.name == 'MAXIMIZE':
                log.info(new_best_message)
            elif trial_value < trial.study.best_value and trial.study.direction.name == 'MINIMIZE':
                log.info(new_best_message)
            else:
                log.info(
                    f'Trial {trial.number} finished. Best trial is {trial.study.best_trial.number} with {valid_metrics[0]}: {trial.study.best_value}')
        else:
            log.info(new_best_message)

    else:
        # log trial results
        right_order_results = [results[k] for k in valid_metrics]
        to_log_trial_values = ''.join(
            f'| {metric}: {right_order_results[i]} |' for i, metric in enumerate(valid_metrics))
        log.info(f'Trial {trial.number} finished. Validation result |{to_log_trial_values}|')
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
                log.info(f'New Best trial for metrics: {improved_metrics_str}')
            else:
                log.info('Trial worse than existing across all metrics')

            best_so_far = ''.join(
                f'trial {associated_trial[i]} ({metric}: {best_values[i]}), ' for i, metric in enumerate(valid_metrics))
            best_so_far = best_so_far[:best_so_far.rfind(',')]
            log.info(f'Best results so far: {best_so_far}')

    return


class ugleTrainer:
    def __init__(self, cfg: DictConfig):
        super(ugleTrainer, self).__init__()

        self.device = ugle.utils.set_device(cfg.trainer.gpu)

        utils.set_random(cfg.args.random_seed)
        if cfg.trainer.show_config:
            log.info(OmegaConf.to_yaml(cfg.args))

        cfg = ugle.utils.process_study_cfg_parameters(cfg)

        self.cfg = cfg

        self.progress_bar = None
        self.model = None
        self.loss_function = None
        self.scheduler = None
        self.optimizers = None
        self.patience_wait = 0
        self.best_loss = 1e9
        self.current_epoch = 0


    def load_database(self):

        if 'synth' not in self.cfg.dataset:
            # loads and splits the dataset
            features, label, train_adjacency, test_adjacency = datasets.load_real_graph_data(
                self.cfg.dataset,
                self.cfg.trainer.training_to_testing_split,
                self.cfg.trainer.split_scheme,
                self.cfg.trainer.split_addition)
        else:
            _, adj_type, n_clusters = self.cfg.dataset.split("_")
            n_clusters = int(n_clusters)
            adjacency, features, label = datasets.create_synth_graph(n_nodes=1000, n_features=500, n_clusters=n_clusters, adj_type=adj_type)
            train_adjacency, test_adjacency = datasets.split_adj(adjacency,
                                                                 self.cfg.trainer.training_to_testing_split,
                                                                 self.cfg.trainer.split_scheme)

        # extract database relevant info
        if not self.cfg.args.n_clusters:
            self.cfg.args.n_clusters = np.unique(label).shape[0]
        self.cfg.args.n_nodes = features.shape[0]
        self.cfg.args.n_features = features.shape[1]

        return features, label, train_adjacency, test_adjacency

    def eval(self):
        # loads the database to train on
        features, label, validation_adjacency, test_adjacency = self.load_database()

        # creates store for range of hyperparameters optimised over
        self.cfg.hypersaved_args = copy.deepcopy(self.cfg.args)

        log.info('splitting dataset into train/validation')
        train_adjacency, validation_adjacency = datasets.split_adj(validation_adjacency, 
                                                          self.cfg.trainer.train_to_valid_split, 
                                                          self.cfg.trainer.split_scheme)

        # process data for training
        processed_data = self.preprocess_data(features, train_adjacency)
        processed_valid_data = self.preprocess_data(features, validation_adjacency)

        if not self.cfg.trainer.load_existing_test:
            optuna.logging.disable_default_handler()
            # creates the hpo study
            if self.cfg.trainer.multi_objective_study:
                study = optuna.create_study(study_name=f'{self.cfg.model}_{self.cfg.dataset}',
                                            directions=self.cfg.trainer.optimisation_directions,
                                            pruner=HyperbandPruner(min_resource=1,
                                                                   max_resource=self.cfg.trainer.n_valid_splits),
                                            sampler=TPESampler(seed=self.cfg.args.random_seed))
            else:
                study = optuna.create_study(study_name=f'{self.cfg.model}_{self.cfg.dataset}',
                                            direction=self.cfg.trainer.optimisation_directions[0],
                                            pruner=HyperbandPruner(min_resource=1,
                                                                   max_resource=self.cfg.trainer.n_valid_splits),
                                            sampler=TPESampler(seed=self.cfg.args.random_seed))
            log.info(f"A new hyperparameter study created: {study.study_name}")
            study.optimize(lambda trial: self.train(trial, self.cfg.args,
                                                    label,
                                                    features,
                                                    processed_data,
                                                    validation_adjacency,
                                                    processed_valid_data),
                           n_trials=self.cfg.trainer.n_trials_hyperopt)

            # assigns test parameters found in the study
            if not self.cfg.trainer.multi_objective_study:
                params_to_assign = study.best_trial.params
                if params_to_assign != {}:
                    log.info('Best Hyperparameters found: ')
                    for hp_key, hp_val in params_to_assign.items():
                        log.info(f'{hp_key} : {hp_val}')
                self.cfg = utils.assign_test_params(self.cfg, params_to_assign)

        # retrains the model on the validation adj and evaluates test performance
        if not self.cfg.trainer.multi_objective_study:
            self.cfg.trainer.calc_time = False
            log.info('Retraining model on full training data')
            self.train(None, self.cfg.args, label, features, processed_data, validation_adjacency, processed_valid_data)
            processed_test_data = self.preprocess_data(features, test_adjacency)
            results = self.testing_loop(label, features, test_adjacency, processed_test_data,
                                        self.cfg.trainer.test_metrics)
            # log test results
            right_order_results = [results[k] for k in self.cfg.trainer.valid_metrics]
            to_log_trial_values = ''.join(f'| {metric}: {right_order_results[i]} |' for i, metric in
                                          enumerate(self.cfg.trainer.valid_metrics))
            log.info(f'Test results |{to_log_trial_values}|')

            objective_results = {'metrics': self.cfg.trainer.valid_metrics[0],
                                 'results': results,
                                 'args': params_to_assign}

        else:
            if not self.cfg.trainer.load_existing_test:
                # best hp found for metrics
                best_values, associated_trial = ugle.utils.extract_best_trials_info(study, self.cfg.trainer.valid_metrics)
                unique_trials = list(np.unique(np.array(associated_trial)))
            elif not self.cfg.get("previous_results", False):
                unique_trials = [-1]
            else:
                unique_trials = list(range(len(self.cfg.previous_results)))

            objective_results = []
            for idx, best_trial_id in enumerate(unique_trials):
                if not self.cfg.trainer.load_existing_test:
                    # find the metrics for which trial is best at
                    best_at_metrics = [metric for i, metric in enumerate(self.cfg.trainer.valid_metrics) if
                                       associated_trial[i] == best_trial_id]
                    best_hp_params = [trial for trial in study.best_trials if trial.number == best_trial_id][0].params
                elif best_trial_id != -1:
                    best_hp_params = self.cfg.previous_results[idx].args
                    best_at_metrics = self.cfg.previous_results[idx].metrics
                else:
                    best_at_metrics = self.cfg.trainer.test_metrics
                    best_hp_params = self.cfg.args

                if best_trial_id != -1:
                    # log the best hyperparameters and metrics at which they are best
                    best_at_metrics_str = ''.join(f'{metric}, ' for metric in best_at_metrics)
                    best_at_metrics_str = best_at_metrics_str[:best_at_metrics_str.rfind(',')]
                    if not self.cfg.trainer.load_existing_test:
                        log.info(f'Best hyperparameters for metrics {best_at_metrics_str}: ')
                        for hp_key, hp_val in best_hp_params.items():
                            log.info(f'{hp_key} : {hp_val}')

                    # assign hyperparameters for the metric optimised over
                    self.cfg = utils.assign_test_params(self.cfg, best_hp_params)

                # do testing
                self.cfg.trainer.calc_time = False
                log.info('Retraining model on full training data')
                self.train(None, self.cfg.args, label, features, processed_data, validation_adjacency,
                           processed_valid_data)
                processed_test_data = self.preprocess_data(features, test_adjacency)
                results = self.testing_loop(label, features, test_adjacency, processed_test_data,
                                            self.cfg.trainer.test_metrics)

                # log test results
                right_order_results = [results[k] for k in self.cfg.trainer.valid_metrics]
                to_log_trial_values = ''.join(f'| {metric}: {right_order_results[i]} |' for i, metric in
                                              enumerate(self.cfg.trainer.valid_metrics))
                log.info(f'Test results |{to_log_trial_values}|')

                objective_results.append({'metrics': best_at_metrics,
                                          'results': results,
                                          'args': best_hp_params})

                # re init the args object for assign_test params
                self.cfg.args = copy.deepcopy(self.cfg.hypersaved_args)

        return objective_results

    def train(self, trial: Trial, args: DictConfig, label: np.ndarray, features: np.ndarray, processed_data: tuple,
              validation_adjacency: np.ndarray, processed_valid_data: tuple):
        # configuration of args if hyperparameter optimisation phase
        if trial is not None:
            log.info(f'Launching Trial {trial.number}')
            self.cfg.args = ugle.utils.sample_hyperparameters(trial, args)

        # start timer
        if self.cfg.trainer.calc_time:
            time_start = time.perf_counter()

        # process model creation
        self.training_preprocessing(self.cfg.args, processed_data)

        # run actual training of model
        self.training_loop(processed_data)

        if self.cfg.trainer.calc_time:
            time_elapsed = (time.perf_counter() - time_start)
            log.info(f"TIME taken in secs: {time_elapsed:5.2f}")

        # validation pass
        results = self.testing_loop(label, features, validation_adjacency, processed_valid_data,
                                    self.cfg.trainer.valid_metrics)
        log.info(f'model trained and tested')

        if trial is None:
            return
        else:
            self.cfg.args = copy.deepcopy(self.cfg.hypersaved_args)
            self.best_loss = 1e9
            log_trial_result(trial, results, self.cfg.trainer.valid_metrics, self.cfg.trainer.multi_objective_study)

            if not self.cfg.trainer.multi_objective_study:
                return results[self.cfg.trainer.valid_metrics[0]]
            else:
                right_order_results = [results[k] for k in self.cfg.trainer.valid_metrics]
                return tuple(right_order_results)

    def training_loop(self, processed_data: tuple):
        """
        the function to handle what happens across all training models including:
        validation checking, early stopping, progress bar output
        """
        self.model.train()
        self.progress_bar = alive_it(range(self.cfg.args.max_epoch))
        for self.current_epoch in self.progress_bar:

            loss, data_returned = self.training_epoch_iter(self.cfg.args, processed_data)
            if data_returned:
                processed_data = data_returned

            for opt in self.optimizers:
                opt.zero_grad()
            loss.backward()
            for opt in self.optimizers:
                opt.step()
            if self.scheduler:
                self.scheduler.step()

            if loss < self.best_loss:
                self.best_loss = loss
                self.patience_wait = 0
                if isinstance(loss, list):
                    self.progress_bar.title = f'Training {self.cfg.model} on {self.cfg.dataset} ... Epoch {self.current_epoch}: Losses=' + \
                                              ' '.join([f'{loss_value.numpy(): .4f}' for loss_value in loss])
                else:
                    self.progress_bar.title = f'Training {self.cfg.model} on {self.cfg.dataset} ... Epoch {self.current_epoch}: Loss={loss.item():.4f}'

            else:
                self.patience_wait += 1

            if self.patience_wait == self.cfg.args.patience:
                log.info(f'Early stopping at epoch {self.current_epoch}!')
                break

        return

    def testing_loop(self, label: np.ndarray, features: np.ndarray, adjacency: np.ndarray, processed_data: tuple, eval_metrics: ListConfig):
        """
        testing loop which processes the testing data, run through the model to get predictions
        evaluate those predictions
        """
        self.model.eval()
        preds = self.test(processed_data)
        results, eval_preds = ugle.process.preds_eval(label,
                                                      preds,
                                                      sf=4,
                                                      adj=adjacency,
                                                      metrics=eval_metrics)

        return results

    def preprocess_data(self, features: np.ndarray, adjacency: np.ndarray) -> tuple:
        log.error('NO PROCESSING STEP IMPLEMENTED FOR MODEL')
        pass

    def test(self, processed_data: tuple) -> np.ndarray:
        log.error('NO TESTING STEP IMPLEMENTED')
        pass

    def training_preprocessing(self, args: DictConfig, processed_data: tuple):
        """
        preparation steps of training
        """
        log.error('NO TRAINING PREPROCESSING PROCEDURE')
        pass

    def training_epoch_iter(self, args: DictConfig, processed_data: tuple):
        """
        this is what happens inside the iteration of epochs
        """
        log.error('NO TRAINING ITERATION LOOP')
        pass