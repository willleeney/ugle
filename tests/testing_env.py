import ugle
import sys
sys.path.append('D:/projects/base/app/modules')  

from pathlib import Path
main_directory = Path(__file__).parent.parent
sys.path.append(main_directory)   

from model_evaluations import run_experiment
from main import neural_run
import numpy as np

def test_loading_real_data():
    features, label, training_adj, testing_adj = ugle.datasets.load_real_graph_data('cora', 0.5, 'drop_edges')
    features, label, training_adj, testing_adj = ugle.datasets.load_real_graph_data('citeseer', 0.5, 'split_edges')
    features, label, training_adj, testing_adj = ugle.datasets.load_real_graph_data('citeseer', 0.5, 'all_edges')
    features, label, training_adj, testing_adj = ugle.datasets.load_real_graph_data('citeseer', 0.5, 'no_edges')


def test_neural():
    neural_run(override_model='grace_default')
    neural_run(override_model='dmon_default')
    neural_run(override_model='daegc_default')
    neural_run(override_model='mvgrl_default')
    neural_run(override_model='bgrl_default')
    neural_run(override_model='selfgnn_default')
    neural_run(override_model='sublime_default')
    neural_run(override_model='vgaer_default')
    neural_run(override_model='dgi_default')
    neural_run(override_model='cagc_default')


def test_exp_configs():
    run_experiment('ugle/configs/testing/min_working_pipeline.yaml')
    run_experiment('ugle/configs/testing/compute_allocation.yaml')
    run_experiment('ugle/configs/testing/min_hpo_resuggest.yaml')
    run_experiment('ugle/configs/testing/min_hpo_reuse.yaml')
    run_experiment('ugle/configs/testing/min_multi_objective.yaml')
    run_experiment('ugle/configs/testing/synth_test.yaml')


def test_pipeline():
    n_nodes = 1000
    n_features = 200
    n_clusters = 3

    # demo to evaluate a In Memory dataset 
    dataset = {'features': np.random.rand(n_nodes, n_features),
               'adjacency': np.random.rand(n_nodes, n_nodes),
               'label': np.random.randint(0, n_clusters+1, size=n_nodes)}

    # load the dmon default hyperparameters
    cfg = ugle.utils.load_model_config(override_model="dmon_default")
    Trainer = ugle.trainer.ugleTrainer("dmon", cfg)
    Trainer.cfg.args.max_epoch = 250
    results = Trainer.eval(dataset)

    # evalute dmon with hpo
    Trainer = ugle.trainer.ugleTrainer("dmon")
    Trainer.cfg.dataset = "cora"
    Trainer.cfg.trainer.n_trials_hyperopt = 2 # this is how you change the config
    Trainer.cfg.args.max_epoch = 250
    results = Trainer.eval(dataset)


if __name__ == "__main__":
    test_exp_configs()
    test_pipeline()
    test_neural()
    test_loading_real_data()