import ugle
from model_evaluations import run_experiment
from main import neural_run

def test_loading_real_data():
    features, label, training_adj, testing_adj = ugle.datasets.load_real_graph_data('cora', 0.5, False)
    features, label, training_adj, testing_adj = ugle.datasets.load_real_graph_data('facebook', 0.5, False)
    return

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


def test_pipeline():
    # test that pipeline fallsback to cpu correctly
    run_experiment('ugle/configs/testing/min_cpu_fallback.yaml')
    # test that pipeline works
    run_experiment('ugle/configs/testing/min_working_pipeline_config.yaml')


def test_multi_objective():
    run_experiment('ugle/configs/testing/min_multi_objective.yaml')
    run_experiment('ugle/configs/testing/min_multi_hpo_non_neural.yaml')


if __name__ == "__main__":
    import ugle
    import numpy as np
    n_nodes = 1000
    n_features = 200
    n_clusters = 3

    # demo to evaluate a In Memory dataset 
    dataset = {'features': np.random.rand(n_nodes, n_features),
               'adjacency': np.random.rand(n_nodes, n_nodes),
               'label': np.random.randint(0, n_clusters+1, size=n_nodes)}
    # evalute dmon with hpo
    #Trainer = ugle.trainer.ugleTrainer("dmon")
    #results = Trainer.eval(dataset)
    #print(results)

    # load the dmon default hyperparameters
    cfg = ugle.utils.load_model_config(override_model="dmon_default")
    Trainer = ugle.trainer.ugleTrainer("dmon", cfg)
    results = Trainer.eval(dataset)

    # demo to evaluate one of the benchmarking datasets
    #Trainer = ugle.trainer.ugleTrainer("daegc")
    #Trainer.cfg.dataset = "cora"
    #results = Trainer.eval()