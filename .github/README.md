# UGLE: Unsupervised GNN Learning Environment


[![Twitter URL](https://img.shields.io/twitter/url/https/twitter.com/willleeney.svg?style=social&label=Follow%20%40willleeney)](https://twitter.com/willleeney)
![GitHub Repo stars](https://img.shields.io/github/stars/willleeney/ugle?style=social)
![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/willleeney/ugle/main-workflow.yaml)


## Introduction

ugle is a library build on pytorch to compare implementations of GNNs for unsupervised clustering.
It consists of a wide range of methods, with the original implementations referenced in the source code.
We provide an experiment abstraction to compare different models and visualisation tools to plot results. 
Any method can be trained individually via the main script using a specified config and the trainer objects. 
[Optuna](https://optuna.readthedocs.io/en/stable/tutorial/index.html) is used to optimize hyperparameters, however models can be trained by specifying parameters. 


## Installation

To use this repository, you need to install `pytorch_geometric` so adjust the variables for the appropriate `TORCH` version and `CUDA` option. Please refer to the [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) installation guide if the following doesn't work for you as pytorch-geometric can be tricky with an M* Series Mac or Windows. Please use a virtual environment manager to install and use a version of `python>=3.9.12`. If you want a different version of `pytorch` then raise a pull request that can pass the version number to setup.py. 


-`CUDA=cpu | cu102 | cu113 | cu116` 
-`pip install --extra-index-url https://data.pyg.org/whl/torch-1.12.0+{CUDA}.html ugle`


## Quick Tour

Here is an example of how you can use the python APIs from this repo.

```python 
import ugle
import numpy as np
n_nodes = 1000
n_features = 200
n_clusters = 3

# demo to evaluate a In Memory dataset 
dataset = {'features': np.random.rand(n_nodes, n_features),
            'adjacency': np.random.rand(n_nodes, n_nodes),
            'label': np.random.randint(0, n_clusters+1, size=n_nodes)}

# load the dmon default hyperparameters and evaluate on in memory dataset
cfg = ugle.utils.load_model_config(override_model="dmon_default")
Trainer = ugle.trainer.ugleTrainer("dmon", cfg)
results = Trainer.eval(dataset)

# evalute dmon with hpo
Trainer = ugle.trainer.ugleTrainer("dmon")
Trainer.cfg.dataset = "cora"
Trainer.cfg.trainer.n_trials_hyperopt = 2 # this is how you change the config
Trainer.cfg.args.max_epoch = 250
results = Trainer.eval(dataset)
```

Changing the cfg property in the Trainer object will change the training and optimisation procedure as demonstrated above. Detailed below are all the options that can be changed. 

 ```ugle/configs/config.yaml```
```yaml
trainer:
  gpu: "0" # GPU index (-1 = CPU)
  show_config: False # prints the working config to cmd line
  
  results_path: './results/' # where to save results to
  data_path: './data/' # where to look for data
  models_path: './models/' # where to save models 

  # if load_exising_test = True then trainer will look in {load_hps_path}{cfg.dataset}_{cfg.model}.pkl"
  # for previous found hyperparameters
  load_existing_test: False
  load_hps_path: './found_hps/'
  
  save_hpo_study: False # whether to save the hpo investigation 
  save_model: False # whether to save the best model

  # hyperparameter options
  test_metrics: ['f1', 'nmi', 'modularity', 'conductance'] # metrics to evaluate test data 
  valid_metrics: ['f1', 'nmi', 'modularity', 'conductance'] # metrics used for hpo and model selection
  validate_every_nepochs: 5 # how many training epochs per validation 
  n_trials_hyperopt: 5 # how many hpo trials
  max_n_pruned: 20 # patience for repeated hpo trials

  training_to_testing_split: 0.2 # percentage of testing data compared to total of training+validation 
  train_to_valid_split: 0.2 # percentage of validation data as proportion of the whole dataset
  save_validation: False # whether to save validation performance in the results object
  split_scheme: 'drop_edges' # one of drop_edges, split_edges, all_edges, no_edges (see ugle.datasets.split_adj() for more info)

  # logging options
  calc_time: True # whether to calculate the time taken for evaluation
  calc_memory: False # whether to calculate the memory used by evaluation 
  log_interval: 5 # how often to refresh the progress bar

args: 
  random_seed: 42 # the random seed to set
  max_epoch: 5000 # the number of epochs to train for 
  # this is also where the specific model args are loaded into 
```

If you want to do the same on the command line then you can parse use some of these arguments when running `python main.py`. 

```python
--model=<NAME-OF-MODEL> 
--dataset=<NAME-OF-DATASET> 
--seed=<SEED-ID> 
--max_epoch=<MAX-EPOCHS-FOR-TRAINING> 
--gpu=<GPU-INDEX>
--load_existing_test # if you specify a model as "dmon_custom" and write some custom args in the appropriate file that you want to test on then use this argument to load these
```

## Running Experiment on Multiple (seeds/datasets/algorithms)

If you want to run a study over multiple seeds/dataset/algorithms then use the `model_evaluations.py` script alongside the `experiments` config files. See the `ugle/configs/testing/` directory for examples on different scripts. If you have the `null` option in both `datasets` and `algorithms` in your config file then you can run the config over a single dataset and algorithm combination by using the `-da` argument which puts this combination into the `dataset_algo_combinations` option. This can also be used to define a list of combinations to choose to evaluate.

```python3 model_evaluations.py -ec=<PATH-TO-EXP-CONFIG> -da=<DATASET>_<MODEL>```


## Existing GNN Implementations 

- [daegc](https://github.com/Tiger101010/DAEGC)
- [dgi](https://github.com/PetarV-/DGI)
- [dmon](https://github.com/google-research/google-research/blob/master/graph_embedding/dmon/dmon.py)
- [mvgrl](https://github.com/kavehhassani/mvgrl)
- [grace](https://github.com/CRIPAC-DIG/GRACE)
- [selfgnn](https://github.com/zekarias-tilahun/SelfGNN)
- [sublime](https://github.com/GRAND-Lab/SUBLIME)
- [bgrl](https://github.com/Namkyeong/BGRL_Pytorch)
- [vgaer](https://github.com/qcydm/VGAER/tree/main/VGAER_codes)
- [cagc](https://github.com/wangtong627/CAGC/)

## Existing Datasets

- acm
- amac 
- amap
- bat
- citeseer
- cora
- cocs
- dblp
- eat
- uat
- pubmed
- texas
- wisc
- cornell
- Physics
- CS
- Photo
- Computers


### how to add a new model <: MODEL-NAME>

To create a model with the name <MODEL-NAME>, you need to create minimum two files:
* create a file for the model: ```ugle/models/<MODEL-NAME>.py```
* create a file to hold the hyperparameters: ```ugle/configs/models/<MODEL-NAME>/<MODEL-NAME>.yaml```
* optional* create file to hold default hyperparameters : ```ugle/configs/models/<MODEL-NAME>/<MODEL-NAME>_default.yaml```
* optional* create a new .yaml file to hold other variations: ```ugle/configs/models/<MODEL_NAME>/<MODEL_NAME>_myparameters.yaml``` 
* run an experiments or a single model run using the name reference as : ```"<MODEL_NAME>_myparameters"```

Inside ```ugle/models/<MODEL-NAME>.py```, define four hooks to process the whole model

```python
from ugle.trainer import ugleTrainer # this is the base class for training any model in this framework

class <MODEL-NAME>_trainer(ugleTrainer):

    def preprocess_data(self, features, adjacency):
        # here we process the data into a required format 
        # some models using pytorch_geometric layers, some use custom layers 
        # this allows us to create new representation formats for features/adjacency matrix
        # additional data structures needed can be returned for access under the tuple: processed_data
        
        return (features, adj, ... )

    def training_preprocessing(self, args, processed_data):
        (features, adj, ...) = processed_data
        
        # here the model and optimisers are defined as follows
        
        self.model = ... Model()
        self.optimizers = [optimizer]
        
        return

    def training_epoch_iter(self, args, processed_data):
    
        # here is the training iterations hook, that defines the forward pass for each model 
        # the only requirement is that a loss is returned.
        # If the processed data changes and is needed for the next iteration, 
        # then, return the tuple you need in place of processed data
        # If this is not needed then None must be returned instead
    
        return loss, (None/updated_processed_data)

    def test(self, processed_data):
    
        # the testing loop
        # the definition of how predictions are returned by this model
        # 1d numpy array returned of predictions for each node
    
        return preds

```