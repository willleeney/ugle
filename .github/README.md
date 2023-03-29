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

![hpo](https://github.com/willleeney/ugle/blob/main/.github/pseudocode.png?raw=true)


## Installation

```
$ conda create -n ugle python=3.9.12
$ conda activate ugle

$ git clone https://github.com/willleeney/ugle.git
$ cd ugle

$ bash install_ugle.sh
```


## Quick Tour

To run a full HPO investigation over all models, use the ```model_evaluations.py``` script, 
and point to the .yaml file ```eval_hpo_investigation.yaml``` as follows:

```python model_evaluations.py --exp=ugle/configs/evaluations/eval_hpo_investigation.yaml```

To run a single model use the ```main.py``` script to train a specific model on a specific dataset as follows:

```python main.py --model=daegc --dataset=cora```
```python main.py --model=daegc_default --dataset=citeseer```


## Tutorial/Examples

### useful documentation 

* experiment results will be saved under the name of the log file
* each study results will be saved under the name "dataset_model"


### how to manipulate .yaml files

The template config that all experiments use is located in ```ugle/configs/evaluations/eval_cfg_template.yaml```.
All arguments in this file are commented with their function in the code. 
This file is overwritten by the .yaml file specified when running model_evaluations.py. 
To run a customised experiment, create a new .yaml file, similar to ```ugle/configs/evaluations/eval_hpo_investigation.yaml```

The default config used as a template for all models is located under ```ugle/configs/config.yaml```. 
This is overwritten by the model config.

To create your own specification of hyperparameters to with which to optimise a model (<MODEL_NAME>).
* create a new .yaml file:  ```ugle/configs/models/<MODEL_NAME>/<MODEL_NAME>_myparameters.yaml``` 
* run an experiments or a single model run using the name reference as: ```"<MODEL_NAME>_myparameters"```

### how to add a new model

To create a model with the name <NEW_MODEL_NAME>, you need to create minimum two files:
* create a file for the model: ```ugle/models/<NEW_MODEL_NAME>.py```
* create a file to hold the hyperparameters: ```ugle/configs/models/<NEW_MODEL_NAME>/<NEW_MODEL_NAME>.yaml```
* optional* create file to hold default hyperparameters : ```ugle/configs/models/<NEW_MODEL_NAME>/<NEW_MODEL_NAME>_default.yaml```

Inside ```ugle/models/<NEW_MODEL_NAME>.py```, define four hooks to process the whole model
```
from ugle.trainer import ugleTrainer # this is the base class for training any model in this framework

class <NEW_MODEL_NAME>_trainer(ugleTrainer):

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

### what is currently included in the repo

```
├── ugle
|   ├── configs/ # directory of all configs
|   ├── models/ # directory of all models implemented
|   | 
|   ├── datasets.py # contains download dataset functions + augmentations
|   ├── gnn_architecture.py # contains custom gnn implementations
|   ├── utils.py # utilities for manipulating custom data structures in project
|   ├── trainer.py # trainer classes for training pipeline structuring
|   ├── process.py # process data using maths
|   ├── logger.py # logger definition 
|   ├── helper.py # helpful functions to manipulate result files and create figures 
|
├── main.py # file to run any model once
├── model_evaluations.py # file to run and compare many models
├── .gitignore
├── README.md 
├── LICENSE
├── data/ # data is stored 
├── results/ # results from experiments
├── tests/testing_env.py # file to test functions in the repo
├── install_ugle.sh # installation script to install torch, dgl, torch-geometric
├── setup.py
└── requirements.txt 
```

