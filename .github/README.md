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

```
$ conda create -n ugle python=3.9.12
$ conda activate ugle

$ git clone https://github.com/willleeney/ugle.git
$ cd ugle

$ bash install_ugle.sh
```


## Quick Tour


```python3 model_evaluations.py -ec=ugle/configs/experiments/unsupervised_limit/hpo_new.yaml -da=<dataset>_<model>```


Multiple datasets and models


The ```model_evaluations.py``` executes a HPO investigation. 
Use ```python model_evaluations.py --exp=ugle/configs/evaluations/hpo_investigation.yaml``` to reproduce the results from the paper.

The ```main.py``` script trains a single model on a single dataset as follows:
```python main.py --model=daegc --dataset=cora```
```python main.py --model=daegc_default --dataset=citeseer```
where the name of the model is as such 


if you want to use this as an api then do
```python 
import ugle
import numpy as np

dataset = {'features': np.ndarry[n_nodes, n_features],
           'adjacency': np.ndarray[n_nodes, n_nodes],
           'label': np.ndarray[n_nodes]}
Trainer = ugle.trainer("dmon")
results = Trainer.eval(dataset)

Trainer = ugle.trainer("daegc")
Trainer.cfg.dataset = "cora"
results = Trainer.eval()
```

## Existing GNN Implementations 

[daegc](https://github.com/Tiger101010/DAEGC)
[dgi](https://github.com/PetarV-/DGI)
[dmon](https://github.com/google-research/google-research/blob/master/graph_embedding/dmon/dmon.py)
[mvgrl](https://github.com/kavehhassani/mvgrl)
[grace](https://github.com/CRIPAC-DIG/GRACE)
[selfgnn](https://github.com/zekarias-tilahun/SelfGNN)
[sublime](https://github.com/GRAND-Lab/SUBLIME)
[bgrl](https://github.com/Namkyeong/BGRL_Pytorch)
[vgaer](https://github.com/qcydm/VGAER/tree/main/VGAER_codes)
[cagc](https://github.com/wangtong627/CAGC/)

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
- facebook 
- twitch
- wikipedia
- github
- lastfm
- deezer
- Physics
- CS
- Photo
- Computers


## Adding a new model <MODEL_NAME> 


To create your own specification of hyperparameters to with which to optimise a model (<MODEL_NAME>).
* create a new .yaml file:  ```ugle/configs/models/<MODEL_NAME>/<MODEL_NAME>_myparameters.yaml``` 
* run an experiments or a single model run using the name reference as: ```"<MODEL_NAME>_myparameters"```

### how to add a new model

To create a model with the name <NEW_MODEL_NAME>, you need to create minimum two files:
* create a file for the model: ```ugle/models/<NEW_MODEL_NAME>.py```
* create a file to hold the hyperparameters: ```ugle/configs/models/<NEW_MODEL_NAME>/<NEW_MODEL_NAME>.yaml```
* optional* create file to hold default hyperparameters : ```ugle/configs/models/<NEW_MODEL_NAME>/<NEW_MODEL_NAME>_default.yaml```
* optional* create a new .yaml file to hold other variations:  ```ugle/configs/models/<MODEL_NAME>/<MODEL_NAME>_myparameters.yaml``` 
* run an experiments or a single model run using the name reference as : ```"<MODEL_NAME>_myparameters"```

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