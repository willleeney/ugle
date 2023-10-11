import ugle
import ugle.utils as utils
from ugle.logger import log
from omegaconf import OmegaConf, DictConfig, open_dict
import argparse
import time
from os.path import isfile
import pickle
from memory_profiler import memory_usage


def neural_run(override_model: str = None,
               override_dataset: str = None,
               override_cfg: DictConfig = None) -> dict:
    """
    runs a GNN neural experiment
    :param override_model: name of model to override default config
    :param override_dataset: name of the dataset to override default config
    :param override_cfg: override config options
    :return results: results from the study
    """

    # load model config
    cfg = utils.load_model_config(override_model=override_model, override_cfg=override_cfg)
    if override_dataset:
        cfg.dataset = override_dataset

    if cfg.trainer.load_existing_test and 'default' not in override_model:
        # try load the pickle file from previous study 
        hpo_path = f"{cfg.trainer.load_hps_path}{cfg.dataset}_{cfg.model}.pkl"
        if isfile(hpo_path):
            log.info(f'loading hpo args: {hpo_path}')
            previously_found = pickle.load(open(hpo_path, "rb"))
            cfg.previous_results = previously_found.results
        # if this doesn't exist then just use the default parameters
        else: 
            log.info(f'loading default args')
            found_args = OmegaConf.load(f'ugle/configs/models/{cfg.model}/{cfg.model}_default.yaml')
            with open_dict(cfg):
                cfg.args = OmegaConf.merge(cfg.args, found_args)
        cfg.trainer.only_testing = True

    # create trainer object defined in models and init with config
    Trainer = getattr(getattr(ugle.models, cfg.model), f"{cfg.model}_trainer")(cfg)

    start_time = time.time()
    if not cfg.trainer.calc_memory: 
        mem_usage, results = memory_usage((Trainer.eval), retval=True)
        log.info(f"Max Memory Usage {max(mem_usage):.2f}MB")
    else:
        results = Trainer.eval()
    if cfg.trainer.calc_time:
        log.info(f"Total Time for {cfg.model} {cfg.dataset}: {round(time.time() - start_time, 3)}s")

    # cpu_memory_dataset - cpu_memory_start = total used by cpu to load dataset
    # cpu_memory_preprocess - cpu_memory_dataset = total used by cpu to processs dataset
    # active_memory_training - active_memory_start = total used by active to move data onto device + create model on device
    # active_memory_fpass_end - active_memory_fpass = total used by active to do one forward pass
    # active_memory_opt - active_memory_fpass_end = total used by active to compute optimisation of loop 
    # active_memory_opt - active_memory_start = total amount of memory needed by active device

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='which model to run')
    parser.add_argument('--model', type=str, default='daegc',
                        help='the name of the model to run')
    parser.add_argument('--dataset', type=str, default='cora',
                        help='the name of the dataset to run model on')
    parser.add_argument('--seed', type=str, default=42,
                        help='the number random seed to train on')
    parser.add_argument('--max_epoch', type=str, default=500,
                        help='the number of epochs to train on')
    parser.add_argument('--gpu', type=str, default="0",
                        help='the gpu to train on')
    parser.add_argument('--load_existing_test', action='store_true',
                        help='load best parameters available')
    parsed = parser.parse_args()
    study_cfg = OmegaConf.create({"args": {"random_seed": int(parsed.seed),
                                           "max_epoch": int(parsed.max_epoch)},
                                  "trainer": {"gpu": int(parsed.gpu), 
                                              "load_existing_test": parsed.load_existing_test}})
    if ugle.utils.is_neural(parsed.model):
        results = neural_run(override_model=parsed.model,
                             override_dataset=parsed.dataset,
                             override_cfg=study_cfg)