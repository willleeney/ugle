import ugle
import ugle.utils as utils
from ugle.logger import log
from ugle.trainer import MyLibrarySniffingClass
from omegaconf import OmegaConf, DictConfig
import argparse
import psutil
import time


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

    # create trainer object defined in models and init with config
    Trainer = getattr(getattr(ugle.models, cfg.model), f"{cfg.model}_trainer")(cfg)

    # memory profiling max memory requires other class
    if 'memory' in Trainer.cfg.trainer.test_metrics:
        # train model
        start_mem = psutil.virtual_memory().active
        mythread = MyLibrarySniffingClass(Trainer.eval)
        mythread.start()

        delta_mem = 0
        max_memory = 0
        memory_usage_refresh = .001  # Seconds

        while True:
            time.sleep(memory_usage_refresh)
            delta_mem = psutil.virtual_memory().active - start_mem
            if delta_mem > max_memory:
                max_memory = delta_mem
                max_percent = psutil.virtual_memory().percent

            # Check to see if the library call is complete
            if mythread.isShutdown():
                break

        max_memory /= 1024.0 ** 2
        log.info(f"MAX Memory Usage in MB: {max_memory:.2f}")
        log.info(f"Max useage %: {max_percent}")

        results = mythread.results
        results['memory'] = max_memory
    else:
        # train and evaluate model
        results = Trainer.eval()

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='which model to run')
    parser.add_argument('--model', type=str, default='daegc',
                        help='the name of the model to run')
    parser.add_argument('--dataset', type=str, default='cora',
                        help='the name of the dataset to run model on')
    parser.add_argument('--seed', type=str, default=42,
                        help='the number random seed to train on')
    parser.add_argument('--gpu', type=str, default="0",
                        help='the gpu to train on')
    parsed = parser.parse_args()
    study_cfg = OmegaConf.create({"args": {"random_seed": int(parsed.seed)},
                                  "trainer": {"gpu": int(parsed.gpu)}})
    if ugle.utils.is_neural(parsed.model):
        results = neural_run(override_model=parsed.model,
                             override_dataset=parsed.dataset,
                             override_cfg=study_cfg)