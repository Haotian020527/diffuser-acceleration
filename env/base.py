import torch.nn as nn
from omegaconf import DictConfig
from utils.registry import Registry

ENV = Registry('Env')
def create_enviroment(cfg: DictConfig) -> nn.Module:
    """ Create an environment for mobile manipulation tasks.

    Args:
        cfg [DictConfig]: Configuration object containing environment parameters.
        nn.Module: The planning environment instance created according to the configuration.
    
    Return:
        An Env for mobile manipulation tasks.
    """
    return ENV.get(cfg.name)(cfg)