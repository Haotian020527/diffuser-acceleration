import pytorch_lightning as pl
from typing import Dict
from utils.registry import Registry

DATAMODULE = Registry('datamodule')

def create_datamodule(cfg: dict, slurm: bool, **kwargs: Dict) -> pl.LightningDataModule:
    """ Create a `torch.utils.data.Dataset` object from configuration.

    Args:
        cfg [dict]: Configuration object containing dataset settings.
        slurm [bool]: Indicates whether the code is running on a SLURM platform, used to specify the data path.
        **kwargs [Dict]: Additional keyword arguments for the datamodule.
    
    Return:
        A Dataset object that has loaded the designated dataset.
    """
    return DATAMODULE.get(cfg.name)(cfg, slurm, **kwargs)
