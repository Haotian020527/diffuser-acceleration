import pytorch_lightning as pl
from pathlib import Path
from typing import Optional, Dict
from cprint import *
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from datamodule.base import DATAMODULE
from datamodule.dataset.base import DatasetType, create_dataset
from datamodule.misc import collate_fn_general, collate_fn_squeeze_pcd_batch


@DATAMODULE.register()
class MKMotionPolicyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        cfg: DictConfig, 
        slurm: bool, 
        **kwargs: Dict
    ):
        """ MKMotionPolicyDataModule is a PyTorch Lightning DataModule for managing motion policy datasets.
        It handles dataset creation, data loading, and batching for training, validation, and testing phases.
        The class supports different data collate functions based on the scene model and can be configured
        for SLURM or local environments.

        Args:
            cfg [DictConfig]: Configuration object containing dataset and dataloader parameters.
            slurm [bool]: Flag indicating whether to use SLURM-specific data directory.
            **kwargs [Dict]: Additional keyword arguments.

        Return:
            None. This class is used to instantiate a DataModule object for PyTorch Lightning workflows.
        """
        super().__init__()
        self.slurm = slurm
        self.cfg = cfg
        self.data_dir = Path(cfg.data_dir_slurm) if self.slurm else Path(cfg.data_dir)
        self.train_batch_size = cfg.train_batch_size
        self.val_batch_size = cfg.val_batch_size
        self.test_batch_size = cfg.test_batch_size
        self.num_workers = cfg.num_workers
        if ('scene_model_name' in cfg) and (cfg.scene_model_name == 'PointTransformer'):
            self.collate_fn = collate_fn_squeeze_pcd_batch
        else:
            self.collate_fn = collate_fn_general
    
    def setup(self, stage: Optional[str] = None):
        """ A Pytorch Lightning method that is called per-device in when doing distributed training.

        Args:
            stage [Optional[str]]: Indicates whether we are in the training procedure or if we are 
                doing ad-hoc testing.
        """
        if stage == "fit" or stage is None:
            self.data_train = create_dataset(
                cfg=self.cfg.dataset, 
                data_dir=self.data_dir, 
                dataset_type=DatasetType.TRAIN
            )
            self.data_val = create_dataset(
                cfg=self.cfg.dataset, 
                data_dir=self.data_dir, 
                dataset_type=DatasetType.VAL
            )
        if stage == "test":
            self.data_test = create_dataset(
                cfg=self.cfg.dataset, 
                data_dir=self.data_dir, 
                dataset_type=DatasetType.TEST
            )
    
    def train_dataloader(self) -> DataLoader:
        """ A Pytorch lightning method to get the dataloader for training.

        Returns:
            DataLoader -- The training dataloader
        """
        return DataLoader(
            self.data_train,
            self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )
    
    def val_dataloader(self) -> DataLoader:
        """ A Pytorch lightning method to get the dataloader for validation.

        Returns:
            DataLoader -- The validation dataloader
        """
        return DataLoader(
            self.data_val,
            self.val_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        """ A Pytorch lightning method to get the dataloader for testing.

        Returns:
            DataLoader -- The dataloader for testing
        """
        return DataLoader(
            self.data_test,
            self.test_batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )
    
    def get_train_dataloader(self) -> DataLoader:
        """ An external method to get the dataloader for training.

        Returns:
            DataLoader -- The training dataloader
        """
        self.setup("fit")
        return self.train_dataloader()
    
    def get_val_dataloader(self) -> DataLoader:
        """ An external method to get the dataloader for validation.

        Returns:
            DataLoader -- The validation dataloader
        """
        self.setup("fit")
        return self.val_dataloader()
    
    def get_test_dataloader(self) -> DataLoader:
        """ An external method to get the dataloader for testing.

        Returns:
            DataLoader -- The dataloader for testing
        """
        self.setup("test")
        return self.test_dataloader()