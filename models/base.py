import torch.nn as nn
from typing import Dict, List, Optional
from omegaconf import DictConfig
from utils.registry import Registry
from models.optimizer.optimizer import Optimizer
from models.planner.planner import Planner

MODEL = Registry('Model')
DIFFUSER = Registry('Diffuser')
LANGEVIN_DIFFUSION = Registry('Langevin_Diffusion')
OPTIMIZER = Registry('Optimizer')
PLANNER = Registry('Planner')

def create_model(cfg: DictConfig, ckpt_path: Optional[str]=None, *args: List, **kwargs: Dict) -> nn.Module:
    """ Create a generative model and return it.
    If 'diffuser' in cfg, this function will call `create_diffuser` function to create a diffusion model.
    Otherwise, this function will create other generative models, e.g., cvae.

    Args:
        cfg: configuration object, the global configuration.
    
    Return:
        A generative model.
    """
    if 'diffuser' in cfg:
        return create_diffuser(cfg, ckpt_path, *args, **kwargs)

    if ckpt_path is not None:
        return MODEL.get(cfg.model.name).load_from_checkpoint(
            checkpoint_path=ckpt_path, **{"cfg": cfg.model},
        )
    return MODEL.get(cfg.model.name)(cfg.model, *args, **kwargs)

def create_diffuser(cfg: DictConfig, ckpt_path: Optional[str]=None, *args: List, **kwargs: Dict) -> nn.Module:
    """ Create a diffuser model, first create a eps_model from model config, then create a diffusion model 
    and use the eps_model as input.

    Args:
        cfg: configuration object.
    
    Return:
        A diffusion model.
    """
    diffuser_name = cfg.diffuser.name

    pose_eps_model = None
    joint_eps_model = None
    if diffuser_name in ('ConsistencyCoupledKinematicsDiffuser', 'CoKinMoEDiffuser'):
        # CoKin dual-space diffuser can optionally use dedicated model cfgs:
        # - cfg.pose_model
        # - cfg.joint_model
        pose_model_cfg = cfg.pose_model if 'pose_model' in cfg else cfg.model
        joint_model_cfg = cfg.joint_model if 'joint_model' in cfg else cfg.model
        pose_eps_model = MODEL.get(pose_model_cfg.name)(pose_model_cfg, *args, **kwargs)
        joint_eps_model = MODEL.get(joint_model_cfg.name)(joint_model_cfg, *args, **kwargs)
        # Keep compatibility with diffuser signature `(eps_model, cfg, has_obser, ...)`.
        eps_model = pose_eps_model
    else:
        # Default: single eps model (original M2Diffuser behavior).
        eps_model = MODEL.get(cfg.model.name)(cfg.model, *args, **kwargs)

    ## if the task has observation, then pass it to diffuser
    has_obser = cfg.task.has_observation if 'has_observation' in cfg.task else False
    
    if ckpt_path is not None:
        diffuser  = DIFFUSER.get(cfg.diffuser.name).load_from_checkpoint(
            checkpoint_path=ckpt_path,
            **{
                "eps_model": eps_model,
                "cfg": cfg.diffuser,
                "has_obser": has_obser,
                "pose_eps_model": pose_eps_model,
                "joint_eps_model": joint_eps_model,
            }
        )
    else:
        diffuser  = DIFFUSER.get(cfg.diffuser.name)(
            eps_model,
            cfg.diffuser,
            has_obser,
            pose_eps_model=pose_eps_model,
            joint_eps_model=joint_eps_model,
            *args,
            **kwargs
        )

    ## if optimizer is in cfg, then load it and pass it to diffuser
    if 'optimizer' in cfg:
        optimizer = create_optimizer(cfg.optimizer, *args, **kwargs)
        diffuser.set_optimizer(optimizer)
    
    ## if planner is in cfg, then load it and pass it to diffuser
    if 'planner' in cfg:
        planner = create_planner(cfg.planner, *args, **kwargs)
        diffuser.set_planner(planner)

    return diffuser

def create_optimizer(cfg: DictConfig, *args: List, **kwargs: Dict) -> Optimizer:
    """ Create a optimizer for constrained sampling.

    Args:
        cfg: configuration object.
    
    Return:
        A optimizer used for guided sampling.
    """
    if cfg is None:
        return None
    
    return OPTIMIZER.get(cfg.name)(cfg, *args, **kwargs)

def create_planner(cfg: DictConfig, *args: List, **kwargs: Dict) -> Planner:
    """ Create a planner for constrained sampling.

    Args:
        cfg: configuration object.
        
    Return:
        A planner used for guided sampling.
    """
    if cfg is None:
        return None
    
    return PLANNER.get(cfg.name)(cfg, *args, **kwargs)
