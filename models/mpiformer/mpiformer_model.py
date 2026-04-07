import torch
import transformers
import torch.nn as nn
import pytorch_lightning as pl
from cprint import *
from models.base import MODEL
from omegaconf import DictConfig
from typing import Dict, Optional
from env.agent.mec_kinova import MecKinova
from env.sampler.mk_sampler import MecKinovaSampler
from models.mpiformer.openai_gpt2 import GPT2Model
from models.mpinets.mpinets_model import MPiNetsPointNet
from utils.transform import transform_pointcloud_torch
from utils.meckinova_utils import transform_configuration_torch
from models.mpinets.mpinets_loss import point_clouds_match_loss, sdf_collision_loss
from transformers.optimization import get_cosine_schedule_with_warmup

@MODEL.register()
class MotionPolicyTransformer(pl.LightningModule):
    """ This model uses GPT to model (q_1, o_1, q_2, o_2, ...)
    """
    def __init__(self, cfg: DictConfig, *args, **kwargs):
        """
        Constructs the model
        """
        super().__init__()
        self.d_x = cfg.d_x
        self.lr = cfg.lr
        self.wd = cfg.wd
        self.wr = cfg.wr
        self.context_length = cfg.context_length
        self.embed_dim = cfg.embed_dim
        self.embed_timesteps = cfg.embed_timesteps
        self.n_inner = cfg.n_inner
        self.n_layer = cfg.n_layer
        self.n_head = cfg.n_head
        self.n_positions = cfg.n_positions
        self.activation_function = cfg.activation_function
        self.resid_pdrop = cfg.resid_pdrop
        self.attn_pdrop = cfg.attn_pdrop

        self.train_epoch = cfg.train_epoch
        self.collision_loss = cfg.loss.collision_loss
        self.collision_loss_weight = cfg.loss.collision_loss_weight
        self.point_match_loss = cfg.loss.point_match_loss
        self.point_match_ratio = cfg.loss.point_match_ratio
        self.point_match_loss_weight = cfg.loss.point_match_loss_weight

        self.mk_sampler = None
        self.num_agent_points = cfg.scene_model.num_agent_points

        self.train_dataloader_len = None

        # configure the parameters of transformers
        config = transformers.GPT2Config(
            vocab_size=1, # doesn't matter -- we don't use the vocab
            n_embd=self.embed_dim,
            n_inner=self.n_inner,
            n_layer=self.n_layer,
            n_head=self.n_head,
            n_positions=self.n_positions,
            activation_function=self.activation_function,
            resid_pdrop=self.resid_pdrop,
            attn_pdrop=self.attn_pdrop,
        )

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.pc_encoder = MPiNetsPointNet()
        self.cfg_encoder = nn.Sequential(
            nn.Linear(self.d_x, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
        )
        self.transformer = GPT2Model(config)
        self.embed_timestep = nn.Embedding(self.embed_timesteps, self.embed_dim)
        self.embed_feat = nn.Linear(2048 + 64, self.embed_dim)
        # 2048 is the dimension of point cloud encoder output
        self.embed_obser = torch.nn.Linear(2048, self.embed_dim)
        self.embed_cfg = torch.nn.Linear(self.d_x, self.embed_dim)
        self.embed_ln = nn.LayerNorm(self.embed_dim)

        self.predict_cfg = nn.Sequential(
            nn.Linear(self.embed_dim, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, self.d_x),
        )

    def configure_optimizers(self):
        """ A standard method in PyTorch lightning to set the optimizer
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.wd
        )

        num_training_steps = self.train_epoch * self.train_dataloader_len
        num_warmup_steps = int(self.wr * num_training_steps)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def forward(
            self, obsers: torch.Tensor, cfgs: torch.Tensor, 
            timesteps: torch.Tensor, 
            attention_mask: Optional[torch.Tensor]=None
        ) -> torch.Tensor:  # type: ignore[override]
        """ Passes data through the network to produce an output.
        """
        B, C, N = obsers.shape[0], obsers.shape[1], obsers.shape[2] # [batch_size, context_length, points_num]

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((B, C), dtype=torch.long)

        ## embed each modality with a different head
        pc_encoder_output = self.pc_encoder(obsers.reshape(-1, N, 4)).reshape(B, C, -1)
        cfg_encoder_output = self.cfg_encoder(cfgs.reshape(-1, self.d_x)).reshape(B, C, -1)

        # embed point cloud and configuration, and plus timesteps embeddings
        time_embeddings = self.embed_timestep(timesteps)
        embeddings = self.embed_feat(torch.cat((pc_encoder_output, cfg_encoder_output), dim=-1))
        embeddings = embeddings + time_embeddings

        # this makes the sequence look like (q_1, o_1, a_1, q_2, o_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        inputs_embeds = self.embed_ln(embeddings)

        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )

        x = transformer_outputs['last_hidden_state']

        # get predicted configurations
        return self.predict_cfg(x)

    def sample(self, q: torch.Tensor) -> torch.Tensor:
        """ Samples a point cloud from the surface of all the robot's links.
        
        Args:
            q [torch.Tensor]: Batched configuration in joint space.

        Return:
            torch.Tensor: Batched point cloud of size [B, self.num_robot_points, 3], sampled from the surface of all the robot's links.
        """
        assert self.mk_sampler is not None
        return self.mk_sampler.sample(q)

    def training_step(  # type: ignore[override]
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """ This function is automatically called by PyTorch Lightning during training. 
        It performs the forward pass, computes various losses (such as collision loss 
        and point match loss), applies necessary transformations to the model outputs 
        and ground truth, and logs relevant metrics. The final weighted loss is returned 
        for backpropagation.

        Args:
            batch [Dict[str, torch.Tensor]]: 
                A batch of data from the data loader, containing input features, ground truth, 
                transformation matrices, SDF values, and other necessary information. All tensors 
                are expected to be on the correct device.
            batch_idx [int]: 
                The index of the current batch (not used in this function).

        Return:
            torch.Tensor: 
                The overall weighted loss computed from the batch, used for backpropagation.
        """
        if self.mk_sampler is None:
            self.mk_sampler = MecKinovaSampler(self.device, self.num_agent_points, use_cache=True)

        xyz_sq, q_sq, timesteps, attention_masks, s_sq = batch['xyz_sq'], batch['configuration_sq'], \
            batch['timesteps'], batch['attention_masks'], batch['supervision_sq']

        # [batch_size, context_length, points_num, agent_dof]
        B, C, N, D = xyz_sq.shape[0], xyz_sq.shape[1], xyz_sq.shape[2], self.d_x

        ## model output
        q_sq_hat_norm = torch.clamp(self(xyz_sq, q_sq, timesteps, attention_masks), min=-1, max=1) # [B, C, D]
        q_sq_hat_unnorm = MecKinova.unnormalize_joints(q_sq_hat_norm) # [B, C, D]
        s_sq_norm = torch.clamp(s_sq, min=-1, max=1) # [B, C, D]
        s_sq_unnorm = MecKinova.unnormalize_joints(s_sq_norm) # [B, C, D]
        # we made sure we train with a mask of one
        q_sq_hat_unnorm = q_sq_hat_unnorm.reshape(-1, D) # [B×C, D]
        s_sq_unnorm = s_sq_unnorm.reshape(-1, D) # [B×C, D]

        ## transform q_hat_unnorm and s_unnorm to agent initial frame
        trans_mats = batch['trans_mat'] # [B, 4, 4]
        trans_mats_inv = torch.inverse(trans_mats) # [B, 4, 4]
        trans_mats_sq_inv = trans_mats_inv.unsqueeze(1).repeat(1, C, 1, 1).reshape(-1, 4, 4) # [B×C, 4, 4]
        rot_angles = batch['rot_angle'] # [B]
        rot_angles_inv = -rot_angles # [B]
        rot_angles_sq_inv = rot_angles_inv.unsqueeze(1).repeat(1, C).reshape(-1) # [B×C]
        q_sq_hat_unnorm = transform_configuration_torch(q_sq_hat_unnorm, trans_mats_sq_inv, rot_angles_sq_inv)
        s_sq_unnorm = transform_configuration_torch(s_sq_unnorm, trans_mats_sq_inv, rot_angles_sq_inv)

        ## compute loss
        loss = 0
        T_aw = batch['T_aw'] # [B, 4, 4]
        T_aw_sq = T_aw.unsqueeze(1).repeat(1, C, 1, 1).reshape(-1, 4, 4) # [B×C, 4, 4]
        # the normalized scene SDF value in world frame, ranging in [-1, 1]
        sdf_norm_value = batch['sdf_norm_value'] # [B, grid_num, grid_num, grid_num]
        resolution = sdf_norm_value.shape[1]
        sdf_norm_value_sq = sdf_norm_value.unsqueeze(1).repeat(1, C, 1, 1, 1). \
            reshape(-1, resolution, resolution, resolution) # [B×C, grid_num, grid_num, grid_num]
        # the scene mesh center in world frame
        scene_mesh_center = batch['scene_mesh_center'] # [B, 3]
        scene_mesh_center_sq = scene_mesh_center.unsqueeze(1).repeat(1, C, 1).reshape(-1, 3) # [B×C, 3]
        # the scene mesh scale for SDF calculation
        scene_mesh_scale = batch['scene_mesh_scale'] # [B]
        scene_mesh_scale_sq = scene_mesh_scale.unsqueeze(1).repeat(1, C).reshape(-1) # [B×C]
        # compute and process agent point clouds
        agent_pcs_a = self.sample(q_sq_hat_unnorm) # [B×C, N, 3]
        agent_pcs_w = transform_pointcloud_torch(agent_pcs_a, T_aw_sq) # [B×C, N, 3]
        norm_agent_pcs_w = (agent_pcs_w - scene_mesh_center_sq.unsqueeze(1)) * \
            scene_mesh_scale_sq.unsqueeze(1).unsqueeze(1)

        # compute collision loss
        if self.collision_loss:
            collision_loss = sdf_collision_loss(
                agent_pcs=norm_agent_pcs_w, 
                sdf_norm_values=sdf_norm_value_sq
            )
            self.log('collision_loss', collision_loss)
            loss += self.collision_loss_weight * collision_loss

        # compute point match loss
        if self.point_match_loss:
            # NOTE：Since the kinematic chain error of the agent is transmitted from the base link to 
            # the end effector link, when calculating point match loss, the weight calculated by the 
            # link at the end of the agent should be greater.
            predicted_base_pcs = self.mk_sampler.sample_base(q_sq_hat_unnorm)
            predicted_arm_pcs = self.mk_sampler.sample_arm(q_sq_hat_unnorm)
            predicted_gripper_pcs = self.mk_sampler.sample_gripper(q_sq_hat_unnorm)
            target_base_pcs = self.mk_sampler.sample_base(s_sq_unnorm)
            target_arm_pcs = self.mk_sampler.sample_arm(s_sq_unnorm)
            target_gripper_pcs = self.mk_sampler.sample_gripper(s_sq_unnorm)

            point_match_loss = (
                point_clouds_match_loss(predicted_base_pcs, target_base_pcs) * self.point_match_ratio[0] + \
                point_clouds_match_loss(predicted_arm_pcs, target_arm_pcs) * self.point_match_ratio[1] + \
                point_clouds_match_loss(predicted_gripper_pcs, target_gripper_pcs) * self.point_match_ratio[2]
            ) / sum(self.point_match_ratio)
            self.log('point_match_loss', point_match_loss)
            loss += self.point_match_loss_weight * point_match_loss

        self.log('val_loss', loss)
        return loss

    def validation_step(  # type: ignore[override]
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """ This is a Pytorch Lightning function run automatically across devices
        during the validation loop.
        """
        pass