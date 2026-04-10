import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple
from einops import rearrange
from omegaconf import DictConfig
from models.model.utils import timestep_embedding
from models.model.utils import ResBlock, SpatialTransformer
from models.model.scene_model import create_scene_model
from models.base import MODEL

@MODEL.register()
class UNetModel(nn.Module):
    def __init__(self, cfg: DictConfig, slurm: bool, *args, **kwargs) -> None:
        super(UNetModel, self).__init__()

        self.d_x = cfg.d_x
        self.d_model = cfg.d_model
        self.nblocks = cfg.nblocks
        self.resblock_dropout = cfg.resblock_dropout
        self.transformer_num_heads = cfg.transformer_num_heads
        self.transformer_dim_head = cfg.transformer_dim_head
        self.transformer_dropout = cfg.transformer_dropout
        self.transformer_depth = cfg.transformer_depth
        self.transformer_mult_ff = cfg.transformer_mult_ff
        self.context_dim = cfg.context_dim
        self.use_position_embedding = cfg.use_position_embedding

        ## create scene model from config
        self.scene_model_name = cfg.scene_model.name # PointTransformer
        scene_model_in_dim = 3 + int(cfg.scene_model.use_color) * 3 + int(cfg.scene_model.use_normal) * 3
        if cfg.scene_model.name == 'PointNet':
            scene_model_args = {'c': scene_model_in_dim, 'num_points': cfg.scene_model.num_points,
                                'num_tokens': cfg.scene_model.num_tokens}
        else:
            scene_model_args = {'c': scene_model_in_dim, 'num_points': cfg.scene_model.num_points}
        self.scene_model = create_scene_model(cfg.scene_model.name, **scene_model_args)
        ## load pretrained weights
        weight_path = cfg.scene_model.pretrained_weights_slurm if slurm else cfg.scene_model.pretrained_weights
        if weight_path is not None:
            self.scene_model.load_pretrained_weight(weigth_path=weight_path)
        if cfg.freeze_scene_model:
            for p in self.scene_model.parameters():
                p.requires_grad_(False)

        time_embed_dim = self.d_model * cfg.time_embed_mult
        self.time_embed = nn.Sequential(
            nn.Linear(self.d_model, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        self.in_layers = nn.Sequential(
            nn.Conv1d(self.d_x, self.d_model, 1)
        )

        self.layers = nn.ModuleList()
        for i in range(self.nblocks):
            self.layers.append(
                ResBlock(
                    self.d_model,
                    time_embed_dim,
                    self.resblock_dropout,
                    self.d_model,
                )
            )
            self.layers.append(
                SpatialTransformer(
                    self.d_model, 
                    self.transformer_num_heads, 
                    self.transformer_dim_head, 
                    depth=self.transformer_depth,
                    dropout=self.transformer_dropout,
                    mult_ff=self.transformer_mult_ff,
                    context_dim=self.context_dim,
                )
            )
        
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, self.d_model),
            nn.SiLU(),
            nn.Conv1d(self.d_model, self.d_x, 1),
        )
        
    def forward(self, x_t: torch.Tensor, ts: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """ Apply the model to an input batch.

        Args:
            x_t: the input data, <B, C> or <B, L, C>
            ts: timestep, 1-D batch of timesteps
            cond: condition feature
        
        Return:
            the denoised target data, i.e., $x_{t-1}$.
        """
        in_shape = len(x_t.shape)
        if in_shape == 2:
            x_t = x_t.unsqueeze(1)
        assert len(x_t.shape) == 3

        ## time embedding
        t_emb = timestep_embedding(ts, self.d_model)
        t_emb = self.time_embed(t_emb) 
        h = rearrange(x_t, 'b l c -> b c l') 
        
        h = self.in_layers(h) # <B, d_model, L>

        ## prepare position embedding for input x
        if self.use_position_embedding:
            B, DX, TX = h.shape
            pos_Q = torch.arange(TX, dtype=h.dtype, device=h.device)
            pos_embedding_Q = timestep_embedding(pos_Q, DX) # <L, d_model>
            h = h + pos_embedding_Q.permute(1, 0) # <B, d_model, L>

        for i in range(self.nblocks):
            h = self.layers[i * 2 + 0](h, t_emb)
            h = self.layers[i * 2 + 1](h, context=cond)
        h = self.out_layers(h)
        h = rearrange(h, 'b c l -> b l c')

        ## reverse to original shape
        if in_shape == 2:
            h = h.squeeze(1)

        return h

    def condition(self, data: Dict) -> torch.Tensor:
        """ Obtain scene feature with scene model.

        Args:
            data: dataloader-provided data

        Return:
            Condition feature.
        """
        if self.scene_model_name == 'PointTransformer':
            b = data['offset'].shape[0]
            pos, feat, offset = data['pos'], data['feat'], data['offset']
            p5, x5, o5 = self.scene_model((pos, feat, offset))
            scene_feat = rearrange(x5, '(b n) c -> b n c', b=b, n=self.scene_model.num_groups)
        elif self.scene_model_name == 'PointNet':
            b = data['pos'].shape[0]
            pos = data['pos'].to(torch.float32)
            scene_feat = self.scene_model(pos).reshape(b, self.scene_model.num_groups, -1)
        else:
            raise Exception('Unexcepted scene model.')

        return scene_feat


class LoRALinear(nn.Module):
    """Wrap a dense linear layer with trainable low-rank residual."""

    def __init__(
        self,
        linear: nn.Linear,
        rank: int,
        alpha: float = 1.0,
        dropout: float = 0.0,
        freeze_base: bool = True,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be > 0, got {rank}")
        self.linear = linear
        self.rank = rank
        self.scaling = alpha / float(rank)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        if freeze_base:
            for p in self.linear.parameters():
                p.requires_grad_(False)

        self.lora_A = nn.Linear(self.linear.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, self.linear.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling


class TemporalAdapter(nn.Module):
    """A lightweight residual adapter operating on temporal hidden states."""

    def __init__(self, channels: int, bottleneck: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.down = nn.Conv1d(channels, bottleneck, kernel_size=1)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.up = nn.Conv1d(bottleneck, channels, kernel_size=1)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.dropout(self.act(self.down(x))))


class FrequencySplitAdapter(nn.Module):
    """Frequency-aware temporal adapter with low/high-band residual paths."""

    def __init__(
        self,
        channels: int,
        low_rank: int,
        high_rank: int,
        low_freq_ratio: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if low_rank <= 0 and high_rank <= 0:
            raise ValueError("At least one of low_rank/high_rank must be > 0.")
        if not (0.0 < low_freq_ratio <= 1.0):
            raise ValueError(f"low_freq_ratio must be in (0, 1], got {low_freq_ratio}")

        self.low_freq_ratio = float(low_freq_ratio)
        self.low_adapter = (
            TemporalAdapter(channels, low_rank, dropout) if low_rank > 0 else None
        )
        self.high_adapter = (
            TemporalAdapter(channels, high_rank, dropout) if high_rank > 0 else None
        )

    def _split_low_high(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, C, T]
        seq_len = x.shape[-1]
        if seq_len <= 2:
            return x, torch.zeros_like(x)

        # Stable low-pass proxy in time domain (avoids cuFFT mixed-precision issues).
        kernel = int(round(1.0 / max(self.low_freq_ratio, 1e-3)))
        kernel = max(3, kernel)
        if kernel % 2 == 0:
            kernel += 1
        kernel = min(kernel, seq_len if seq_len % 2 == 1 else max(3, seq_len - 1))
        pad = kernel // 2

        x_f = x.to(torch.float32)
        low = F.avg_pool1d(x_f, kernel_size=kernel, stride=1, padding=pad).to(dtype=x.dtype)
        high = x - low
        return low, high

    def low_frequency_component(self, x: torch.Tensor) -> torch.Tensor:
        low, _ = self._split_low_high(x)
        return low

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        low, high = self._split_low_high(x)
        low_res = self.low_adapter(low) if self.low_adapter is not None else torch.zeros_like(x)
        high_res = self.high_adapter(high) if self.high_adapter is not None else torch.zeros_like(x)

        # Penalize cross-band leakage in adapter outputs.
        low_of_high = self.low_frequency_component(high_res)
        high_of_low = high_res.new_zeros(high_res.shape)
        _, high_of_low = self._split_low_high(low_res)
        leak_loss = low_of_high.pow(2).mean() + high_of_low.pow(2).mean()
        return low_res + high_res, leak_loss


@MODEL.register()
class AdapterUNetModel(UNetModel):
    """Plain full-band adapter baseline on top of frozen dense UNet."""

    def __init__(self, cfg: DictConfig, slurm: bool, *args, **kwargs) -> None:
        super().__init__(cfg, slurm, *args, **kwargs)
        adapter_rank = int(cfg.get("adapter_rank", 16))
        adapter_dropout = float(cfg.get("adapter_dropout", 0.0))
        freeze_backbone = bool(cfg.get("freeze_backbone", True))
        if adapter_rank <= 0:
            raise ValueError(f"adapter_rank must be > 0, got {adapter_rank}")

        self.adapters = nn.ModuleList(
            [TemporalAdapter(self.d_model, adapter_rank, adapter_dropout) for _ in range(self.nblocks)]
        )

        if freeze_backbone:
            for p in self.parameters():
                p.requires_grad_(False)
            for p in self.adapters.parameters():
                p.requires_grad_(True)

    def forward(self, x_t: torch.Tensor, ts: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        in_shape = len(x_t.shape)
        if in_shape == 2:
            x_t = x_t.unsqueeze(1)
        assert len(x_t.shape) == 3

        t_emb = timestep_embedding(ts, self.d_model)
        t_emb = self.time_embed(t_emb)
        h = rearrange(x_t, 'b l c -> b c l')
        h = self.in_layers(h)

        if self.use_position_embedding:
            _, DX, TX = h.shape
            pos_Q = torch.arange(TX, dtype=h.dtype, device=h.device)
            pos_embedding_Q = timestep_embedding(pos_Q, DX)
            h = h + pos_embedding_Q.permute(1, 0)

        for i in range(self.nblocks):
            h = self.layers[i * 2 + 0](h, t_emb)
            h = self.layers[i * 2 + 1](h, context=cond)
            h = h + self.adapters[i](h)
        h = self.out_layers(h)
        h = rearrange(h, 'b c l -> b l c')

        if in_shape == 2:
            h = h.squeeze(1)
        return h


@MODEL.register()
class FreqAdapterUNetModel(UNetModel):
    """Branch-asymmetric frequency adapter on top of frozen dense UNet."""

    def __init__(self, cfg: DictConfig, slurm: bool, *args, **kwargs) -> None:
        super().__init__(cfg, slurm, *args, **kwargs)

        low_rank = int(cfg.get("low_band_rank", 16))
        high_rank = int(cfg.get("high_band_rank", 0))
        low_freq_ratio = float(cfg.get("low_freq_ratio", 0.25))
        adapter_dropout = float(cfg.get("adapter_dropout", 0.0))
        freeze_backbone = bool(cfg.get("freeze_backbone", True))

        self.freq_adapters = nn.ModuleList(
            [
                FrequencySplitAdapter(
                    channels=self.d_model,
                    low_rank=low_rank,
                    high_rank=high_rank,
                    low_freq_ratio=low_freq_ratio,
                    dropout=adapter_dropout,
                )
                for _ in range(self.nblocks)
            ]
        )
        self.latest_band_leak_loss = torch.tensor(0.0)

        if freeze_backbone:
            for p in self.parameters():
                p.requires_grad_(False)
            for p in self.freq_adapters.parameters():
                p.requires_grad_(True)

    def _forward_impl(
        self,
        x_t: torch.Tensor,
        ts: torch.Tensor,
        cond: torch.Tensor,
        use_adapters: bool,
    ) -> torch.Tensor:
        in_shape = len(x_t.shape)
        if in_shape == 2:
            x_t = x_t.unsqueeze(1)
        assert len(x_t.shape) == 3

        t_emb = timestep_embedding(ts, self.d_model)
        t_emb = self.time_embed(t_emb)
        h = rearrange(x_t, "b l c -> b c l")
        h = self.in_layers(h)

        if self.use_position_embedding:
            _, DX, TX = h.shape
            pos_Q = torch.arange(TX, dtype=h.dtype, device=h.device)
            pos_embedding_Q = timestep_embedding(pos_Q, DX)
            h = h + pos_embedding_Q.permute(1, 0)

        band_leak_terms = []
        for i in range(self.nblocks):
            h = self.layers[i * 2 + 0](h, t_emb)
            h = self.layers[i * 2 + 1](h, context=cond)
            if use_adapters:
                delta, leak = self.freq_adapters[i](h)
                h = h + delta
                band_leak_terms.append(leak)

        if use_adapters and band_leak_terms:
            self.latest_band_leak_loss = torch.stack(band_leak_terms).mean()
        else:
            self.latest_band_leak_loss = h.new_zeros(())

        h = self.out_layers(h)
        h = rearrange(h, "b c l -> b l c")
        if in_shape == 2:
            h = h.squeeze(1)
        return h

    def forward(self, x_t: torch.Tensor, ts: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x_t=x_t, ts=ts, cond=cond, use_adapters=True)

    def forward_backbone(self, x_t: torch.Tensor, ts: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x_t=x_t, ts=ts, cond=cond, use_adapters=False)

    def low_frequency_component(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim != 3:
            raise ValueError(f"Expected [B, L, C] or [B, C], got shape={tuple(x.shape)}")
        x_ch = rearrange(x, "b l c -> b c l")
        low = self.freq_adapters[0].low_frequency_component(x_ch)
        low = rearrange(low, "b c l -> b l c")
        return low


@MODEL.register()
class LoRAUNetModel(UNetModel):
    """Rank-matched LoRA baseline for dense UNet attention layers."""

    def __init__(self, cfg: DictConfig, slurm: bool, *args, **kwargs) -> None:
        super().__init__(cfg, slurm, *args, **kwargs)
        self.lora_rank = int(cfg.get("lora_rank", 8))
        self.lora_alpha = float(cfg.get("lora_alpha", 16.0))
        self.lora_dropout = float(cfg.get("lora_dropout", 0.0))
        self.freeze_backbone = bool(cfg.get("freeze_backbone", True))
        self.lora_targets = list(cfg.get("lora_targets", ["to_q", "to_k", "to_v", "to_out"]))

        if self.lora_rank <= 0:
            raise ValueError(f"lora_rank must be > 0, got {self.lora_rank}")

        self._inject_lora(
            rank=self.lora_rank,
            alpha=self.lora_alpha,
            dropout=self.lora_dropout,
            targets=self.lora_targets,
            freeze_base=self.freeze_backbone,
        )

        if self.freeze_backbone:
            for p in self.parameters():
                p.requires_grad_(False)
            for module in self.modules():
                if isinstance(module, LoRALinear):
                    for p in module.lora_A.parameters():
                        p.requires_grad_(True)
                    for p in module.lora_B.parameters():
                        p.requires_grad_(True)

    def _inject_lora(
        self,
        rank: int,
        alpha: float,
        dropout: float,
        targets: List[str],
        freeze_base: bool,
    ) -> None:
        for layer in self.layers:
            if not isinstance(layer, SpatialTransformer):
                continue
            for block in layer.transformer_blocks:
                if "to_q" in targets and isinstance(block.attn1.to_q, nn.Linear):
                    block.attn1.to_q = LoRALinear(block.attn1.to_q, rank, alpha, dropout, freeze_base)
                if "to_k" in targets and isinstance(block.attn1.to_k, nn.Linear):
                    block.attn1.to_k = LoRALinear(block.attn1.to_k, rank, alpha, dropout, freeze_base)
                if "to_v" in targets and isinstance(block.attn1.to_v, nn.Linear):
                    block.attn1.to_v = LoRALinear(block.attn1.to_v, rank, alpha, dropout, freeze_base)
                if "to_out" in targets and isinstance(block.attn1.to_out[0], nn.Linear):
                    block.attn1.to_out[0] = LoRALinear(block.attn1.to_out[0], rank, alpha, dropout, freeze_base)

                if "to_q" in targets and isinstance(block.attn2.to_q, nn.Linear):
                    block.attn2.to_q = LoRALinear(block.attn2.to_q, rank, alpha, dropout, freeze_base)
                if "to_k" in targets and isinstance(block.attn2.to_k, nn.Linear):
                    block.attn2.to_k = LoRALinear(block.attn2.to_k, rank, alpha, dropout, freeze_base)
                if "to_v" in targets and isinstance(block.attn2.to_v, nn.Linear):
                    block.attn2.to_v = LoRALinear(block.attn2.to_v, rank, alpha, dropout, freeze_base)
                if "to_out" in targets and isinstance(block.attn2.to_out[0], nn.Linear):
                    block.attn2.to_out[0] = LoRALinear(block.attn2.to_out[0], rank, alpha, dropout, freeze_base)
