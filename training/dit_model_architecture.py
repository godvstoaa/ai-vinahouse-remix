"""
Diffusion Transformer (DiT) Architecture for Audio Generation
Based on "Scalable Diffusion Models with Transformers" (Peebles & Xie, 2023)
Optimized for 8GB VRAM with gradient checkpointing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class DiTConfig:
    """Configuration for DiT model"""
    # Input dimensions
    input_channels: int = 1       # Mono audio / Mel channels
    latent_dim: int = 64          # VAE latent dimension
    context_length: int = 512     # Sequence length for attention
    
    # Model architecture
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    
    # Conditioning
    bpm_bins: int = 40            # BPM range: 100-180
    key_classes: int = 24         # 12 major + 12 minor
    energy_levels: int = 5        # Low, Medium, High, Build, Drop
    set_phase_classes: int = 4    # warmup, buildup, peak, cooldown
    
    # Diffusion
    num_timesteps: int = 1000
    
    # Optimization
    dropout: float = 0.1
    use_checkpoint: bool = True   # Gradient checkpointing for memory


class TimestepEmbedding(nn.Module):
    """
    Sinusoidal timestep embedding for diffusion timesteps.
    """
    def __init__(self, hidden_size: int, max_period: int = 10000):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_period = max_period
        
        # MLP to project embedding
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
    
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: (batch_size,) tensor of timestep indices
        Returns:
            (batch_size, hidden_size) embedding
        """
        half = self.hidden_size // 2
        freqs = torch.exp(
            -math.log(self.max_period) * 
            torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device) / half
        )
        
        args = timesteps.float().unsqueeze(-1) * freqs.unsqueeze(0)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        return self.mlp(embedding)


class ConditionEmbedding(nn.Module):
    """
    Embedding for conditioning inputs (BPM, key, energy).
    """
    def __init__(self, config: DiTConfig):
        super().__init__()
        self.config = config
        
        # BPM embedding (continuous -> discrete bins)
        self.bpm_embed = nn.Embedding(config.bpm_bins, config.hidden_size // 4)
        
        # Key embedding (24 classes)
        self.key_embed = nn.Embedding(config.key_classes, config.hidden_size // 4)
        
        # Energy embedding (5 levels)
        self.energy_embed = nn.Embedding(config.energy_levels, config.hidden_size // 4)
        
        # Set phase embedding (warmup/buildup/peak/cooldown)
        self.phase_embed = nn.Embedding(config.set_phase_classes, config.hidden_size // 4)
        
        # Combined projection (4 * hidden_size//4 = hidden_size)
        self.proj = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.hidden_size)
        )
    
    def forward(
        self, 
        bpm: torch.Tensor, 
        key: torch.Tensor, 
        energy: torch.Tensor,
        set_phase: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            bpm: (batch_size,) BPM bin indices
            key: (batch_size,) key indices
            energy: (batch_size,) energy level indices
            set_phase: (batch_size,) set phase indices (0=warmup,1=buildup,2=peak,3=cooldown)
        Returns:
            (batch_size, hidden_size) conditioning embedding
        """
        bpm_emb = self.bpm_embed(bpm)
        key_emb = self.key_embed(key)
        energy_emb = self.energy_embed(energy)
        
        if set_phase is not None:
            phase_emb = self.phase_embed(set_phase)
        else:
            # Default to "peak" (index 2) if not provided
            phase_emb = self.phase_embed(torch.full_like(bpm, 2))
        
        combined = torch.cat([bpm_emb, key_emb, energy_emb, phase_emb], dim=-1)
        return self.proj(combined)


class Attention(nn.Module):
    """
    Multi-head self-attention with optional flash attention.
    """
    def __init__(self, config: DiTConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        
        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=False)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)
        
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Flash attention if available
        if hasattr(F, 'scaled_dot_product_attention'):
            attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            if mask is not None:
                attn = attn.masked_fill(mask == 0, float('-inf'))
            attn = F.softmax(attn, dim=-1)
            attn = attn @ v
        
        attn = attn.transpose(1, 2).reshape(B, N, C)
        return self.proj(attn)


class MLP(nn.Module):
    """
    MLP block with GELU activation.
    """
    def __init__(self, config: DiTConfig):
        super().__init__()
        hidden = int(config.hidden_size * config.mlp_ratio)
        self.fc1 = nn.Linear(config.hidden_size, hidden)
        self.fc2 = nn.Linear(hidden, config.hidden_size)
        self.act = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class DiTBlock(nn.Module):
    """
    DiT block with adaptive layer norm (adaLN-Zero).
    """
    def __init__(self, config: DiTConfig):
        super().__init__()
        self.config = config
        
        # Layer norms
        self.norm1 = nn.LayerNorm(config.hidden_size, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(config.hidden_size, elementwise_affine=False)
        
        # Attention and MLP
        self.attn = Attention(config)
        self.mlp = MLP(config)
        
        # Adaptive modulation (adaLN-Zero)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.hidden_size, 6 * config.hidden_size)
        )
        
        # Initialize to zero for residual paths
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
    
    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # Adaptive modulation
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=-1)
        
        # Attention with residual
        x = x + gate_msa.unsqueeze(1) * self.attn(
            self.norm1(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        )
        
        # MLP with residual
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            self.norm2(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        )
        
        return x


class PatchEmbed(nn.Module):
    """
    Convert audio latent to patch embeddings.
    """
    def __init__(self, config: DiTConfig):
        super().__init__()
        self.config = config
        
        # 1D convolution for audio patches
        self.proj = nn.Conv1d(
            config.latent_dim,
            config.hidden_size,
            kernel_size=4,
            stride=2,
            padding=1
        )
        
        # Positional embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, config.context_length, config.hidden_size)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, latent_dim, seq_len) audio latent
        Returns:
            (batch, num_patches, hidden_size) patch embeddings
        """
        # Project patches
        x = self.proj(x)  # (B, hidden_size, seq_len/2)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, hidden_size)
        
        # Add positional embedding
        x = x + self.pos_embed[:, :x.size(1), :]
        
        return x


class FinalLayer(nn.Module):
    """
    Final output layer with adaLN.
    """
    def __init__(self, config: DiTConfig):
        super().__init__()
        self.norm = nn.LayerNorm(config.hidden_size, elementwise_affine=False)
        self.proj = nn.Linear(config.hidden_size, config.latent_dim)
        
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.hidden_size, 2 * config.hidden_size)
        )
        
        # Initialize to zero
        nn.init.constant_(self.proj.weight, 0)
        nn.init.constant_(self.proj.bias, 0)
    
    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return self.proj(x)


class DiT(nn.Module):
    """
    Diffusion Transformer for Audio Generation.
    
    Architecture:
    1. Patch embed audio latent
    2. Add timestep + conditioning embeddings
    3. Pass through DiT blocks
    4. Project to output latent
    """
    
    def __init__(self, config: DiTConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.patch_embed = PatchEmbed(config)
        self.t_embed = TimestepEmbedding(config.hidden_size)
        self.c_embed = ConditionEmbedding(config)
        
        # Blocks
        self.blocks = nn.ModuleList([
            DiTBlock(config) for _ in range(config.num_layers)
        ])
        
        # Output
        self.final_layer = FinalLayer(config)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # VRAM optimization: gradient checkpointing
        self.use_checkpoint = config.use_checkpoint
    
    def _init_weights(self, m):
        """Initialize weights with truncated normal"""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv1d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.trunc_normal_(m.weight, std=0.02)
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        bpm: torch.Tensor,
        key: torch.Tensor,
        energy: torch.Tensor,
        set_phase: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, latent_dim, seq_len) noisy latent
            t: (batch,) timestep
            bpm: (batch,) BPM bin index
            key: (batch,) key index  
            energy: (batch,) energy level index
            set_phase: (batch,) set phase index (optional)
        Returns:
            (batch, latent_dim, seq_len) denoised prediction
        """
        # Patch embed
        x = self.patch_embed(x)  # (B, N, C)
        
        # Embeddings
        t_emb = self.t_embed(t)  # (B, C)
        c_emb = self.c_embed(bpm, key, energy, set_phase)  # (B, C)
        c = t_emb + c_emb  # Combined conditioning
        
        # DiT blocks with optional gradient checkpointing
        for block in self.blocks:
            if self.use_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, c, use_reentrant=False
                )
            else:
                x = block(x, c)
        
        # Output
        x = self.final_layer(x, c)  # (B, N, latent_dim)
        
        # Reshape back to latent shape
        x = x.transpose(1, 2)  # (B, latent_dim, N)
        x = F.interpolate(x, scale_factor=2, mode='linear')  # Upsample
        
        return x
    
    def get_param_count(self) -> dict:
        """Get parameter counts"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'total_params': total,
            'trainable_params': trainable,
            'total_M': total / 1e6,
            'trainable_M': trainable / 1e6
        }


class GaussianDiffusion:
    """
    Gaussian diffusion process for training and sampling.
    """
    
    def __init__(self, config: DiTConfig):
        self.config = config
        self.num_timesteps = config.num_timesteps
        
        # Beta schedule (cosine)
        self.betas = self._cosine_beta_schedule(config.num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # Precompute useful values
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod) / (1.0 - self.alphas_cumprod)
        )
    
    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008) -> torch.Tensor:
        """Cosine schedule for betas"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)
    
    def q_sample(
        self, 
        x_start: torch.Tensor, 
        t: torch.Tensor, 
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward diffusion: add noise at timestep t.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alpha = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise
    
    def p_losses(
        self,
        model: nn.Module,
        x_start: torch.Tensor,
        t: torch.Tensor,
        bpm: torch.Tensor,
        key: torch.Tensor,
        energy: torch.Tensor,
        set_phase: torch.Tensor = None,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Training loss: MSE between predicted and actual noise.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # Add noise
        x_noisy = self.q_sample(x_start, t, noise)
        
        # Predict noise
        model_out = model(x_noisy, t, bpm, key, energy, set_phase)
        
        # MSE loss
        loss = F.mse_loss(model_out, noise)
        
        return loss
    
    @torch.no_grad()
    def p_sample(
        self,
        model: nn.Module,
        x: torch.Tensor,
        t: int,
        bpm: torch.Tensor,
        key: torch.Tensor,
        energy: torch.Tensor,
        set_phase: torch.Tensor = None,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Reverse diffusion: denoise one step.
        """
        t_tensor = torch.full((x.size(0),), t, device=x.device, dtype=torch.long)
        
        # Predict noise
        model_out = model(x, t_tensor, bpm, key, energy, set_phase)
        
        # Compute x_{t-1}
        betas_t = self._extract(self.betas, t_tensor, x.shape)
        sqrt_recip = self._extract(self.sqrt_recip_alphas, t_tensor, x.shape)
        sqrt_one_minus = self._extract(self.sqrt_one_minus_alphas_cumprod, t_tensor, x.shape)
        
        # Predicted x_0
        pred_x0 = sqrt_recip * (x - betas_t * model_out / sqrt_one_minus)
        
        # Add noise (except at t=0)
        if t > 0:
            noise = torch.randn_like(x) * temperature
            posterior_var = self._extract(self.posterior_variance, t_tensor, x.shape)
            pred_x0 = pred_x0 + torch.sqrt(posterior_var) * noise
        
        return pred_x0
    
    @torch.no_grad()
    def p_sample_loop(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        bpm: torch.Tensor,
        key: torch.Tensor,
        energy: torch.Tensor,
        set_phase: torch.Tensor = None,
        temperature: float = 1.0,
        progress: bool = True
    ) -> torch.Tensor:
        """
        Full reverse diffusion: generate samples.
        """
        device = next(model.parameters()).device
        
        # Start from random noise
        x = torch.randn(shape, device=device)
        
        # Iteratively denoise
        timesteps = range(self.num_timesteps - 1, -1, -1)
        if progress:
            from tqdm import tqdm
            timesteps = tqdm(timesteps, desc="Sampling")
        
        for t in timesteps:
            x = self.p_sample(model, x, t, bpm, key, energy, set_phase, temperature)
        
        return x
    
    def _extract(self, arr: torch.Tensor, t: torch.Tensor, shape: Tuple) -> torch.Tensor:
        """Extract values from arr at indices t and reshape to shape."""
        batch_size = t.shape[0]
        out = arr.to(t.device).gather(0, t)
        return out.view(batch_size, *([1] * (len(shape) - 1)))


def create_model(config: Optional[DiTConfig] = None) -> DiT:
    """Create DiT model with default or custom config."""
    if config is None:
        config = DiTConfig()
    return DiT(config)


# Model size variants
def dit_small():
    """DiT-S: ~22M params"""
    return DiT(DiTConfig(hidden_size=384, num_layers=12, num_heads=6))


def dit_base():
    """DiT-B: ~86M params"""
    return DiT(DiTConfig(hidden_size=768, num_layers=12, num_heads=12))


def dit_large():
    """DiT-L: ~300M params (requires more VRAM)"""
    return DiT(DiTConfig(hidden_size=1024, num_layers=24, num_heads=16))


if __name__ == "__main__":
    # Test model
    config = DiTConfig()
    model = create_model(config)
    
    print(f"Model parameters: {model.get_param_count()}")
    
    # Test forward pass
    batch_size = 2
    x = torch.randn(batch_size, config.latent_dim, config.context_length)
    t = torch.randint(0, config.num_timesteps, (batch_size,))
    bpm = torch.randint(0, config.bpm_bins, (batch_size,))
    key = torch.randint(0, config.key_classes, (batch_size,))
    energy = torch.randint(0, config.energy_levels, (batch_size,))
    
    output = model(x, t, bpm, key, energy)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")