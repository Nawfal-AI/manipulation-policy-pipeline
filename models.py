"""Policy architectures: a Behavior Cloning baseline and a simplified Diffusion Policy."""
import torch
import torch.nn as nn

OBS_DIM = 8
ACTION_DIM = 2


class BCPolicy(nn.Module):
    """Simple feed-forward Behavior Cloning policy: obs -> action, trained with MSE."""

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, ACTION_DIM),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self.forward(obs)


class SinusoidalTimeEmbedding(nn.Module):
    """Standard sinusoidal embedding for the diffusion timestep, as in DDPM."""

    def __init__(self, dim: int = 32):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(-torch.arange(half, device=device) * (torch.log(torch.tensor(10000.0)) / (half - 1)))
        args = t.float()[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class DiffusionPolicy(nn.Module):
    """
    A simplified conditional Diffusion Policy (DDPM-style), following the core
    idea of Chi et al. 2023: instead of regressing directly to an action (which
    collapses multimodal expert behavior to its mean, a known BC failure mode),
    train a denoiser that predicts the noise added to an action conditioned on
    the observation, then sample actions at inference time via iterative
    denoising. Simplified here: no receding-horizon action chunking, single-step
    actions only, small MLP denoiser instead of a U-Net (appropriate for this
    task's 8-dim observation / 2-dim action, not for image observations).
    """

    def __init__(self, hidden_dim: int = 128, time_dim: int = 32, num_timesteps: int = 50):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM + ACTION_DIM + time_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, ACTION_DIM),
        )
        # Linear beta schedule (standard DDPM)
        betas = torch.linspace(1e-4, 0.02, num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

    def predict_noise(self, obs: torch.Tensor, noisy_action: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_embed = self.time_embed(t)
        x = torch.cat([obs, noisy_action, t_embed], dim=-1)
        return self.net(x)

    def training_loss(self, obs: torch.Tensor, clean_action: torch.Tensor) -> torch.Tensor:
        batch_size = obs.shape[0]
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=obs.device)
        noise = torch.randn_like(clean_action)
        sqrt_alpha_cumprod = self.alphas_cumprod[t].sqrt().unsqueeze(-1)
        sqrt_one_minus_alpha_cumprod = (1 - self.alphas_cumprod[t]).sqrt().unsqueeze(-1)
        noisy_action = sqrt_alpha_cumprod * clean_action + sqrt_one_minus_alpha_cumprod * noise
        predicted_noise = self.predict_noise(obs, noisy_action, t)
        return nn.functional.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """Sample an action via the reverse diffusion process (DDPM ancestral sampling)."""
        self.eval()
        batch_size = obs.shape[0]
        action = torch.randn(batch_size, ACTION_DIM, device=obs.device)
        for t_idx in reversed(range(self.num_timesteps)):
            t = torch.full((batch_size,), t_idx, device=obs.device, dtype=torch.long)
            predicted_noise = self.predict_noise(obs, action, t)
            alpha = self.alphas[t_idx]
            alpha_cumprod = self.alphas_cumprod[t_idx]
            beta = self.betas[t_idx]
            mean = (1 / alpha.sqrt()) * (action - (beta / (1 - alpha_cumprod).sqrt()) * predicted_noise)
            if t_idx > 0:
                noise = torch.randn_like(action)
                action = mean + beta.sqrt() * noise
            else:
                action = mean
        return action
