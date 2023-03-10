from .schedule import linear_beta_schedule
import torch
from torch import nn
import torch.nn.functional as F
from libs.backbone.unet import Unet
from libs.util.interpolation import spherical_linear_interpolation


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


class DiffusionModel(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()

        self.timesteps = cfg.DM.TIME_STEPS
        self.img_channels = cfg.DATASET.CHANNELS
        self.img_size = cfg.DATASET.IMAGE_SIZE

        # define beta schedule
        self.betas = linear_beta_schedule(timesteps=cfg.DM.TIME_STEPS)

        # define alphas
        self.alphas = 1.0 - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, axis=0)
        self.one_minus_alphas_bar = 1.0 - self.alphas_bar
        self.alphas_bar_prev = F.pad(self.alphas_bar[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_bar = torch.sqrt(self.alphas_bar)
        self.sqrt_one_minus_alphas_bar = torch.sqrt(self.one_minus_alphas_bar)
        self.one_minus_alphas_bar_prev = 1.0 - self.alphas_bar_prev

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * self.one_minus_alphas_bar_prev / self.one_minus_alphas_bar
        )

        # ddim
        self.use_ddim = cfg.DM.USE_DDIM
        self.ddim_eta = cfg.DM.DDIM_VAR_RATIO
        self.ddim_steps = cfg.DM.DDIM_STEPS

        self.denoise_model = Unet(
            dim=self.img_size,
            channels=self.img_channels,
            dim_mults=(
                1,
                2,
                4,
            ),
        )

    def forward(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        # forward
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        # reverse
        predicted_noise = self.denoise_model(x_noisy, t)

        loss = self.loss(noise, predicted_noise)

        return loss

    def loss(self, noise, predicted_noise, loss_type="huber"):
        if loss_type == "l1":
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == "l2":
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()

        return loss

    # forward diffusion (using the nice property)
    @torch.no_grad()
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_bar_t = extract(self.sqrt_alphas_bar, t, x_start.shape)
        sqrt_one_minus_alphas_bar_t = extract(
            self.sqrt_one_minus_alphas_bar, t, x_start.shape
        )

        return sqrt_alphas_bar_t * x_start + sqrt_one_minus_alphas_bar_t * noise

    @torch.no_grad()
    def p_sample(self, x, t, t_index):
        betas_t = self.betas[t_index]
        sqrt_one_minus_alphas_bar_t = self.sqrt_one_minus_alphas_bar[t_index]
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t_index]
        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * self.denoise_model(x, t) / sqrt_one_minus_alphas_bar_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self.posterior_variance[t_index]
            noise = torch.randn_like(x)
            # Algorithm 2 line 4:
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    # Algorithm 2 (including returning all images)
    @torch.no_grad()
    def p_sample_loop(self, shape):
        device = next(self.denoise_model.parameters()).device

        b = shape[0]
        # start from pure noise (for each example in the batch)
        img = torch.randn(shape, device=device)
        imgs = []

        for i in reversed(range(0, self.timesteps)):
            img = self.p_sample(img, torch.full((b,), i, device=device), i)
            imgs.append(img)
        return imgs  # t , batchsize, chennels, h, w

    @torch.no_grad()
    def ddim_sample(self, x, t_prev, t_next):
        sqrt_alphas_bar_tprev = self.sqrt_alphas_bar[t_prev]
        sqrt_one_minus_alphas_bar_tprev = self.sqrt_one_minus_alphas_bar[t_prev]
        noise = torch.randn_like(x)
        pred_noise = self.denoise_model(
            x, torch.full((x.shape[0],), t_prev, device=x.device)
        )
        x0 = (x - sqrt_one_minus_alphas_bar_tprev * pred_noise) / sqrt_alphas_bar_tprev

        if t_next < 0:
            return x0
        else:
            alphas_bar_tprev = self.alphas_bar[t_prev]
            alphas_bar_tnext = self.alphas_bar[t_next]
            one_minus_alphas_bar_tprev = self.one_minus_alphas_bar[t_prev]
            one_minus_alphas_bar_tnext = self.one_minus_alphas_bar[t_next]
            sqrt_alphas_bar_tnext = self.sqrt_alphas_bar[t_next]
            var = (
                one_minus_alphas_bar_tnext
                / one_minus_alphas_bar_tprev
                * (1.0 - alphas_bar_tprev / alphas_bar_tnext)
            )
            sigma = self.ddim_eta * var.sqrt()
            mean = (
                sqrt_alphas_bar_tnext * x0
                + (one_minus_alphas_bar_tnext - sigma**2).sqrt() * pred_noise
            )
            return mean + sigma * noise

    @torch.no_grad()
    def ddim_sample_loop(self, shape):
        device = next(self.denoise_model.parameters()).device
        # start from pure noise (for each example in the batch)
        img = torch.randn(shape, device=device)
        imgs = []

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, self.timesteps - 1, steps=self.ddim_steps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        for time, time_next in time_pairs:
            img = self.ddim_sample(img, time, time_next)
            imgs.append(img)
        return imgs  # t , batchsize, chennels, h, w

    @torch.no_grad()
    def ddim_sample_loop_interpolation(self, shape):
        device = next(self.denoise_model.parameters()).device
        img = spherical_linear_interpolation(shape, device)
        imgs = []
        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, self.timesteps - 1, steps=self.ddim_steps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        for time, time_next in time_pairs:
            img = self.ddim_sample(img, time, time_next)
            imgs.append(img)
        return imgs  # t , batchsize, chennels, h, w

    @torch.no_grad()
    def sample(self, image_size, batch_size=16, channels=3):
        if self.use_ddim:
            return self.ddim_sample_loop(
                shape=(batch_size, channels, image_size, image_size)
            )
        else:
            return self.p_sample_loop(
                shape=(batch_size, channels, image_size, image_size)
            )
