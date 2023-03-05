from libs.schedule import linear_beta_schedule
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from config.default import config
from libs.unet import Unet


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


class DiffusionModel(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()

        self.timesteps = cfg.MODEL.TIME_STEPS
        self.img_size = cfg.MODEL.IMAGE_SIZE
        self.img_channels = cfg.MODEL.CHANNELS

        # define beta schedule
        self.betas = linear_beta_schedule(timesteps=cfg.MODEL.TIME_STEPS)

        # define alphas
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

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

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    @torch.no_grad()
    def p_sample(self, x, t, t_index):
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * self.denoise_model(x, t) / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
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

        for i in tqdm(
            reversed(range(0, self.timesteps)),
            desc="sampling loop time step",
            total=self.timesteps,
        ):
            img = self.p_sample(
                img, torch.full((b,), i, device=device, dtype=torch.long), i
            )
            imgs.append(img)
        return imgs  # t , batchsize, chennels, h, w

    @torch.no_grad()
    def sample(self, image_size, batch_size=16, channels=3):
        return self.p_sample_loop(shape=(batch_size, channels, image_size, image_size))
