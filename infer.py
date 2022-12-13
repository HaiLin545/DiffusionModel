from libs.diffusion import DiffusionModel
from config.default import load_default_config, load_config
import os
import torch
from torchvision import transforms
from pathlib import Path
from torchvision.utils import save_image
import matplotlib.pyplot as plt

cfg = load_default_config()
image_size = cfg['image_size']
channels = cfg['channels']
batch_size = cfg['batch_size']
device = "cuda" if torch.cuda.is_available() else "cpu"

model = DiffusionModel(cfg)
model.to(device)

ckpt_folder = Path(cfg['ckpt_folder'])
ckpt = torch.load(os.path.join(ckpt_folder, 'model_final.pth'))
model.load_state_dict(ckpt)
model.eval()

samples = model.sample(image_size=image_size, batch_size=64, channels=channels)

step = cfg['timesteps'] // 10
all_images = torch.cat(samples[::step], dim=1).reshape(-1, 1, image_size, image_size)
all_images = (all_images + 1) * 0.5
result_folder = Path('./results')
save_image(all_images, str(result_folder / f'infer_result_{cfg["timesteps"]}.png'), nrow=10)