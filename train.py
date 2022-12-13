import os
import torch
import torchvision
from pathlib import Path
from torch.optim import Adam
from libs.unet import Unet
from libs.diffusion import *
from torchvision.utils import save_image
from torchvision import transforms
from torch.utils.data import DataLoader
from config.default import load_default_config, printcfg

cfg = load_default_config()
result_folder = Path(cfg['result_folder'])
result_folder.mkdir(exist_ok=True)
ckpt_folder = Path(cfg['ckpt_folder'])
ckpt_folder.mkdir(exist_ok=True)
save_and_sample_every = cfg['save_every']
transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Lambda(lambda t: (t * 2) - 1)])

# load dataset
dataset = torchvision.datasets.FashionMNIST(root='./dataset/', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True, drop_last=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = DiffusionModel(cfg)
model.to(device)

optimizer = Adam(model.parameters(), lr=cfg['lr'])
epochs = cfg['epoch']

print('Start training..')
if cfg['printcfg']:
    printcfg(cfg)

for epoch in range(epochs):
    for step, (x, _) in enumerate(dataloader):

        optimizer.zero_grad()
        x = x.to(device)
        t = torch.randint(0, cfg['timesteps'], (cfg['batch_size'], ), device=device).long()
        loss = model(x, t)

        if step % 100 == 0:
            print(f"Epoch: {epoch}, Iter: {step}, Loss: {loss.item()}")

        loss.backward()
        optimizer.step()

torch.save(model.state_dict(), os.path.join(ckpt_folder, f'model_final.pth'))
print("Done!")