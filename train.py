import os
import torch
import argparse
import torchvision
from pathlib import Path
from torch.optim import Adam
from libs.diffusion import *
from torchvision import transforms
from torch.utils.data import DataLoader
from config import cfg, print_cfg, update_config


def parse_args():
    parser = argparse.ArgumentParser(description="Train DDPM")
    parser.add_argument(
        "--cfg",
        type=str,
        help="experiment config file",
        default="./config/fashion_mnist.yaml",
    )
    args = parser.parse_args()
    return args


def train(
    cfg,
    model,
    train_loader,
    optimizer,
    device,
):
    model.train()
    for iter, (x, _) in enumerate(train_loader):
        optimizer.zero_grad()
        x = x.to(device)
        t = torch.randint(
            0, cfg.DM.TIME_STEPS, (cfg.DATASET.BATCH_SIZE,), device=device
        ).long()
        loss = model(x, t)
        loss.backward()
        optimizer.step()

    return loss


def main():
    args = parse_args()
    update_config(args.cfg)

    output_dir = Path(cfg.TRAIN.OUTPUT_DIR)
    output_dir.mkdir()
    output_folder = output_dir / cfg.TRAIN.OUTPUT_FOLDER
    output_folder.mkdir()
    ckpt_folder = output_folder / Path(cfg.TRAIN.CKPT_DIR)
    ckpt_folder.mkdir()

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1),
        ]
    )

    # load dataset
    dataset = torchvision.datasets.FashionMNIST(
        root="./dataset/", train=True, download=True, transform=transform
    )
    dataloader = DataLoader(
        dataset, batch_size=cfg.DATASET.BATCH_SIZE, shuffle=True, drop_last=True
    )

    device = cfg.DEVICE

    model = DiffusionModel(cfg)
    model.to(device)

    optimizer = Adam(model.parameters(), lr=cfg.TRAIN.LR)
    epochs = cfg.TRAIN.EPOCH

    if cfg.TRAIN.PRINT_CFG:
        print_cfg(cfg)

    print("Start training..")
    for epoch in range(1, epochs + 1):
        print(f"Trainning epoch {epoch}")

        loss = train(cfg, model, dataloader, optimizer, device)

        if epoch % cfg.TRAIN.SAVE_EVERY_EPOCH == 0 or epoch == epochs:
            print(f"Epoch: {epoch}, Loss: {loss.item()}")
            torch.save(
                model.state_dict(), os.path.join(ckpt_folder, f"model_{epoch}.pth")
            )

    print("Done!")


if __name__ == "__main__":
    main()
