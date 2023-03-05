import os
import torch
import torchvision
from pathlib import Path
from torch.optim import Adam
from libs.diffusion import *
from torchvision import transforms
from torch.utils.data import DataLoader
from config.default import config, print_cfg


def main():
    cfg = config
    result_folder = Path(cfg.TRAIN.OUTPUT_DIR)
    result_folder.mkdir(exist_ok=True)
    ckpt_folder = Path(cfg.TRAIN.CKPT_DIR)
    ckpt_folder.mkdir(exist_ok=True)

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
        dataset, batch_size=cfg["batch_size"], shuffle=True, drop_last=True
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DiffusionModel(cfg)
    model.to(device)

    optimizer = Adam(model.parameters(), lr=cfg["lr"])
    epochs = cfg.TRAIN.EPOCH

    if cfg.TRAIN.PRINT_CFG:
        print_cfg(cfg)

    print("Start training..")
    for epoch in range(1, epochs + 1):
        print(f"Trainning epoch {epoch+1}")
        for iter, (x, _) in enumerate(dataloader):
            optimizer.zero_grad()
            x = x.to(device)
            t = torch.randint(
                0, cfg.MODEL.TIME_STEPS, (cfg.DATASET.BATCH_SIZE,), device=device
            ).long()
            loss = model(x, t)

            if iter % cfg.TRAIN.SAVE_EVERY_ITER == 0:
                print(f"Epoch: {epoch}, Iter: {iter}, Loss: {loss.item()}")
                torch.save(
                    model.state_dict(),
                    os.path.join(ckpt_folder, f"model_{epoch}_{iter}.pth"),
                )

            loss.backward()
            optimizer.step()

        if epoch % cfg.TRAIN.SAVE_EVERY_EPOCH == 0 or epoch == epochs - 1:
            print(f"Epoch: {epoch}, Loss: {loss.item()}")
            torch.save(
                model.state_dict(), os.path.join(ckpt_folder, f"model_{epoch}.pth")
            )

    print("Done!")


if __name__ == "__main__":
    main()
