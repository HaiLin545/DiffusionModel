import os
import torch
import matplotlib.pyplot as plt
import argparse
from libs.diffusion import DiffusionModel
from config import cfg, update_config
from torchvision import transforms
from pathlib import Path
from torchvision.utils import save_image


def parse_args():
    parser = argparse.ArgumentParser(description="DDPM")
    parser.add_argument(
        "--cfg",
        type=str,
        help="experiment config file",
        default="./config/fashion_mnist.yaml",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    update_config(args.cfg)

    image_size = cfg.DATASET.IMAGE_SIZE
    channels = cfg.DATASET.CHANNELS
    device = cfg.DEVICE

    model = DiffusionModel(cfg)
    model.to(device)

    output_dir = Path(cfg.TRAIN.OUTPUT_DIR)
    output_folder = output_dir / cfg.TRAIN.OUTPUT_FOLDER
    ckpt_folder = output_folder / Path(cfg.TRAIN.CKPT_DIR)
    ckpt = torch.load(ckpt_folder / f"model_{cfg.TRAIN.EPOCH}_0.pth")

    model.load_state_dict(ckpt)
    model.eval()

    print("Sampling...")
    samples = model.sample(image_size=image_size, batch_size=64, channels=channels)

    step = cfg.DM.TIME_STEPS // 10
    all_images = torch.cat(samples[::step], dim=1).reshape(
        -1, 1, image_size, image_size
    )

    all_images = (all_images + 1) * 0.5
    result_folder = output_folder / "result"
    save_image(all_images, str(result_folder / f"infer_result_new.png"), nrow=10)
    print("Done!")


if __name__ == "__main__":
    main()
