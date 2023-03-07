import os
import torch
import matplotlib.pyplot as plt
import argparse
from libs.diffusion.diffusion import DiffusionModel
from config import cfg, update_config
from torchvision import transforms
from pathlib import Path
from torchvision.utils import save_image
from libs.util.logger import Logger


def parse_args():
    parser = argparse.ArgumentParser(description="DDPM")
    parser.add_argument(
        "--cfg",
        type=str,
        help="experiment config file",
        default="./config/fashion_mnist.yaml",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        help="model checkpoint dir",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="output dir",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    update_config(args.cfg)

    if args.output:
        cfg.OUTPUT_FOLDER = args.output
    if cfg.DEBUG:
        cfg.OUTPUT_FOLDER = cfg.OUTPUT_FOLDER + "_debug"

    if args.ckpt:
        ckpt_dir = args.ckpt
        output_folder = Path(ckpt_dir).parent.parent
    else:
        output_folder = Path(cfg.OUTPUT_DIR) / cfg.OUTPUT_FOLDER
        ckpt_dir = output_folder / cfg.CKPT_FOLDER / f"model_{cfg.TRAIN.EPOCH}.pth"

    result_folder = output_folder / "result"
    result_folder.mkdir(exist_ok=True)

    ckpt = torch.load(ckpt_dir)
    if type(ckpt) == dict:
        ckpt = ckpt["state_dict"]

    image_size = cfg.DATASET.IMAGE_SIZE
    channels = cfg.DATASET.CHANNELS

    model = DiffusionModel(cfg)
    model.to(cfg.DEVICE)

    model.load_state_dict(ckpt)
    model.eval()

    logger = Logger()
    logger("Sampling...")
    samples = model.sample(image_size=image_size, batch_size=20, channels=channels)

    nrow = 20
    step = cfg.DM.TIME_STEPS // nrow
    all_images = torch.cat(samples[::step], dim=1).reshape(
        -1, channels, image_size, image_size
    )

    all_images = (all_images + 1) * 0.5

    save_image(
        all_images, str(result_folder / f"{os.path.basename(ckpt_dir)}.png"), nrow=nrow
    )
    logger("Done!")


if __name__ == "__main__":
    main()
