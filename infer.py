from libs.diffusion import DiffusionModel
from config.default import config, update_config
import os
import torch
from torchvision import transforms
from pathlib import Path
from torchvision.utils import save_image
import matplotlib.pyplot as plt


def main():
    cfg = config
    image_size = cfg.MODEL.IMAGE_SIZE
    channels = cfg.MODEL.CHANNELS
    batch_size = cfg.DATASET.BATCH_SIZE
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = DiffusionModel(cfg)
    model.to(device)

    ckpt_folder = Path(cfg.TRAIN.CKPT_DIR)
    ckpt = torch.load(os.path.join(ckpt_folder, f"model_{cfg.TRAIN.EPOCH + 1}.pth"))
    model.load_state_dict(ckpt)
    model.eval()

    samples = model.sample(image_size=image_size, batch_size=64, channels=channels)

    step = cfg["timesteps"] // 10
    all_images = torch.cat(samples[::step], dim=1).reshape(
        -1, 1, image_size, image_size
    )
    all_images = (all_images + 1) * 0.5
    result_folder = Path("./results")
    save_image(
        all_images, str(result_folder / f'infer_result_{cfg["timesteps"]}.png'), nrow=10
    )


if __name__ == "__main__":
    main()
