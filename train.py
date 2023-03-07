import os
import time
import torch
import argparse
import shutil
from pathlib import Path
from libs.diffusion.diffusion import *
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from config import cfg, print_cfg, update_config
from libs.util.logger import time_cost, Logger
from libs.util.tools import fix_random_seed
from libs.dataset import make_dataset
from libs.core.func import make_optimizer, make_scheduler, train


def parse_args():
    parser = argparse.ArgumentParser(description="Train DDPM")
    parser.add_argument(
        "--cfg",
        type=str,
        help="experiment config file",
        default="./config/fashion_mnist.yaml",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="trainning output dir",
    )
    parser.add_argument(
        "--debug",
        help="is debug mode",
        action="store_true",
    )
    args = parser.parse_args()
    return args


def preprocess_args(args):
    update_config(args.cfg)
    if args.output:
        cfg.OUTPUT_FOLDER = args.output
    if args.debug:
        cfg.DEBUG = True
    if cfg.DEBUG:
        cfg.OUTPUT_FOLDER = cfg.OUTPUT_FOLDER + "_debug"


def main():
    args = parse_args()
    preprocess_args(args)
    begin_time = time.time()

    exist_ok = cfg.DEBUG or cfg.TRAIN.RESUME
    output_dir = Path(cfg.OUTPUT_DIR)
    output_dir.mkdir(exist_ok=exist_ok)
    output_folder = output_dir / cfg.OUTPUT_FOLDER
    output_folder.mkdir(exist_ok=exist_ok)
    ckpt_folder = output_folder / Path(cfg.CKPT_DIR)
    ckpt_folder.mkdir(exist_ok=exist_ok)

    logger = Logger(output_folder / cfg.LOG_FILE)

    if cfg.PRINT_CFG:
        print_cfg(cfg)
        shutil.copyfile(args.cfg, output_folder / "config.yaml")

    if cfg.BACKUP_CODE:

        def ignore(_a, _b):
            return [
                "config",
                "data",
                "__pycache__",
                "output",
                ".gitignore",
                "README.md",
            ]

        logger("Backup code...")
        if (output_folder / "code").exists():
            shutil.rmtree(output_folder / "code")
        shutil.copytree("./", output_folder / "code", ignore=ignore)

    logger("Dataset loading...")
    fix_random_seed(cfg)
    start_time = time.time()
    dataset = make_dataset(cfg)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.DATASET.BATCH_SIZE,
        num_workers=cfg.DATASET.NUM_WORKER,
        shuffle=True,
        drop_last=True,
    )
    logger(f"Dataset loaded. Time cost = {time_cost(start_time)}")
    start_time = time.time()
    logger("Model initializing...")
    model = DiffusionModel(cfg)
    model.to(cfg.DEVICE)
    logger(f"Model initialized. Time cost = {time_cost(start_time)}")

    optimizer = make_optimizer(cfg, model)
    scheduler = make_scheduler(cfg, optimizer)

    start_epoch = 1
    epochs = cfg.TRAIN.EPOCH

    if cfg.TRAIN.RESUME:
        if Path(cfg.TRAIN.RESUME).is_file():
            checkpoint = torch.load(
                cfg.TRAIN.RESUME,
                map_location=lambda storage, loc: storage.cuda(cfg.DEVICE),
            )
            start_epoch = checkpoint["epoch"] + 1
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            logger(
                f"loaded checkpoint '{cfg.TRAIN.RESUME}' (epoch {checkpoint['epoch']})"
            )
            del checkpoint
        else:
            raise ValueError(f"Resume checkpoint is not found: {cfg.TRAIN.RESUME}")

    writer = SummaryWriter(output_folder / "log")
    logger("Start training...")
    start_train_time = time.time()
    best_model = None
    min_loss = float("Inf")

    for epoch in range(start_epoch, epochs + 1):
        logger(f"Trainning epoch {epoch}...")
        start_time = time.time()
        loss = train(cfg, model, dataloader, optimizer, scheduler, epoch, writer)

        if loss.item() < min_loss:
            min_loss = loss.item()
            best_model = model

        writer.add_scalar("Train Loss", loss.item(), epoch)
        logger(f"Epoch: {epoch}, Loss: {loss.item():.5f}")

        if epoch % cfg.TRAIN.SAVE_EVERY_EPOCH == 0 or epoch == epochs:
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                ckpt_folder / f"model_{epoch}.pth",
            )

        logger(f"Epoch {epoch} end. Time cost = {time_cost(start_time)}")

    torch.save(best_model.state_dict(), ckpt_folder / f"model_best.pth")
    logger(f"Train done! Time cost = {time_cost(start_train_time)}")
    logger(f"All done! Total running time = {time_cost(begin_time)}")


if __name__ == "__main__":
    main()
