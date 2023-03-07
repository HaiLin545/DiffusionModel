import torch
from torch.optim import Adam
from torch.optim import lr_scheduler


def make_optimizer(cfg, model):
    return Adam(model.parameters(), lr=cfg.TRAIN.LR)


def make_scheduler(cfg, optimizer):
    if cfg.TRAIN.SCHEDULER == "StepLR":
        return lr_scheduler.StepLR(
            optimizer, step_size=cfg.TRAIN.LR_DECAY_STEP, gamma=cfg.TRAIN.LR_DECAY
        )
    else:
        return None


def train(
    cfg,
    model,
    train_loader,
    optimizer,
    scheduler,
    epoch,
    writer,
):
    model.train()
    num_iters = len(train_loader)  # 390
    for iter, (x, _) in enumerate(train_loader):
        optimizer.zero_grad()

        x = x.to(cfg.DEVICE)
        t = torch.randint(
            0, cfg.DM.TIME_STEPS, (cfg.DATASET.BATCH_SIZE,), device=cfg.DEVICE
        ).long()

        loss = model(x, t)
        loss.backward()
        optimizer.step()
        scheduler.step()

        lr = scheduler.get_last_lr()[0]
        writer.add_scalar("lr", lr, epoch * num_iters + iter)

    return loss
