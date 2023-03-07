import torchvision
from torchvision import transforms


def make_dataset(cfg):
    if cfg.DATASET.NAME == "FashionMNIST":
        return torchvision.datasets.FashionMNIST(
            root=cfg.DATASET.ROOT,
            train=True,
            download=False,
            transform=transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda t: (t * 2) - 1),
                ]
            ),
        )

    elif cfg.DATASET.NAME == "cifar10":
        return torchvision.datasets.CIFAR10(
            root=cfg.DATASET.ROOT,
            train=True,
            download=False,
            transform=transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda t: (t * 2) - 1),
                ]
            ),
        )

    else:
        raise ValueError(f"'{cfg.DATASET.NAME}' is an invalid dataset")
