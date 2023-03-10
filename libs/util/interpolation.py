import torch


def slerp(x0, x1, alpha):
    theta = torch.acos(torch.sum(x1 * x0) / torch.norm(x1) / torch.norm(x0))
    return (
        torch.sin((1 - alpha) * theta) / torch.sin(theta) * x1
        + torch.sin(alpha * theta) / torch.sin(theta) * x0
    )


def spherical_linear_interpolation(shape, device):
    # shape: bs, c, h, w
    x0 = torch.randn(shape[1:], device=device)
    x1 = torch.randn(shape[1:], device=device)
    alphas = torch.arange(0.0, 0.1 * (shape[0]), 0.1).to(device)
    out = []
    for alpha in alphas:
        out.append(slerp(x0, x1, alpha))
    return torch.cat(out, dim=0).reshape(shape)
