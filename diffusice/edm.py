import torch

def sample_sigma(P_mean, P_std, n, device):
    lnsigma = torch.normal(
        mean = torch.tensor([P_mean], device = device).repeat(n),
        std = torch.tensor([P_std], device = device)
    )
    return torch.exp(lnsigma), lnsigma

def precondition(sigma, lnsigma, sigma_data):
    c_skip = sigma_data**2 / (sigma**2 + sigma_data**2)
    c_out = (sigma * sigma_data) / torch.sqrt(sigma_data**2 + sigma**2)
    c_in = 1. / torch.sqrt(sigma**2 + sigma_data**2)

    c_noise = 0.25 * lnsigma

    return c_skip, c_out, c_in, c_noise

def loss_weight(sigma, sigma_data, n):
    return ((sigma**2 + sigma_data**2) / (sigma * sigma_data)**2).view(-1, *([1]*(n)))

def denoise(backbone, inputs, noisy_y, c_noise, c_skip, c_out, n):
    F_out = backbone(inputs, c_noise)
    return c_skip.view(-1, *([1]*(n))) * noisy_y + c_out.view(-1, *([1]*(n))) * F_out