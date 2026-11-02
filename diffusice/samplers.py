import torch

from tqdm import tqdm

from abc import ABC, abstractmethod

class BaseSampler(ABC):
    @abstractmethod
    def setup_schedule(self, N, sigma_min, sigma_max, rho, device):
        ...

    @abstractmethod
    @torch.no_grad()
    def step(self, z, denoiser_fn, time_steps, gamma, N, step_callback = None):
        ...


class HeunStochasticSampler(BaseSampler):
    def __init__(self, rho = 7, sigma_min = 0.002, sigma_max = 80, S_churn = 40, S_tmin = 0.05, S_tmax = 50, S_noise = 1.003):
        """
        Set according to Karras Table 1 and 5
        """
        self.rho = rho
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.S_churn = S_churn
        self.S_tmin = S_tmin
        self.S_tmax = S_tmax
        self.S_noise = S_noise

    def setup_schedule(self, N, device):
        idx = torch.arange(N, device = device)
        sigma_i = (self.sigma_max ** (1. / self.rho) + (idx / (N - 1)) * (self.sigma_min ** (1. / self.rho) - self.sigma_max ** (1. / self.rho))) ** (self.rho)
        time_steps = torch.concat([
            sigma_i,
            torch.zeros(1, device = device)
        ])

        gamma = torch.zeros_like(time_steps)
        noise_idxs = torch.where((time_steps > self.S_tmin) & (time_steps < self.S_tmax))
        gamma[noise_idxs] = torch.min(torch.tensor((self.S_churn / N, torch.sqrt(torch.tensor(2.)) - 1)))
        return time_steps, gamma
    
    def step(self, z, denoiser_fn, time_steps, gamma, N, step_callback = None):
        b, c, h, w = z.size()
        z_step = z
        for i in tqdm(range(N), leave = False):
            eps_i = torch.randn_like(z) * self.S_noise
            t_hat_i = time_steps[i] * (1 + gamma[i])
            z_hat_i = z + torch.sqrt((t_hat_i ** 2) - (time_steps[i] ** 2)) * eps_i

            D_out = denoiser_fn(z_hat_i, t_hat_i)

            # Dynamic thresholding (Saharia 2022)
            # s = torch.quantile(D_out.reshape(b, c, -1), q = torch.tensor([0.005, 0.995], device = z.device), dim = -1)
            # D_out = torch.clamp(D_out, min = s[0].unsqueeze(-1).unsqueeze(-1), max = s[1].unsqueeze(-1).unsqueeze(-1))

            scale_factor = 1. / t_hat_i
            dxdt = scale_factor * (z_hat_i - D_out)

            # Take euler step from t_(i) -> t_(i+1)
            z_step = z_hat_i + (time_steps[i+1] - t_hat_i) * dxdt

            z_corrected = None

            if time_steps[i+1] != 0:
                D_out_step = denoiser_fn(z_step, time_steps[i+1])
    
                # Dynamic thresholding (Saharia 2022)
                # s_step = torch.quantile(D_out_step.reshape(b, c, -1), q = torch.tensor([0.005, 0.995], device = z.device), dim = -1)
                # D_out_step = torch.clamp(D_out_step, min = s_step[0].unsqueeze(-1).unsqueeze(-1), max = s_step[1].unsqueeze(-1).unsqueeze(-1))

                scale_factor_step = 1. / time_steps[i+1]

                dxdt_step = scale_factor_step * (z_step - D_out_step)

                z_corrected = z_hat_i + (time_steps[i+1] - t_hat_i) * (0.5*dxdt + 0.5*dxdt_step)
                z = z_corrected

            if step_callback is not None:
                step_callback(i, z_step, z_corrected)

        return z_step