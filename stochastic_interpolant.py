import torch
from tqdm import tqdm


class StochasticInterpolant(torch.nn.Module):
    def __init__(self, drift_net, denoiser_net):
        """
        SI: 
            Xt|X0,X1 = I(t, X0, X1) + gamma(t) * z
            I(t, X0, X1) = (1-t) * X0 + t * X1
            gamma(t) = sqrt(t * (1 - t))
        """
        super().__init__()
        self.drift_net = drift_net
        self.denoiser_net = denoiser_net

    @property
    def device(self):
        return next(self.denoiser_net.parameters()).device

    def gamma(self, t):
        if isinstance(t, float):
            t = torch.tensor(t, device=self.device)
        return torch.sqrt(t * (1. - t))
    
    def gamma_dot(self, t):
        if isinstance(t, float):
            t = torch.tensor(t, device=self.device)
        return 0.5 * (1. - 2. * t) / self.gamma(t)
    
    def interpolant(self, t, x0, x1):
        return (1 - t) * x0 + t * x1
    
    def I_dot(self, t, x0, x1):
        return x1 - x0
    
    def b(self, t, x0, x1, xt, z=None):
        """
            b(t, xt|x0, x1) = I_dot(t, x0, x1) + gamma_dot(t) * z, where xt = I(t, x0, x1) + gamma(t) * z
        =>  b(t, xt|x0, x1) = I_dot(t, x0, x1) + gamma_dot(t) / gamma(t) * (xt - I(t, x0, x1))
        """
        if z is None:
            return self.I_dot(t, x0, x1) + self.gamma_dot(t) / self.gamma(t) * (xt - self.interpolant(t, x0, x1))
        else:
            return self.I_dot(t, x0, x1) + self.gamma_dot(t) * z
    
    def sample(self, t, x0, x1, return_z=False):
        I = self.interpolant(t, x0, x1)
        z = torch.randn_like(I)
        if return_z:
            return I + self.gamma(t) * z, z
        else:
            return I + self.gamma(t) * z
        
    def compute_denoiser_loss(self, t, z, xt):
        z_hat = self.denoiser_net(t, xt)
        return torch.mean((z - z_hat) ** 2, dim=-1)
    
    def compute_drift_loss(self, t, x0, x1, xt, z=None):
        b = self.b(t, x0, x1, xt, z=z)
        b_hat = self.drift_net(t, xt)
        return torch.mean((b - b_hat) ** 2, dim=-1)
    
    def compute_loss(self, x0, x1):
        assert x0.shape == x1.shape
        t = torch.rand(x0.shape[0], 1, device=x0.device)
        t = t.clamp(min=1e-5, max=1-1e-5)
        xt, z = self.sample(t, x0, x1, return_z=True)
        denoiser_loss = self.compute_denoiser_loss(t, z, xt)
        drift_loss = self.compute_drift_loss(t, x0, x1, xt, z=z)
        return denoiser_loss.mean(), drift_loss.mean()

    def _calc_optimal_weights(self, t, xt, x0, x1, return_diff=False):
        """
        Calculate the optimal weights using for posterior sampling, 
        which could be used for estimating optimal drift and optimal denoiser/score
        """
        interpolants = self.interpolant(t, x0, x1)
        gamma_t = self.gamma(t).unsqueeze(-1)
        diff = xt.unsqueeze(1) - interpolants.unsqueeze(0)
        log_w = -0.5 / gamma_t ** 2 * torch.sum(diff ** 2, dim=-1, keepdim=True)
        if return_diff:
            return torch.softmax(log_w, dim=1), diff
        else:
            return torch.softmax(log_w, dim=1)

    def optimal_denoiser(self, t, xt, x0, x1):
        w, diff = self._calc_optimal_weights(t, xt, x0, x1, return_diff=True)
        cond_denoiser = diff / self.gamma(t).unsqueeze(-1)
        return (w * cond_denoiser).sum(dim=-2)

    def optimal_drift(self, t, xt, x0, x1):
        w, diff = self._calc_optimal_weights(t, xt, x0, x1, return_diff=True)
        cond_denoiser = diff / self.gamma(t).unsqueeze(-1)
        cond_velocity = self.I_dot(t, x0, x1).unsqueeze(0)
        cond_drift = cond_velocity + self.gamma_dot(t).unsqueeze(-1) * cond_denoiser
        return (w * cond_drift).sum(dim=-2)

    def optimal_drift_denoiser(self, t, xt, x0, x1):
        w, diff = self._calc_optimal_weights(t, xt, x0, x1, return_diff=True)
        cond_denoiser = diff / self.gamma(t).unsqueeze(-1)
        cond_velocity = self.I_dot(t, x0, x1).unsqueeze(0)
        cond_drift = cond_velocity + self.gamma_dot(t).unsqueeze(-1) * cond_denoiser
        return (w * cond_drift).sum(dim=-2), (w * cond_denoiser).sum(dim=-2)

    def denoiser_to_score_wrapper(self, denoiser_fn):
        def score_fn(t, xt, **kwargs):
            gamma_t_inv = 1. / self.gamma(t)
            return -gamma_t_inv * self.denoiser_net(t, xt)
        return score_fn

    def score_to_denoiser_wrapper(self, score_fn):
        def denoiser_fn(t, xt, **kwargs):
            gamma_t = self.gamma(t)
            return -gamma_t * score_fn(t, xt)
        return denoiser_fn

    def simulate(self, step_fn, x0, start=1e-3, end=1.-1e-3, n_steps=100, return_traj=True):
        ts = torch.linspace(start, end, n_steps).to(x0.device)
        samples = x0.clone().detach()
        traj = [samples.clone().detach().cpu()]
        progress_bar = tqdm(range(n_steps - 1), desc="Simulating")
        for i in progress_bar:
            t_cur = torch.ones((x0.shape[0], 1), device=x0.device) * ts[i]
            t_next = torch.ones((x0.shape[0], 1), device=x0.device) * ts[i + 1]
            samples = step_fn(t_cur, t_next, samples)
            if return_traj:
                traj.append(samples.clone().detach().cpu())
        if return_traj:
            return samples, torch.stack(traj, dim=0)
        else:
            return samples

    def forward_ode(self, drift_fn, x0, start=1e-3, end=1.-1e-3, n_steps=100, return_traj=True):
        def ode_step(t, s, x):
            return x + (s - t) * drift_fn(t, x)
        return self.simulate(ode_step, x0, start=start, end=end, n_steps=n_steps, return_traj=return_traj)

    def backward_ode(self, drift_fn, x1, start=1e-3, end=1.-1e-3, n_steps=100, return_traj=True):
        return self.forward_ode(drift_fn, x1, start=end, end=start, n_steps=n_steps, return_traj=return_traj)

    def forward_sde(self, drift_fn, score_fn, diffusion_fn, x0, start=1e-3, end=1.-1e-3, n_steps=100, return_traj=True):
        """
        Simulate the forward sde for SI:
            dXt = [drift_fn(t, Xt) + diffusion_fn(t) * score_fn(t, Xt)]dt + \sqrt{2*diffusion_fn(t)}dWt
        """
        def sde_step(t, s, x):
            dt = s - t
            diffusion = diffusion_fn(t)
            forward_drift = drift_fn(t, x) + (2. * end > start - 1.) * diffusion * score_fn(t, x)
            noise = torch.randn_like(x)
            return x + forward_drift * dt + torch.sqrt(2 * diffusion * dt.abs()) * noise
        return self.simulate(sde_step, x0, start=start, end=end, n_steps=n_steps, return_traj=return_traj)

    def backward_sde(self, drift_fn, score_fn, diffusion_fn, x1, start=1e-3, end=1.-1e-3, n_steps=100, return_traj=True):
        return self.forward_sde(drift_fn, score_fn, diffusion_fn, x1, start=end, end=start, n_steps=n_steps, return_traj=return_traj)