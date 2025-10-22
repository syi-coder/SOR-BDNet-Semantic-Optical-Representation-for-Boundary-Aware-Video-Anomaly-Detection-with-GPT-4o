import torch
import torch.nn.functional as F

def nt_xent_loss(z_i, z_j, temperature=0.07, eps=1e-12, t_min=1e-3):
    n = z_i.size(0)
    t = max(float(temperature), t_min)
    z = torch.cat([z_i, z_j], dim=0)
    z = F.normalize(z, dim=1)
    sim = torch.mm(z, z.t()) / t
    pos = torch.cat([torch.diag(sim, n), torch.diag(sim, -n)], dim=0)
    mask = torch.ones((2*n, 2*n), device=z.device, dtype=torch.bool)
    mask.fill_diagonal_(False)
    sim_masked = sim.masked_fill(~mask, float("-inf"))
    log_denom = torch.logsumexp(sim_masked, dim=1)
    loss = -(pos - log_denom).mean()
    if torch.isnan(loss) or torch.isinf(loss):
        loss = torch.clamp(loss, min=-1e6, max=1e6) + eps
    return loss