"""Shared-MLP fitting of noisy four-analyzer observations in float32."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_case(name, seed, renderer_path):
    x = np.linspace(-1, 1, 32, dtype=np.float32)
    y = np.linspace(-1, 1, 32, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    if name == "renderer":
        data = np.load(renderer_path)
        stokes = data["stokes"][:, :, :3, 0].astype(np.float32)
        target = stokes.reshape(-1, 3)
    elif name == "high_dolp":
        I = 1 + 0.2 * np.cos(np.pi * xx) * np.cos(np.pi * yy)
        d = 0.90 + 0.09 * (xx + 1) / 2
        theta = np.pi * (0.35 * xx + 0.2 * yy)
        target = np.stack([I, I * d * np.cos(2 * theta), I * d * np.sin(2 * theta)], axis=-1).reshape(-1, 3).astype(np.float32)
    else:
        raise ValueError("unknown case")
    A = 0.5 * np.array([[1, 1, 0], [1, 0, 1], [1, -1, 0], [1, 0, -1]], dtype=np.float32)
    gen = torch.Generator().manual_seed(seed + 1000)
    y = target @ A.T + 0.01 * torch.randn(target.shape[0], 4, generator=gen).numpy().astype(np.float32)
    return target, y, A


class FoldMLP(nn.Module):
    def __init__(self, initialization="standard"):
        super().__init__()
        if initialization not in ("standard", "broad"):
            raise ValueError("invalid initialization")
        self.fc1 = nn.Linear(4, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 3)
        with torch.no_grad():
            self.fc3.bias[0] = float(np.log(np.expm1(1)))
            self.fc3.bias[1:] = 0.0
            if initialization == "standard":
                self.fc3.weight[1:] *= 0.2
            elif initialization == "broad":
                self.fc3.weight[1:] *= 3.0

    def forward(self, x):
        h = F.silu(self.fc1(x))
        h = F.silu(self.fc2(h))
        raw = self.fc3(h)
        I = F.softplus(raw[:, 0]) + 1e-6
        w = raw[:, 1:]
        s = torch.cat([I[:, None], I[:, None] * 2 * w / (1 + (w * w).sum(-1, keepdim=True))], dim=1)
        return s, I, w


def train_case(case_name, seed, initialization, renderer_path, steps=1500, learning_rate=.003):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    target, y_np, A = make_case(case_name, seed, renderer_path)
    y_t = torch.from_numpy(y_np)
    A_t = torch.from_numpy(A)
    model = FoldMLP(initialization=initialization)
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_curve = []
    checkpoints = {}
    for step in range(steps + 1):
        if step % 25 == 0:
            s, I, w = model(y_t)
            with torch.no_grad():
                loss = (0.5 * ((s @ A_t.T - y_t) ** 2).sum(-1)).mean()
                loss_curve.append(loss.item())
        if step in [0, 100, 500, 1000, 1500] and step <= steps:
            s, I, w = model(y_t)
            loss = (0.5 * ((s @ A_t.T - y_t) ** 2).sum(-1)).mean()
            grads = torch.autograd.grad(loss, model.parameters(), create_graph=False)
            grad_norm = float(torch.sqrt(sum(g.pow(2).sum() for g in grads if g is not None)).detach())
            with torch.no_grad():
                r = torch.norm(w, dim=1)
                dolp = 2 * r / (1 + r * r)
                q25, q50, q75 = [float(v) for v in torch.quantile(r, torch.tensor([.25, .5, .75]))]
                min_dolp = float(dolp.min())
                max_dolp = float(dolp.max())
                diff = s - torch.from_numpy(target).to(s.device)
                weighted_sq = (diff ** 2 * torch.tensor([1.0, 0.5, 0.5], dtype=s.dtype, device=s.device)).sum(dim=1)
                target_I = torch.from_numpy(target[:, 0]).to(s.device)
                clean_w_err = float((weighted_sq / (target_I ** 2 + 1e-12)).mean())
            assert bool(torch.all(torch.isfinite(loss))), "non-finite loss"
            for p in model.parameters():
                assert bool(torch.all(torch.isfinite(p))), "non-finite parameter"
            slack = I - torch.norm(s[:, 1:], dim=1)
            min_slack = float(slack.min().detach())
            assert min_slack >= -2e-6, f"cone violation: {min_slack}"
            checkpoints[step] = {
                "loss": loss.item(),
                "grad_norm": grad_norm,
                "radius_q25": q25,
                "radius_q50": q50,
                "radius_q75": q75,
                "min_dolp": min_dolp,
                "max_dolp": max_dolp,
                "clean_w_err": clean_w_err,
                "s_copy": s.detach().cpu().numpy(),
                "I_copy": I.detach().cpu().numpy(),
                "w_copy": w.detach().cpu().numpy(),
            }
        if step < steps:
            opt.zero_grad()
            s, I, w = model(y_t)
            loss = (0.5 * ((s @ A_t.T - y_t) ** 2).sum(-1)).mean()
            loss.backward()
            opt.step()
    with torch.no_grad():
        s, I, w = model(y_t)
        final_loss = float((0.5 * ((s @ A_t.T - y_t) ** 2).sum(-1)).mean())
        r = torch.norm(w, dim=1)
        dolp = 2 * r / (1 + r * r)
        assert bool(torch.all(torch.isfinite(s))), "non-finite s"
        assert bool(torch.all(torch.isfinite(I))), "non-finite I"
        assert bool(torch.all(torch.isfinite(w))), "non-finite w"
        assert float((I - 1e-6).min()) >= -2e-6, "intensity violation"
        assert float(dolp.min()) >= -2e-6 and float(dolp.max()) <= 1 + 2e-6, "dolp violation"
    state_dict = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    return {
        "target": target,
        "y": y_np,
        "A": A,
        "s": s.detach().cpu().numpy(),
        "I": I.detach().cpu().numpy(),
        "w": w.detach().cpu().numpy(),
        "loss_curve": loss_curve,
        "checkpoints": checkpoints,
        "state_dict": state_dict,
    }
