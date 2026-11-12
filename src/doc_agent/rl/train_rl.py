"""Stage 7 — RL/RLVR training loop"""
from __future__ import annotations

import numpy as np
import torch

from ..contracts import *  # noqa
from ..ingest.loader import data_paths
from ..logging_conf import get_logger
from .env import ToolSelectionEnv
from .policy import PolicyNet

log = get_logger(__name__)


def _rollout(env: ToolSelectionEnv, net: PolicyNet, episodes: int, gamma: float):
    S, A, LP, R = [], [], [], []
    for _ in range(episodes):
        s, _ = env.reset()
        ep_s, ep_a, ep_lp, ep_r, done = [], [], [], [], False
        while not done:
            with torch.no_grad():
                logits, _ = net(torch.from_numpy(s))
                dist = torch.distributions.Categorical(logits=logits)
                a = dist.sample()
            s2, r, done, _, _ = env.step(int(a))
            ep_s.append(s); ep_a.append(int(a)); ep_lp.append(float(dist.log_prob(a))); ep_r.append(r)
            s = s2
        g, rets = 0.0, []
        for r in reversed(ep_r):          # discounted return-to-go
            g = r + gamma * g
            rets.append(g)
        S += ep_s; A += ep_a; LP += ep_lp; R += rets[::-1]
    f = torch.tensor
    return (f(np.array(S)), f(A), f(LP, dtype=torch.float32), f(R, dtype=torch.float32))


def train(cfg: dict, iterations: int = 60, episodes: int = 64) -> float:
    """Train the tool-selection policy with PPO (clipped surrogate + value baseline) on the
    ToolSelectionEnv simulator; the reward there is verifiable (answer correctness / hallucination),
    i.e. RLVR-style. Saves models/policy.pt; returns mean episode reward of the last iteration."""
    torch.manual_seed(cfg.get("seed", 42))
    env = ToolSelectionEnv(cfg)
    env.reset(seed=cfg.get("seed", 42))
    net = PolicyNet()
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    mean_r = 0.0
    for it in range(iterations):
        S, A, LP, R = _rollout(env, net, episodes, gamma=0.99)
        mean_r = float(R.mean())
        for _ in range(4):                               # PPO epochs over the same batch
            logits, v = net(S)
            adv = (R - v.detach())
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            dist = torch.distributions.Categorical(logits=logits)
            ratio = torch.exp(dist.log_prob(A) - LP)
            pg = -torch.min(ratio * adv, torch.clamp(ratio, 0.8, 1.2) * adv).mean()
            loss = pg + 0.5 * (R - v).pow(2).mean() - 0.01 * dist.entropy().mean()
            opt.zero_grad(); loss.backward(); opt.step()
        if it % 10 == 0:
            log.info("ppo iter %d mean_return %.3f", it, mean_r)
    out = data_paths(cfg)["models_dir"]
    out.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), out / "policy.pt")
    return mean_r
