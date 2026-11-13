"""Fast CPU check: build a tiny terrain, step random actions, sanity-check shapes and physics."""
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
import genesis as gs

from spider_stairs.config import get_cfgs
from spider_stairs.env import SpiderStairsEnv


def main():
    gs.init(backend=gs.cpu, logging_level="warning", seed=0)
    cfgs = get_cfgs()
    cfgs["env"]["terrain"].update(n_levels=2, col_types=("ascend", "descend"))
    cfgs["env"]["init_max_level"] = 1
    t0 = time.time()
    env = SpiderStairsEnv(num_envs=8, cfgs=cfgs)
    print(f"build: {time.time() - t0:.1f}s")
    obs = env.reset()
    assert obs["policy"].shape == (8, 158), obs["policy"].shape
    assert obs["privileged"].shape == (8, 13), obs["privileged"].shape

    # 1) standing still: the default pose must hold the body up without touching down
    for _ in range(100):
        obs, rew, done, extras = env.step(torch.zeros((8, 24), device=gs.device))
        assert not done.any(), "spider collapsed while standing"
    h = env.base_pos[:, 2] - env.env_origins[:, 2]
    print("standing height:", h.mean().item(), "feet in contact:", env.feet_contact.float().sum(1).mean().item())
    assert (h > 0.10).all() and (h < 0.20).all()
    assert env.feet_contact.all()

    # 2) random actions: nothing may go NaN, resets must work
    t0, n_done = time.time(), 0
    for _ in range(300):
        obs, rew, done, extras = env.step(torch.randn((8, 24), device=gs.device))
        n_done += int(done.sum())
        assert torch.isfinite(obs["policy"]).all() and torch.isfinite(rew).all()
    print(f"random: {300 * 8 / (time.time() - t0):.0f} env-steps/s, resets={n_done}, mean rew={rew.mean().item():.4f}")
    print({k: round(float(v), 4) for k, v in extras["episode"].items()})
    print("OK")


if __name__ == "__main__":
    main()
