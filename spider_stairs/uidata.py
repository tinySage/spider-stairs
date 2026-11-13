"""Data for the dashboard: training curves from TensorBoard logs and recorded rollouts."""

from __future__ import annotations

import glob
import json
import os
import pickle
import re
import time
from dataclasses import asdict

from .robot import DEFAULT_POSE, SpiderSpec, leg_names
from .terrain import TerrainCfg


def read_metrics(log_dir: str, max_points: int = 400) -> dict:
    """All scalar curves of a run, merged across resumes and thinned to max_points."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    merged: dict[str, dict[int, float]] = {}
    for path in sorted(glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))):
        acc = EventAccumulator(path, size_guidance={"scalars": 0})
        acc.Reload()
        for tag in acc.Tags()["scalars"]:
            if tag.endswith("/time"):
                continue
            series = merged.setdefault(tag, {})
            for ev in acc.Scalars(tag):
                series[ev.step] = ev.value
    if not merged:
        return {"iters": [], "series": {}}
    iters = sorted({s for series in merged.values() for s in series})
    stride = max(1, len(iters) // max_points)
    keep = iters[::stride]
    if keep[-1] != iters[-1]:
        keep.append(iters[-1])
    out = {}
    for tag, series in merged.items():
        name = tag.split("/", 1)[1] if tag.startswith(("Episode/", "Train/")) else tag
        out[name] = [None if i not in series else round(series[i], 5) for i in keep]
    return {"iters": keep, "series": out}


def meta(cfgs: dict, exp_name: str) -> dict:
    tc = TerrainCfg(**cfgs["env"]["terrain"])
    return {
        "exp": exp_name,
        "generated": time.strftime("%Y-%m-%d %H:%M"),
        "dt": cfgs["env"]["dt"],
        "robot": {**asdict(SpiderSpec()), "legs": leg_names(), "default_pose": DEFAULT_POSE},
        "terrain": {
            "tile_size": tc.tile_size,
            "platform_half": tc.platform_half,
            "step_depth": tc.step_depth,
            "n_steps": tc.n_steps,
            "n_levels": tc.n_levels,
            # snapped to the height-field resolution, exactly as terrain.py builds them
            "step_heights": [
                round(round(tc.step_height(lv) / tc.vertical_scale) * tc.vertical_scale, 4) for lv in range(tc.n_levels)
            ],
        },
        "reward_scales": cfgs["reward"]["reward_scales"],
        "episode_length_s": cfgs["env"]["episode_length_s"],
    }


def latest_ckpt(log_dir: str) -> int:
    its = [int(re.search(r"model_(\d+)\.pt", f).group(1)) for f in glob.glob(f"{log_dir}/model_*.pt")]
    if not its:
        raise FileNotFoundError(f"no checkpoints in {log_dir}")
    return max(its)


def record_rollouts(log_dir: str, ckpt: int | None = None, gpu: bool = False, every: int = 2) -> dict:
    """Run the policy once on every (level, tile type) and store the motion for the 3D replay."""
    import torch
    from rsl_rl.runners import OnPolicyRunner

    import genesis as gs

    from .env import SpiderStairsEnv

    gs.init(backend=gs.gpu if gpu else gs.cpu, logging_level="warning", seed=7)
    with open(f"{log_dir}/cfgs.pkl", "rb") as f:
        cfgs, train_cfg = pickle.load(f)
    kinds = ("ascend", "descend")
    env_cfg = cfgs["env"]
    env_cfg["terrain"]["col_types"] = kinds
    env_cfg.update(curriculum=False, eval_level=None, max_push_vel_xy=0.0, friction_range=[1.0, 1.0])
    cfgs["obs"]["add_noise"] = False
    cfgs["reward"]["reward_scales"] = {}
    n_levels = env_cfg["terrain"]["n_levels"]
    n = n_levels * len(kinds)
    env = SpiderStairsEnv(num_envs=n, cfgs=cfgs)

    ckpt = latest_ckpt(log_dir) if ckpt is None else ckpt
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.load(os.path.join(log_dir, f"model_{ckpt}.pt"))
    policy = runner.get_inference_policy(device=gs.device)

    def r(t, nd=3):
        return [round(v, nd) for v in t.tolist()]

    rolls = [
        {
            "level": int(env.terrain_level[i]),
            "tile": kinds[int(env.terrain_col[i])],
            "heading": round(float(env.heading_target[i]), 3),
            "vx": round(float(env.commands[i, 0]), 3),
            "outcome": "timeout",
            "pos": [], "quat": [], "q": [], "contact": [],
        }
        for i in range(n)
    ]  # fmt: skip
    live = [True] * n
    obs = env.reset()
    for i in range(n):  # reset() redrew headings and speeds
        rolls[i]["heading"] = round(float(env.heading_target[i]), 3)
        rolls[i]["vx"] = round(float(env.commands[i, 0]), 3)
    origins = env.env_origins.clone()
    with torch.no_grad():
        for step in range(env.max_episode_length + 1):
            if step % every == 0:
                rel = env.base_pos.clone()
                rel[:, :2] -= origins[:, :2]
                bits = (env.feet_contact.long() * (2 ** torch.arange(env.num_feet, device=gs.device))).sum(1)
                for i in range(n):
                    if live[i]:
                        rolls[i]["pos"].append(r(rel[i]))
                        rolls[i]["quat"].append(r(env.base_quat[i], 4))
                        rolls[i]["q"].append(r(env.dof_pos[i]))
                        rolls[i]["contact"].append(int(bits[i]))
            obs, _, dones, _ = env.step(policy(obs))
            for i in range(n):
                if live[i] and bool(dones[i]):
                    live[i] = False
                    fell, ok = bool(env.terminated[i]), bool(env.success_mask[i])
                    rolls[i]["outcome"] = "fell" if fell else ("climbed" if ok else "timeout")
            if not any(live):
                break
    return {"ckpt": ckpt, "frame_dt": env.dt * every, "rollouts": rolls}


def assemble(log_dir: str, exp_name: str) -> dict:
    """Everything the dashboard needs, read fresh from disk."""
    from .config import get_cfgs

    cfg_path = os.path.join(log_dir, "cfgs.pkl")
    if os.path.exists(cfg_path):
        with open(cfg_path, "rb") as f:
            cfgs, _ = pickle.load(f)
    else:
        cfgs = get_cfgs()
    data = {"meta": meta(cfgs, exp_name), "metrics": read_metrics(log_dir), "replay": None}
    roll_path = os.path.join(log_dir, "rollouts.json")
    if os.path.exists(roll_path):
        with open(roll_path) as f:
            data["replay"] = json.load(f)
    return data
