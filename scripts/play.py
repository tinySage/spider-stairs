"""Roll out a trained policy in the viewer, or record it to a video.

    python scripts/play.py --level 5                       # viewer
    python scripts/play.py --level 7 --record climb.mp4    # headless video
    python scripts/play.py --random --record stand.mp4     # no checkpoint: zero actions (debug the scene)
"""

import argparse
import glob
import os
import pickle
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from spider_stairs.config import get_cfgs, get_train_cfg
from spider_stairs.env import SpiderStairsEnv


def latest_ckpt(log_dir):
    its = [int(re.search(r"model_(\d+)\.pt", f).group(1)) for f in glob.glob(f"{log_dir}/model_*.pt")]
    if not its:
        raise FileNotFoundError(f"no checkpoints in {log_dir}")
    return max(its)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-e", "--exp-name", default="spider-stairs")
    p.add_argument("--ckpt", type=int, default=None, help="iteration to load (default: latest)")
    p.add_argument("--level", type=int, default=None, help="terrain level 0..n-1 (default: hardest)")
    p.add_argument("--tile", choices=["ascend", "descend"], default="ascend")
    p.add_argument("--record", default=None, help="write an mp4 here instead of opening the viewer")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--random", action="store_true", help="skip the checkpoint and send zero actions")
    p.add_argument("--gpu", action="store_true")
    args = p.parse_args()

    gs.init(backend=gs.gpu if args.gpu else gs.cpu, logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    if args.random:
        cfgs, train_cfg = get_cfgs(), get_train_cfg(args.exp_name)
    else:
        with open(f"{log_dir}/cfgs.pkl", "rb") as f:
            cfgs, train_cfg = pickle.load(f)

    env_cfg = cfgs["env"]
    n_levels = env_cfg["terrain"]["n_levels"]
    level = n_levels - 1 if args.level is None else args.level
    # keep the trained terrain layout but put the single env on the requested tile type
    cols = list(env_cfg["terrain"]["col_types"])
    cols.insert(0, cols.pop(cols.index(args.tile)))
    env_cfg["terrain"]["col_types"] = tuple(cols)
    env_cfg.update(curriculum=False, eval_level=level, max_push_vel_xy=0.0, friction_range=[1.0, 1.0])
    cfgs["obs"]["add_noise"] = False
    cfgs["reward"]["reward_scales"] = {}

    env = SpiderStairsEnv(
        num_envs=1, cfgs=cfgs, show_viewer=args.record is None, camera={} if args.record else None
    )

    if args.random:
        policy = lambda obs: torch.zeros((1, env.num_actions), device=gs.device)  # noqa: E731
    else:
        ckpt = args.ckpt if args.ckpt is not None else latest_ckpt(log_dir)
        runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
        runner.load(os.path.join(log_dir, f"model_{ckpt}.pt"))
        policy = runner.get_inference_policy(device=gs.device)
        print(f"loaded model_{ckpt}.pt | level {level} ({100 * env.level_step_height[level].item():.1f} cm steps)")

    obs = env.reset()
    cam_target = env.base_pos[0].cpu().numpy().copy()
    if env.camera is not None:
        # Genesis >= 1.x: the file is named up front and a frame is encoded on every scene step
        env.camera.set_pose(pos=cam_target + (1.6, -1.6, 1.1), lookat=cam_target, up=(0.0, 0.0, 1.0))
        env.camera.start_recording(save_to_filename=args.record, fps=int(1 / env.dt))
    with torch.no_grad():
        for _ in range(args.steps):
            obs, _, _, _ = env.step(policy(obs))
            if env.camera is not None:
                cam_target += 0.1 * (env.base_pos[0].cpu().numpy() - cam_target)  # smooth follow
                env.camera.set_pose(pos=cam_target + (1.6, -1.6, 1.1), lookat=cam_target, up=(0.0, 0.0, 1.0))
    if env.camera is not None:
        env.camera.stop_recording()
        print(f"saved {args.record}")

if __name__ == "__main__":
    main()
