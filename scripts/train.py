"""Train the spider with PPO (rsl-rl).

    python scripts/train.py                      # 4096 envs on GPU
    python scripts/train.py -b 1024              # smaller GPU
    python scripts/train.py --resume 1000        # continue from logs/<exp>/model_1000.pt
    tensorboard --logdir logs
"""

import argparse
import os
import pickle
import shutil
import sys
from importlib import metadata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if int(metadata.version("rsl-rl-lib").split(".")[0]) < 5:
    raise ImportError("Please install 'rsl-rl-lib>=5.0.0'.")
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from spider_stairs.config import get_cfgs, get_train_cfg
from spider_stairs.env import SpiderStairsEnv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-e", "--exp-name", default="spider-stairs")
    p.add_argument("-b", "--num-envs", type=int, default=4096)
    p.add_argument("--max-iterations", type=int, default=3000)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--cpu", action="store_true", help="run on CPU (debugging only; far too slow to train)")
    p.add_argument("--resume", type=int, default=None, help="checkpoint iteration to resume from")
    p.add_argument("--small-terrain", action="store_true", help="4x2 tile grid for quick tests")
    args = p.parse_args()

    log_dir = f"logs/{args.exp_name}"
    if args.resume is not None:
        with open(f"{log_dir}/cfgs.pkl", "rb") as f:
            cfgs, train_cfg = pickle.load(f)
    else:
        cfgs, train_cfg = get_cfgs(), get_train_cfg(args.exp_name)
        if args.small_terrain:
            cfgs["env"]["terrain"].update(n_levels=4, col_types=("ascend", "descend"))
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        os.makedirs(log_dir, exist_ok=True)
        with open(f"{log_dir}/cfgs.pkl", "wb") as f:
            pickle.dump([cfgs, train_cfg], f)

    gs.init(
        backend=gs.cpu if args.cpu else gs.gpu,
        precision="32",
        logging_level="warning",
        seed=args.seed,
        performance_mode=not args.cpu,
    )
    env = SpiderStairsEnv(num_envs=args.num_envs, cfgs=cfgs)
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    if args.resume is not None:
        runner.load(os.path.join(log_dir, f"model_{args.resume}.pt"))
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=args.resume is None)


if __name__ == "__main__":
    main()
