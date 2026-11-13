# spider-stairs

An 8-legged, 24-DOF spider that learns to climb stairs in [Genesis](https://github.com/Genesis-Embodied-AI/genesis-world), trained with PPO across thousands of parallel simulations.

![the dashboard](docs/dashboard.png)

## Layout

```
spider-stairs/
├── assets/spider.xml          generated robot model (MJCF)
├── spider_stairs/
│   ├── robot.py               procedural spider: 8 legs x (coxa yaw, femur pitch, tibia pitch)
│   ├── terrain.py             stair curriculum height field
│   ├── env.py                 parallel Genesis environment, observations, rewards, curriculum
│   ├── config.py              every tunable: PD gains, rewards, noise, PPO hyperparameters
│   └── uidata.py              reads training logs and records replays for the dashboard
├── scripts/
│   ├── train.py               PPO training (rsl-rl)
│   ├── play.py                viewer rollout or mp4 recording of a checkpoint
│   └── dashboard.py           serves the dashboard, records replays, builds a shareable file
├── ui/index.html              the dashboard: one file, no build step, no dependencies
└── tests/smoke_test.py        60-second CPU check that the sim and env are sane
```

## Design

**Robot.** Generated from code, so leg count, lengths and masses are parameters (`SpiderSpec`). 3.1 kg, body 15 cm off the ground, feet on a 55 x 59 cm footprint. The default stance holds the knees above the body, which is what lets a leg reach over a riser.

**Terrain.** A grid of 6 m tiles. Rows are difficulty levels with step heights from 2 cm to 12 cm (12 cm is 78% of the spider's standing height). Columns are tile types:

- `ascend`: a pit. The spider starts on the floor and climbs up six steps to the rim.
- `descend`: a pyramid. It starts on the summit and climbs down.

Steps are square rings, so every walking direction meets a staircase and diagonal headings cross the corners.

**Curriculum.** Each spider is told to walk in a random direction. Reach the far end of the staircase and it moves up a level; fail to get halfway and it moves down. Graduates of the hardest level are recycled to random levels. Nobody is asked to climb 12 cm steps until they have climbed 10 cm ones.

**Observations (158).** Body angular velocity, gravity direction, command, 24 joint positions, 24 joint velocities, previous action, and a 11 x 7 terrain height scan around the body. The critic additionally sees true body velocity, foot contacts, friction and step height (asymmetric actor-critic), which stabilises value learning without the policy depending on them.

**Actions (24).** Joint position targets around the default stance, tracked by PD control at 200 Hz; the policy runs at 50 Hz with one step of latency.

**Rewards.** Velocity and heading tracking drive the task. The spider walks at 0.25 to 0.6 m/s, so the tracking kernel is five times tighter than the usual quadruped setting, and a dense `forward_progress` term pays for every fraction of the commanded speed. With the stock settings, standing still earned over half the tracking reward and the policy learned to do exactly that. Light penalties on body bounce, tilt and height keep it stable without forbidding the pitching that climbing needs. `feet_air_time` rewards real steps over shuffling, `feet_slip` penalises a planted foot that slides (without it the policy learns to jitter its joints and ratchet along on friction, a gait that only exists while exploration noise is on), `feet_stumble` penalises legs hitting risers side-on, `collision` penalises knees and hips touching the stairs, and body contact ends the episode.

**Robustness.** Randomised friction, random pushes, sensor noise.

## Run it

Needs a GPU for training (NVIDIA with CUDA, or Apple Silicon). A free Colab T4 works.

```bash
pip install genesis-world "rsl-rl-lib>=5.0.0" tensorboard    # plus PyTorch for your platform

python tests/smoke_test.py            # sanity check, runs on CPU
python scripts/train.py               # 4096 envs, 3000 iterations
tensorboard --logdir logs             # watch Episode/terrain_level and Episode/success_rate

python scripts/play.py --level 7                     # watch the hardest stairs in the viewer
python scripts/play.py --level 7 --record climb.mp4  # or record headless
python scripts/play.py --tile descend --level 5
```

On a small GPU use `-b 1024` or `-b 2048`. `--resume N` continues from `model_N.pt`.

## Dashboard

```bash
python scripts/dashboard.py                    # http://localhost:8000, curves refresh every 15 s while training
python scripts/dashboard.py --record           # replay the latest checkpoint on every staircase, then serve
python scripts/dashboard.py --build run.html   # one self-contained file to share or put in a portfolio
```

- **Plate.** A 3D replay drawn from the recorded joint angles. Drag to orbit. Feet turn yellow on contact and leave prints; the dashed line is the path of the body.
- **Footfall.** The gait diagram for the replay, which doubles as the timeline. An alternating pattern across the eight rows is a real gait; solid bars mean it is dragging its feet.
- **Where the colony stands.** The curriculum drawn as a staircase to scale, with the share of spiders practising each step height. Scrub the iteration slider to watch the population move up.
- **Reward ledger.** What each reward term pays or costs per second at the selected iteration.

## What to watch

- `Episode/terrain_level` is the real progress metric (the staircase panel in the dashboard). It starts near 1 and should rise as the population clears harder stairs. Mean reward can dip while the level rises, because harder stairs pay less.
- If the spider shuffles without lifting its feet, raise `feet_air_time`.
- If it stalls at one level, look at `rew_feet_stumble` and `rew_collision`: high values mean legs are catching on risers. Widening `scan_x` forward or raising `action_scale` helps.
- If the gait is jittery, raise `action_rate` and `dof_acc` penalties.

## Tested with

genesis-world 1.4.6, rsl-rl-lib 5.5.2, torch 2.15, Python 3.13.
