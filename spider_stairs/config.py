"""All tunables in one place. Behaviour is tuned here, not in env.py."""

from __future__ import annotations

from dataclasses import asdict

from .robot import default_joint_angles, joint_names, nominal_stance_height
from .terrain import TerrainCfg


def get_cfgs() -> dict:
    env_cfg = {
        "num_actions": 24,
        "joint_names": joint_names(),
        "default_joint_angles": default_joint_angles(),
        # sim: 50 Hz policy, 200 Hz physics
        "dt": 0.02,
        "substeps": 4,
        "max_collision_pairs": 64,
        # PD position control
        "kp": 25.0,
        "kd": 0.6,
        "torque_limit": 6.0,
        "action_scale": 0.3,
        "clip_actions": 10.0,
        "simulate_action_latency": True,
        # episodes
        "episode_length_s": 20.0,
        "spawn_height": nominal_stance_height() + 0.03,
        "spawn_xy_jitter": 0.25,
        # termination
        "terminate_on_base_contact_force": 1.0,  # N
        "terminate_if_tilt_cos_below": 0.3,  # body z-axis vs world up (~72 deg)
        # domain randomisation
        "friction_range": [0.6, 1.3],
        "push_interval_s": 7.0,
        "max_push_vel_xy": 0.4,
        "terrain": asdict(TerrainCfg()),
        # curriculum
        "curriculum": True,
        "init_max_level": 2,  # envs start spread over levels [0, init_max_level]
    }
    obs_cfg = {
        "obs_scales": {"lin_vel": 2.0, "ang_vel": 0.25, "dof_pos": 1.0, "dof_vel": 0.05, "height": 5.0},
        # terrain scan in the yaw-aligned body frame: the spider "feels" the steps ahead
        "scan_x": [-0.4, 0.6, 11],  # start, stop, count  [m]
        "scan_y": [-0.3, 0.3, 7],
        "height_clip": 1.0,
        "add_noise": True,
        "noise": {"ang_vel": 0.2, "gravity": 0.05, "dof_pos": 0.01, "dof_vel": 1.5, "height": 0.02},
    }
    reward_cfg = {
        # The spider walks at 0.25-0.6 m/s, so the tracking kernel must be much tighter than the 0.25 used for
        # 1 m/s quadrupeds: with 0.25, standing still already earns 53-78% of the tracking reward.
        "tracking_sigma": 0.05,
        "tracking_sigma_yaw": 0.25,
        "base_height_target": nominal_stance_height(),
        "feet_air_time_target": 0.15,
        "contact_force_threshold": 1.0,
        "soft_dof_pos_limit": 0.9,
        "reward_scales": {
            # task
            "tracking_lin_vel": 1.5,
            "forward_progress": 1.0,  # dense: fraction of the commanded speed achieved, so "stand still" is never optimal
            "tracking_ang_vel": 0.75,
            # stability (kept light: climbing needs the body to pitch and heave)
            "lin_vel_z": -0.5,
            "ang_vel_xy": -0.05,
            "orientation": -0.5,
            "base_height": -5.0,
            # gait quality
            "feet_air_time": 2.0,
            "feet_slip": -0.5,
            "feet_stumble": -0.5,
            "collision": -1.0,
            # effort / smoothness
            "torques": -2e-4,
            "dof_acc": -2.5e-7,
            "action_rate": -0.01,
            "dof_pos_limits": -5.0,
            "termination": -2.0,
        },
    }
    command_cfg = {
        "num_commands": 3,  # vx, vy, yaw rate (yaw rate is driven by a heading target)
        "lin_vel_x_range": [0.25, 0.6],
        "heading_gain": 0.8,
        "max_yaw_rate": 1.0,
    }
    return {"env": env_cfg, "obs": obs_cfg, "reward": reward_cfg, "command": command_cfg}


def get_train_cfg(exp_name: str) -> dict:
    net = {"class_name": "MLPModel", "hidden_dims": [512, 256, 128], "activation": "elu", "obs_normalization": True}
    return {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.005,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 1e-3,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "actor": {
            **net,
            "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 0.5, "std_type": "scalar"},
        },
        "critic": dict(net),
        # asymmetric actor-critic: the critic also sees true velocity, contacts and clean heights
        "obs_groups": {"actor": ["policy"], "critic": ["policy", "privileged"]},
        "num_steps_per_env": 24,
        "save_interval": 100,
        "run_name": exp_name,
        "logger": "tensorboard",
    }
