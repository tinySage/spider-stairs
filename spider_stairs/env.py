"""Parallel Genesis environment: an 8-legged spider learning to climb stairs.

Follows the structure of Genesis' own `examples/locomotion/go2_env.py` (and so Legged Gym),
extended with a stair terrain, a terrain height scan, contact-based rewards, an
asymmetric critic and a game-inspired terrain curriculum.
"""

from __future__ import annotations

import math

import torch
from tensordict import TensorDict

import genesis as gs
from genesis.utils.geom import inv_quat, transform_by_quat

from .robot import ASSET_PATH, SpiderSpec, leg_names, write_mjcf
from .terrain import TerrainCfg, build_terrain


def _rand(shape, lo, hi):
    return (hi - lo) * torch.rand(shape, dtype=gs.tc_float, device=gs.device) + lo


def _wrap_to_pi(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class SpiderStairsEnv:
    def __init__(self, num_envs: int, cfgs: dict, show_viewer: bool = False, camera: dict | None = None):
        self.num_envs = num_envs
        self.env_cfg, self.obs_cfg = cfgs["env"], cfgs["obs"]
        self.reward_cfg, self.command_cfg = cfgs["reward"], cfgs["command"]
        self.cfg = cfgs  # rsl-rl reads this
        self.device = gs.device
        self.num_actions = self.env_cfg["num_actions"]
        self.num_commands = self.command_cfg["num_commands"]
        self.dt = self.env_cfg["dt"]
        self.max_episode_length = math.ceil(self.env_cfg["episode_length_s"] / self.dt)
        self.simulate_action_latency = self.env_cfg["simulate_action_latency"]
        self.obs_scales = self.obs_cfg["obs_scales"]
        self.reward_scales = dict(self.reward_cfg["reward_scales"])
        N = num_envs

        # ------------------------------------------------------------------ scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=self.env_cfg["substeps"]),
            rigid_options=gs.options.RigidOptions(
                enable_self_collision=False,
                max_collision_pairs=self.env_cfg["max_collision_pairs"],
                tolerance=1e-5,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=[0]),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(-2.0, -2.0, 1.5), camera_lookat=(0.0, 0.0, 0.0), camera_fov=40
            ),
            show_viewer=show_viewer,
        )

        self.terrain_cfg = TerrainCfg(**self.env_cfg["terrain"])
        self.terrain = build_terrain(self.terrain_cfg)
        self.scene.add_entity(
            gs.morphs.Terrain(
                pos=self.terrain.pos,
                horizontal_scale=self.terrain_cfg.horizontal_scale,
                vertical_scale=self.terrain_cfg.vertical_scale,
                height_field=self.terrain.height_field,
            )
        )

        if not ASSET_PATH.exists():
            write_mjcf(ASSET_PATH)
        self.robot = self.scene.add_entity(gs.morphs.MJCF(file=str(ASSET_PATH), pos=(0.0, 0.0, 1.0)))

        self.camera = None
        if camera is not None:
            self.camera = self.scene.add_camera(
                res=camera.get("res", (1280, 720)), pos=(2, 2, 1.5), lookat=(0, 0, 0), fov=40, GUI=False
            )

        self.scene.build(n_envs=N)

        # ------------------------------------------------------------- indices
        self.motors_dof_idx = torch.tensor(
            [self.robot.get_joint(name).dof_start for name in self.env_cfg["joint_names"]],
            dtype=gs.tc_int,
            device=gs.device,
        )
        legs = leg_names()
        link_idx = lambda names: torch.tensor(  # noqa: E731
            [self.robot.get_link(n).idx_local for n in names], dtype=gs.tc_int, device=gs.device
        )
        self.base_link_idx = link_idx(["base"])
        self.feet_link_idx = link_idx([f"{leg}_tibia" for leg in legs])  # tibia link carries the foot
        self.penalised_link_idx = link_idx([f"{leg}_{seg}" for leg in legs for seg in ("coxa", "femur")])
        self.num_feet = len(legs)

        # ------------------------------------------------------------- actuators
        n = self.num_actions
        self.robot.set_dofs_kp([self.env_cfg["kp"]] * n, self.motors_dof_idx)
        self.robot.set_dofs_kv([self.env_cfg["kd"]] * n, self.motors_dof_idx)
        tl = self.env_cfg["torque_limit"]
        self.robot.set_dofs_force_range([-tl] * n, [tl] * n, self.motors_dof_idx)
        lower, upper = self.robot.get_dofs_limit(self.motors_dof_idx)
        lower, upper = lower.reshape(-1, n)[0], upper.reshape(-1, n)[0]
        mid, half = (lower + upper) / 2, (upper - lower) / 2 * self.reward_cfg["soft_dof_pos_limit"]
        self.soft_dof_lower, self.soft_dof_upper = mid - half, mid + half

        # ------------------------------------------------------- randomisation
        lo, hi = self.env_cfg["friction_range"]
        self.friction_ratio = _rand((N, 1), lo, hi)
        if hi > lo:
            self.robot.set_friction_ratio(
                self.friction_ratio.expand(N, self.robot.n_links), list(range(self.robot.n_links))
            )
        self.push_interval = math.ceil(self.env_cfg["push_interval_s"] / self.dt)
        self.common_step_counter = 0

        # --------------------------------------------------------------- terrain
        tc = self.terrain_cfg
        self.height_map = torch.tensor(self.terrain.heights_m, dtype=gs.tc_float, device=gs.device)
        self.tile_centers = torch.tensor(self.terrain.tile_centers, dtype=gs.tc_float, device=gs.device)
        self.level_step_height = torch.tensor(self.terrain.step_heights, dtype=gs.tc_float, device=gs.device)
        self.max_level = tc.n_levels - 1
        self.success_dist = tc.platform_half + tc.stair_run + 0.1
        self.fail_dist = tc.platform_half + 0.5 * tc.stair_run
        self.terrain_col = torch.arange(N, device=gs.device) % tc.n_cols
        self.terrain_level = torch.randint(
            0, min(self.env_cfg["init_max_level"], self.max_level) + 1, (N,), device=gs.device
        )
        if not self.env_cfg["curriculum"]:
            # evaluation: a fixed level if given, otherwise spread envs over every tile
            self.terrain_level = (torch.arange(N, device=gs.device) // tc.n_cols) % tc.n_levels
            if self.env_cfg.get("eval_level") is not None:
                self.terrain_level[:] = min(int(self.env_cfg["eval_level"]), self.max_level)
        self.env_origins = self.tile_centers[self.terrain_level, self.terrain_col]

        sx, sy = self.obs_cfg["scan_x"], self.obs_cfg["scan_y"]
        gx, gy = torch.meshgrid(
            torch.linspace(sx[0], sx[1], sx[2], device=gs.device),
            torch.linspace(sy[0], sy[1], sy[2], device=gs.device),
            indexing="ij",
        )
        self.scan_points = torch.stack((gx.flatten(), gy.flatten()), dim=-1).to(gs.tc_float)  # (P, 2)
        self.num_scan = self.scan_points.shape[0]
        self.under_body = (self.scan_points[:, 0].abs() <= 0.21) & (self.scan_points[:, 1].abs() <= 0.11)

        # --------------------------------------------------------------- buffers
        f = dict(dtype=gs.tc_float, device=gs.device)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], **f)
        self.forward_vec = torch.tensor([1.0, 0.0, 0.0], **f).expand(N, 3)
        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["joint_names"]], **f
        )
        self.base_pos = torch.zeros((N, 3), **f)
        self.base_quat = torch.zeros((N, 4), **f)
        self.base_lin_vel = torch.zeros((N, 3), **f)
        self.base_ang_vel = torch.zeros((N, 3), **f)
        self.projected_gravity = torch.zeros((N, 3), **f)
        self.yaw = torch.zeros((N,), **f)
        self.commands = torch.zeros((N, self.num_commands), **f)
        self.heading_target = torch.zeros((N,), **f)
        self.commands_scale = torch.tensor(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]], **f
        )
        self.actions = torch.zeros((N, n), **f)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.torques = torch.zeros_like(self.actions)
        self.contact_forces = torch.zeros((N, self.robot.n_links, 3), **f)
        self.feet_contact = torch.zeros((N, self.num_feet), dtype=gs.tc_bool, device=gs.device)
        self.last_feet_contact = torch.zeros_like(self.feet_contact)
        self.feet_air_time = torch.zeros((N, self.num_feet), **f)
        # the foot sits at the far end of the tibia link (its x axis runs along the shin)
        self.foot_offset = torch.tensor([SpiderSpec().tibia[0], 0.0, 0.0], **f).expand(N, self.num_feet, 3)
        self.feet_pos = torch.zeros((N, self.num_feet, 3), **f)
        self.feet_vel = torch.zeros((N, self.num_feet, 3), **f)
        self.scan_heights = torch.zeros((N, self.num_scan), **f)
        self.rew_buf = torch.zeros((N,), **f)
        self.reset_buf = torch.ones((N,), dtype=gs.tc_bool, device=gs.device)
        self.terminated = torch.zeros((N,), dtype=gs.tc_bool, device=gs.device)
        self.episode_length_buf = torch.zeros((N,), dtype=gs.tc_int, device=gs.device)
        self.extras = {}
        self._ep_log, self._ep_sum = {}, {}
        self._ep_n = torch.zeros((), dtype=gs.tc_float, device=gs.device)
        self.success_mask = torch.zeros((N,), dtype=gs.tc_bool, device=gs.device)

        self.reward_functions, self.episode_sums = {}, {}
        for name in self.reward_scales:
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((N,), **f)

        self.reset()

    # ====================================================================== API
    def reset(self):
        self._reset_idx(None)
        self._refresh_state()
        self._update_observation()
        return self.get_observations()

    def get_observations(self):
        return TensorDict({"policy": self.obs_buf, "privileged": self.privileged_obs_buf}, batch_size=[self.num_envs])

    def step(self, actions):
        clip = self.env_cfg["clip_actions"]
        self.actions = torch.clip(actions, -clip, clip)
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        target = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        self.robot.control_dofs_position(target, self.motors_dof_idx)
        self.scene.step()

        self.episode_length_buf += 1
        self.common_step_counter += 1
        if self.env_cfg["max_push_vel_xy"] > 0 and self.common_step_counter % self.push_interval == 0:
            self._push_robots()

        self._refresh_state()

        # heading controller: steer back towards the direction chosen at spawn
        yaw_err = _wrap_to_pi(self.heading_target - self.yaw)
        m = self.command_cfg["max_yaw_rate"]
        self.commands[:, 2] = torch.clip(self.command_cfg["heading_gain"] * yaw_err, -m, m)

        # rewards
        self.rew_buf.zero_()
        for name, fn in self.reward_functions.items():
            if name == "termination":
                continue
            rew = fn() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # termination
        base_force = torch.norm(self.contact_forces[:, self.base_link_idx], dim=-1).amax(dim=1)
        self.terminated = base_force > self.env_cfg["terminate_on_base_contact_force"]
        self.terminated |= -self.projected_gravity[:, 2] < self.env_cfg["terminate_if_tilt_cos_below"]
        self.terminated |= self.scene.get_error_envs_mask()
        time_out = self.episode_length_buf > self.max_episode_length
        # reaching the far end of the staircase ends the episode as a success (bootstrapped)
        time_out |= self._distance_from_origin() > self.success_dist
        time_out &= ~self.terminated
        self.reset_buf = self.terminated | time_out
        self.extras["time_outs"] = time_out.to(gs.tc_float)

        if "termination" in self.reward_functions:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew

        self._reset_idx(self.reset_buf)
        # reset envs get a consistent state/observation for their first step
        self._refresh_state(count_air_time=False)
        self._update_observation()

        self.last_actions.copy_(self.actions)
        self.last_dof_vel.copy_(self.dof_vel)
        return self.get_observations(), self.rew_buf, self.reset_buf, self.extras

    # ================================================================ internals
    def _distance_from_origin(self):
        """Chebyshev distance from the tile centre: stairs are square rings."""
        return (self.base_pos[:, :2] - self.env_origins[:, :2]).abs().amax(dim=1)

    def _sample_heights(self, xy):
        """Terrain height (world z) at world points xy (..., 2), nearest grid vertex."""
        hs = self.terrain_cfg.horizontal_scale
        ix = torch.round((xy[..., 0] - self.terrain.pos[0]) / hs).long().clamp_(0, self.height_map.shape[0] - 1)
        iy = torch.round((xy[..., 1] - self.terrain.pos[1]) / hs).long().clamp_(0, self.height_map.shape[1] - 1)
        return self.height_map[ix, iy]

    def _refresh_state(self, count_air_time: bool = True):
        self.base_pos = self.robot.get_pos()
        self.base_quat = self.robot.get_quat()
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        fwd = transform_by_quat(self.forward_vec, self.base_quat)
        self.yaw = torch.atan2(fwd[:, 1], fwd[:, 0])
        self.dof_pos = self.robot.get_dofs_position(self.motors_dof_idx)
        self.dof_vel = self.robot.get_dofs_velocity(self.motors_dof_idx)
        self.torques = self.robot.get_dofs_control_force(self.motors_dof_idx)
        self.contact_forces = self.robot.get_links_net_contact_force()

        thr = self.reward_cfg["contact_force_threshold"]
        contact = self.contact_forces[:, self.feet_link_idx, 2] > thr
        self.feet_contact = contact | self.last_feet_contact  # filter single-step dropouts
        self.last_feet_contact = contact
        feet_pos = self.robot.get_links_pos(self.feet_link_idx) + transform_by_quat(
            self.foot_offset, self.robot.get_links_quat(self.feet_link_idx)
        )
        if count_air_time:
            self.first_contact = (self.feet_air_time > 0.0) & self.feet_contact
            self.feet_air_time += self.dt
            self.feet_vel = (feet_pos - self.feet_pos) / self.dt
        self.feet_pos = feet_pos  # the post-reset refresh only re-bases this, so teleports give no velocity

        # terrain scan in the yaw-aligned frame
        c, s = torch.cos(self.yaw)[:, None], torch.sin(self.yaw)[:, None]
        px, py = self.scan_points[None, :, 0], self.scan_points[None, :, 1]
        xy = torch.stack((c * px - s * py, s * px + c * py), dim=-1) + self.base_pos[:, None, :2]
        self.scan_heights = self._sample_heights(xy)

    def _push_robots(self):
        m = self.env_cfg["max_push_vel_xy"]
        vel = self.robot.get_dofs_velocity([0, 1])
        self.robot.set_dofs_velocity(vel + _rand((self.num_envs, 2), -m, m), [0, 1])

    def _update_curriculum(self, mask):
        dist = self._distance_from_origin()
        up = mask & (dist > self.success_dist)
        self.success_mask = up
        down = mask & (dist < self.fail_dist) & ~up
        if not self.env_cfg["curriculum"]:
            return
        level = self.terrain_level + up.long() - down.long()
        # graduates of the hardest level are recycled to a random level to keep coverage
        level = torch.where(level > self.max_level, torch.randint_like(level, 0, self.max_level + 1), level)
        self.terrain_level = level.clamp_(min=0)
        self.env_origins = self.tile_centers[self.terrain_level, self.terrain_col]

    def _reset_idx(self, envs_idx=None):
        N = self.num_envs
        mask = torch.ones((N,), dtype=gs.tc_bool, device=gs.device) if envs_idx is None else envs_idx
        if envs_idx is not None:
            self._update_curriculum(mask)

        # spawn on the tile's central platform with random yaw; walk in a random direction
        j = self.env_cfg["spawn_xy_jitter"]
        pos = self.env_origins.clone()
        pos[:, :2] += _rand((N, 2), -j, j)
        pos[:, 2] += self.env_cfg["spawn_height"]
        yaw = _rand((N,), -math.pi, math.pi)
        quat = torch.zeros((N, 4), dtype=gs.tc_float, device=gs.device)
        quat[:, 0], quat[:, 3] = torch.cos(yaw / 2), torch.sin(yaw / 2)
        dof = self.default_dof_pos + _rand((N, self.num_actions), -0.1, 0.1)
        qpos = torch.cat((pos, quat, dof), dim=-1)
        self.robot.set_qpos(qpos[mask], envs_idx=mask, zero_velocity=True)

        heading = _rand((N,), -math.pi, math.pi)
        vx = _rand((N,), *self.command_cfg["lin_vel_x_range"])
        self.heading_target = torch.where(mask, heading, self.heading_target)
        self.commands[:, 0] = torch.where(mask, vx, self.commands[:, 0])
        self.commands[:, 1:].masked_fill_(mask[:, None], 0.0)

        self.actions.masked_fill_(mask[:, None], 0.0)
        self.last_actions.masked_fill_(mask[:, None], 0.0)
        self.last_dof_vel.masked_fill_(mask[:, None], 0.0)
        self.feet_air_time.masked_fill_(mask[:, None], 0.0)
        self.last_feet_contact.masked_fill_(mask[:, None], False)
        self.episode_length_buf.masked_fill_(mask, 0)

        # logging: episode means over finished episodes, as count-weighted running averages (about the last
        # 500 steps). Averaging per step instead would over-weight the steps on which only a few envs reset.
        n = mask.sum() if envs_idx is not None else mask.sum() * 0  # the very first reset ends no episode
        decay = 0.998
        self._ep_n = decay * self._ep_n + n
        vals = {"success_rate": self.success_mask.sum() if envs_idx is not None else n * 0.0}
        for key, value in self.episode_sums.items():
            vals["rew_" + key] = (value * mask).sum() / self.env_cfg["episode_length_s"]
            value.masked_fill_(mask, 0.0)
        for key, value in vals.items():
            self._ep_sum[key] = decay * self._ep_sum.get(key, 0.0) + value
            self._ep_log[key] = self._ep_sum[key] / self._ep_n.clamp(min=1.0)
        self._ep_log["terrain_level"] = self.terrain_level.float().mean()
        self._ep_log["step_height_cm"] = 100 * self.level_step_height[self.terrain_level].mean()
        pop = torch.bincount(self.terrain_level, minlength=self.max_level + 1).float() / self.num_envs
        for lv in range(self.max_level + 1):  # population on each level, for the dashboard staircase
            self._ep_log[f"level_pop_{lv}"] = pop[lv]
        self.extras["episode"] = dict(self._ep_log)

    def _update_observation(self):
        s, oc = self.obs_scales, self.obs_cfg
        target_h = self.reward_cfg["base_height_target"]
        clip = oc["height_clip"]
        heights = torch.clip(self.base_pos[:, 2:3] - target_h - self.scan_heights, -clip, clip) * s["height"]
        parts = [
            self.base_ang_vel * s["ang_vel"],  # 3
            self.projected_gravity,  # 3
            self.commands * self.commands_scale,  # 3
            (self.dof_pos - self.default_dof_pos) * s["dof_pos"],  # 24
            self.dof_vel * s["dof_vel"],  # 24
            self.actions,  # 24
            heights,  # 77
        ]
        if oc["add_noise"]:
            nz = oc["noise"]
            amp = [nz["ang_vel"] * s["ang_vel"], nz["gravity"], 0.0, nz["dof_pos"] * s["dof_pos"],
                   nz["dof_vel"] * s["dof_vel"], 0.0, nz["height"] * s["height"]]  # fmt: skip
            parts = [p + a * (2 * torch.rand_like(p) - 1) if a > 0 else p for p, a in zip(parts, amp)]
        self.obs_buf = torch.cat(parts, dim=-1)
        self.privileged_obs_buf = torch.cat(
            (
                self.base_lin_vel * s["lin_vel"],  # 3
                self.feet_contact.to(gs.tc_float),  # 8
                self.friction_ratio,  # 1
                self.level_step_height[self.terrain_level, None] * 10.0,  # 1
            ),
            dim=-1,
        )

    # ================================================================== rewards
    def _reward_tracking_lin_vel(self):
        err = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-err / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_ang_vel(self):
        err = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-err / self.reward_cfg["tracking_sigma_yaw"])

    def _reward_forward_progress(self):
        cmd = self.commands[:, 0]
        return torch.clip(torch.minimum(self.base_lin_vel[:, 0], cmd) / cmd, min=-1.0)

    def _reward_lin_vel_z(self):
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        ground = self.scan_heights[:, self.under_body].mean(dim=1)
        return torch.square(self.base_pos[:, 2] - ground - self.reward_cfg["base_height_target"])

    def _reward_feet_air_time(self):
        # reward proper swings (not shuffling); paid out on touchdown
        target = self.reward_cfg["feet_air_time_target"]
        rew = torch.sum((self.feet_air_time - target) * self.first_contact, dim=1)
        self.feet_air_time *= ~self.feet_contact
        return rew * (torch.norm(self.commands[:, :2], dim=1) > 0.1)

    def _reward_feet_slip(self):
        # a planted foot must not slide. Without this the cheapest way to move is to jitter every joint and
        # ratchet along on friction, which only works while exploration noise is switched on.
        slip = torch.sum(torch.square(self.feet_vel[..., :2]), dim=-1)
        return torch.sum(slip * self.feet_contact, dim=1)

    def _reward_feet_stumble(self):
        # legs slamming into risers: horizontal contact force dominating vertical
        f = self.contact_forces[:, self.feet_link_idx]
        return torch.any(torch.norm(f[..., :2], dim=-1) > 4.0 * f[..., 2].abs() + 1.0, dim=1).to(gs.tc_float)

    def _reward_collision(self):
        f = torch.norm(self.contact_forces[:, self.penalised_link_idx], dim=-1)
        return torch.sum((f > 0.1).to(gs.tc_float), dim=1)

    def _reward_torques(self):
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_acc(self):
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    def _reward_action_rate(self):
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_dof_pos_limits(self):
        out = -(self.dof_pos - self.soft_dof_lower).clip(max=0.0) + (self.dof_pos - self.soft_dof_upper).clip(min=0.0)
        return torch.sum(out, dim=1)

    def _reward_termination(self):
        return self.terminated.to(gs.tc_float)
