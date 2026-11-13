"""Procedural 8-legged spider (24 actuated DOF), emitted as MJCF.

Each leg is a 3-link chain in its own frame (x points outward from the body):

    coxa  : yaw   about z   -> swings the leg fore/aft   (positive = rearward on both sides)
    femur : pitch about y   -> lifts / lowers the knee   (positive = down)
    tibia : pitch about y   -> folds the shin            (positive = down / inward)

The zero pose is all legs straight out and flat; `DEFAULT_POSE` is the high-knee
stance a real spider uses, which is also what gives it clearance over stair edges.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

SIDES = ("L", "R")
LEG_IDS = (1, 2, 3, 4)  # front -> back
SEGMENTS = ("coxa", "femur", "tibia")

# femur up 0.5 rad, tibia 1.2 rad below horizontal -> feet ~0.24 m out, ~0.14 m below the hip
DEFAULT_POSE = {"coxa": 0.0, "femur": -0.5, "tibia": 1.7}


@dataclass
class SpiderSpec:
    # body
    body_radii: tuple[float, float, float] = (0.11, 0.08, 0.04)  # cephalothorax ellipsoid
    body_mass: float = 1.2
    abdomen_radius: float = 0.07
    abdomen_half_len: float = 0.045
    abdomen_mass: float = 0.35
    # leg mounts: angle from +x (forward), mirrored for the right side
    mount_angles_deg: tuple[float, ...] = (35.0, 70.0, 110.0, 145.0)
    mount_ellipse: tuple[float, float] = (0.095, 0.07)
    # leg segments (length, radius, mass)
    coxa: tuple[float, float, float] = (0.04, 0.016, 0.05)
    femur: tuple[float, float, float] = (0.14, 0.013, 0.08)
    tibia: tuple[float, float, float] = (0.22, 0.010, 0.05)
    foot_radius: float = 0.016
    foot_mass: float = 0.015
    # joints
    ranges: dict = field(
        default_factory=lambda: {"coxa": (-0.6, 0.6), "femur": (-1.5, 0.6), "tibia": (0.0, 2.6)}
    )
    damping: float = 0.05
    armature: float = 0.01
    frictionloss: float = 0.01
    friction: float = 1.0
    spawn_height: float = 0.17


def leg_names() -> list[str]:
    return [f"{s}{i}" for s in SIDES for i in LEG_IDS]


def joint_names() -> list[str]:
    """Actuated joints, in the order they appear in the MJCF (and in the action vector)."""
    return [f"{leg}_{seg}" for leg in leg_names() for seg in SEGMENTS]


def default_joint_angles() -> dict[str, float]:
    return {f"{leg}_{seg}": DEFAULT_POSE[seg] for leg in leg_names() for seg in SEGMENTS}


def nominal_stance_height(spec: SpiderSpec | None = None) -> float:
    """Height of the body centre above flat ground in the default pose."""
    spec = spec or SpiderSpec()
    f, t = DEFAULT_POSE["femur"], DEFAULT_POSE["femur"] + DEFAULT_POSE["tibia"]
    drop = spec.femur[0] * math.sin(f) + spec.tibia[0] * math.sin(t)
    return drop + spec.foot_radius


def _leg_xml(name: str, side: str, angle_deg: float, spec: SpiderSpec) -> str:
    sign = 1.0 if side == "L" else -1.0
    ang = math.radians(sign * angle_deg)
    a, b = spec.mount_ellipse
    mx, my = a * math.cos(ang), b * math.sin(ang)
    qw, qz = math.cos(ang / 2), math.sin(ang / 2)
    (cl, cr, cm), (fl, fr, fm), (tl, tr, tm) = spec.coxa, spec.femur, spec.tibia
    r = spec.ranges
    yaw_axis = f"0 0 {-sign:.0f}"  # positive coxa angle sweeps the leg rearward on both sides
    return f"""
      <body name="{name}_coxa" pos="{mx:.5f} {my:.5f} 0" quat="{qw:.6f} 0 0 {qz:.6f}">
        <joint name="{name}_coxa" axis="{yaw_axis}" range="{r['coxa'][0]} {r['coxa'][1]}"/>
        <geom name="{name}_coxa" type="capsule" fromto="0 0 0 {cl} 0 0" size="{cr}" mass="{cm}" class="leg"/>
        <body name="{name}_femur" pos="{cl} 0 0">
          <joint name="{name}_femur" axis="0 1 0" range="{r['femur'][0]} {r['femur'][1]}"/>
          <geom name="{name}_femur" type="capsule" fromto="0 0 0 {fl} 0 0" size="{fr}" mass="{fm}" class="leg"/>
          <body name="{name}_tibia" pos="{fl} 0 0">
            <joint name="{name}_tibia" axis="0 1 0" range="{r['tibia'][0]} {r['tibia'][1]}"/>
            <geom name="{name}_tibia" type="capsule" fromto="0 0 0 {tl} 0 0" size="{tr}" mass="{tm}" class="leg"/>
            <geom name="{name}_foot" type="sphere" pos="{tl} 0 0" size="{spec.foot_radius}" mass="{spec.foot_mass}" class="foot"/>
          </body>
        </body>
      </body>"""


def build_mjcf(spec: SpiderSpec | None = None) -> str:
    spec = spec or SpiderSpec()
    bx, by, bz = spec.body_radii
    legs = "".join(
        _leg_xml(f"{side}{i}", side, angle, spec)
        for side in SIDES
        for i, angle in zip(LEG_IDS, spec.mount_angles_deg)
    )
    ab_x = -(bx + spec.abdomen_half_len * 0.6)
    return f"""<mujoco model="spider">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.005"/>
  <default>
    <joint type="hinge" damping="{spec.damping}" armature="{spec.armature}" frictionloss="{spec.frictionloss}"/>
    <geom friction="{spec.friction} 0.005 0.0001" condim="3"/>
    <default class="body"><geom rgba="0.13 0.11 0.10 1"/></default>
    <default class="leg"><geom rgba="0.20 0.17 0.15 1"/></default>
    <default class="foot"><geom rgba="0.85 0.35 0.10 1" friction="{spec.friction * 1.2} 0.005 0.0001"/></default>
  </default>
  <worldbody>
    <body name="base" pos="0 0 {spec.spawn_height}">
      <freejoint name="root"/>
      <geom name="cephalothorax" type="ellipsoid" size="{bx} {by} {bz}" mass="{spec.body_mass}" class="body"/>
      <geom name="abdomen" type="capsule" fromto="{ab_x + spec.abdomen_half_len} 0 0.025 {ab_x - spec.abdomen_half_len} 0 0.035" size="{spec.abdomen_radius}" mass="{spec.abdomen_mass}" class="body"/>{legs}
    </body>
  </worldbody>
</mujoco>
"""


def write_mjcf(path: str | Path, spec: SpiderSpec | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_mjcf(spec))
    return path


ASSET_PATH = Path(__file__).resolve().parent.parent / "assets" / "spider.xml"

if __name__ == "__main__":
    out = write_mjcf(ASSET_PATH)
    print(f"wrote {out}  (stance height {nominal_stance_height():.3f} m, {len(joint_names())} joints)")
