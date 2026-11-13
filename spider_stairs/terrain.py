"""Stair curriculum terrain.

A grid of square tiles. Rows (x) are difficulty levels: step height grows with the row.
Columns (y) are tile types:

    "ascend"  : a pit. The spider spawns on the floor in the middle and must climb UP
                the stairs to reach the rim.
    "descend" : a pyramid. The spider spawns on the summit and must climb DOWN.

Every tile's rim sits at z = 0, so tiles join seamlessly. Steps are square rings, so a
spider walking out in any direction meets the same staircase (diagonals cross corners).

The height field is built here rather than with Genesis' built-in sub-terrains so that the
env knows exactly where every tile, platform and step is, and can sample heights on-device.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TerrainCfg:
    n_levels: int = 8
    col_types: tuple[str, ...] = ("ascend", "ascend", "descend", "ascend", "ascend", "descend")
    tile_size: float = 6.0
    horizontal_scale: float = 0.05  # m per cell
    vertical_scale: float = 0.005  # m per height unit
    step_depth: float = 0.30
    step_height_range: tuple[float, float] = (0.02, 0.12)
    platform_half: float = 0.7  # half-width of the central spawn platform
    rim: float = 0.4  # flat band at the outside of each tile
    border: float = 1.0  # flat margin around the whole grid

    @property
    def n_cols(self) -> int:
        return len(self.col_types)

    @property
    def n_steps(self) -> int:
        run = self.tile_size / 2 - self.platform_half - self.rim
        return int(run / self.step_depth + 1e-6)

    @property
    def stair_run(self) -> float:
        """Horizontal distance from the platform edge to the top/bottom of the stairs."""
        return self.n_steps * self.step_depth

    def step_height(self, level: int) -> float:
        lo, hi = self.step_height_range
        if self.n_levels == 1:
            return hi
        return lo + (hi - lo) * level / (self.n_levels - 1)


@dataclass
class Terrain:
    cfg: TerrainCfg
    height_field: np.ndarray  # int, (nx, ny), in vertical_scale units, min == 0
    pos: tuple[float, float, float]  # world position of cell (0, 0)
    tile_centers: np.ndarray  # (n_levels, n_cols, 3) world xyz of each spawn platform
    step_heights: np.ndarray  # (n_levels,)

    @property
    def heights_m(self) -> np.ndarray:
        """Heights in metres, in world z."""
        return self.height_field * self.cfg.vertical_scale + self.pos[2]


def build_terrain(cfg: TerrainCfg) -> Terrain:
    hs, vs = cfg.horizontal_scale, cfg.vertical_scale
    tile_n = int(round(cfg.tile_size / hs))
    border_n = int(round(cfg.border / hs))
    nx = cfg.n_levels * tile_n + 2 * border_n
    ny = cfg.n_cols * tile_n + 2 * border_n
    hf = np.zeros((nx, ny), dtype=np.float64)  # metres for now

    # Chebyshev distance of each cell centre from the tile centre -> square rings
    c = (np.arange(tile_n) + 0.5) * hs - cfg.tile_size / 2
    cheb = np.maximum(np.abs(c)[:, None], np.abs(c)[None, :])
    ring = np.clip(np.ceil((cheb - cfg.platform_half) / cfg.step_depth - 1e-6), 0, cfg.n_steps)

    centers = np.zeros((cfg.n_levels, cfg.n_cols, 3))
    step_heights = np.array([cfg.step_height(lv) for lv in range(cfg.n_levels)])
    for lv in range(cfg.n_levels):
        # snap to the vertical resolution so every riser is identical
        h = round(step_heights[lv] / vs) * vs
        step_heights[lv] = h
        for col, kind in enumerate(cfg.col_types):
            if kind == "ascend":
                tile = (ring - cfg.n_steps) * h  # floor at -n*h, rim at 0
            elif kind == "descend":
                tile = (cfg.n_steps - ring) * h  # summit at +n*h, rim at 0
            elif kind == "flat":
                tile = np.zeros_like(ring)
            else:
                raise ValueError(f"unknown tile type {kind!r}")
            x0, y0 = border_n + lv * tile_n, border_n + col * tile_n
            hf[x0 : x0 + tile_n, y0 : y0 + tile_n] = tile
            centers[lv, col] = (x0 + tile_n / 2, y0 + tile_n / 2, tile[tile_n // 2, tile_n // 2])

    # shift so the field is non-negative; the morph's z offset puts the rims back at z = 0
    z_min = hf.min()
    hf_int = np.round((hf - z_min) / vs).astype(np.int32)
    # centre the whole grid on the world origin
    pos = (-nx * hs / 2, -ny * hs / 2, float(z_min))
    centers[..., 0] = centers[..., 0] * hs + pos[0]
    centers[..., 1] = centers[..., 1] * hs + pos[1]
    return Terrain(cfg=cfg, height_field=hf_int, pos=pos, tile_centers=centers, step_heights=step_heights)
