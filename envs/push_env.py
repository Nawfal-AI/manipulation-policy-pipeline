"""
A lightweight 2D "pick object up and carry it to a goal" manipulation task.

Deliberately not MuJoCo/robosuite: those need offscreen-rendering deps that are
fragile in headless Colab/Kaggle environments. This environment has zero
external dependencies beyond numpy, runs anywhere (including CPU), and still
exercises the same core loop a real manipulation policy has to learn:
observation -> action -> contact/grasp dynamics -> task success.

State:
    end-effector position (2,), object position (2,), goal position (2,)
Action:
    (dx, dy) end-effector velocity command, clipped to [-max_step, max_step]
Dynamics:
    A simple rigid-grasp model: when the end-effector moves within
    `grasp_radius` of the object, the object snaps to the end-effector's
    position (as if picked up) and moves with it every subsequent step while
    still within range. This avoids the directional-approach sensitivity of a
    true pushing model, which made early testing show the scripted expert
    dragging objects the wrong way when approaching from certain angles.
Success:
    object within `success_radius` of the goal at any point in the episode.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class PushEnvConfig:
    max_step: float = 0.05
    grasp_radius: float = 0.08
    success_radius: float = 0.06
    horizon: int = 75
    world_size: float = 1.0  # workspace is [-world_size, world_size]^2
    seed: int = 0
    # Optional static circular obstacle. obstacle_radius=0.0 disables it
    # (default), keeping the baseline task's behavior unchanged.
    obstacle_pos: tuple = (0.0, 0.0)
    obstacle_radius: float = 0.0


def _clip_from_obstacle(pos: np.ndarray, obstacle_pos: np.ndarray, obstacle_radius: float) -> np.ndarray:
    """Radially project a point back outside the obstacle if it landed inside."""
    if obstacle_radius <= 0:
        return pos
    offset = pos - obstacle_pos
    dist = np.linalg.norm(offset)
    if dist < obstacle_radius and dist > 1e-8:
        pos = obstacle_pos + offset / dist * obstacle_radius
    return pos


class PushEnv:
    """A minimal gym-like environment: reset() -> obs, step(action) -> obs, reward, done, info."""

    def __init__(self, config: PushEnvConfig = None):
        self.cfg = config or PushEnvConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.t = 0
        self.ee_pos = np.zeros(2)
        self.object_pos = np.zeros(2)
        self.goal_pos = np.zeros(2)
        self.grasped = False

    def seed(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def reset(self) -> np.ndarray:
        self.t = 0
        self.grasped = False
        ws = self.cfg.world_size
        obstacle_pos = np.asarray(self.cfg.obstacle_pos)
        margin = self.cfg.obstacle_radius + 0.05

        def sample_outside_obstacle(low_scale):
            while True:
                p = self.rng.uniform(-ws * low_scale, ws * low_scale, size=2)
                if self.cfg.obstacle_radius <= 0 or np.linalg.norm(p - obstacle_pos) > margin:
                    return p

        self.ee_pos = sample_outside_obstacle(1.0)
        self.object_pos = sample_outside_obstacle(0.6)
        while True:
            self.goal_pos = sample_outside_obstacle(0.6)
            if np.linalg.norm(self.goal_pos - self.object_pos) > self.cfg.success_radius * 3:
                break
        return self._obs()

    def _obs(self) -> np.ndarray:
        base = np.concatenate(
            [self.ee_pos, self.object_pos, self.goal_pos, self.object_pos - self.goal_pos]
        )
        if self.cfg.obstacle_radius > 0:
            base = np.concatenate([base, np.asarray(self.cfg.obstacle_pos), [self.cfg.obstacle_radius]])
        return base.astype(np.float32)

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float64), -self.cfg.max_step, self.cfg.max_step)
        new_ee_pos = np.clip(self.ee_pos + action, -self.cfg.world_size, self.cfg.world_size)
        new_ee_pos = _clip_from_obstacle(
            new_ee_pos, np.asarray(self.cfg.obstacle_pos), self.cfg.obstacle_radius
        )

        if np.linalg.norm(new_ee_pos - self.object_pos) < self.cfg.grasp_radius:
            self.grasped = True
        if self.grasped:
            self.object_pos = new_ee_pos.copy()

        self.ee_pos = new_ee_pos
        self.t += 1

        dist_to_goal = np.linalg.norm(self.object_pos - self.goal_pos)
        success = dist_to_goal < self.cfg.success_radius
        done = success or self.t >= self.cfg.horizon
        reward = -dist_to_goal  # dense reward, unused by BC/diffusion training but useful for debugging
        info = {"success": success, "dist_to_goal": float(dist_to_goal), "grasped": self.grasped}
        return self._obs(), reward, done, info


class ScriptedExpert:
    """Two-phase controller: move to the object, then (while carrying it) move to the goal."""

    def __init__(self, config: PushEnvConfig = None, noise_std: float = 0.0):
        self.cfg = config or PushEnvConfig()
        self.noise_std = noise_std
        self.rng = np.random.default_rng()

    def act(self, env: PushEnv) -> np.ndarray:
        target = env.goal_pos if env.grasped else env.object_pos
        if env.cfg.obstacle_radius > 0:
            target = self._detour(env.ee_pos, target, np.asarray(env.cfg.obstacle_pos), env.cfg.obstacle_radius)
        delta = target - env.ee_pos
        norm = np.linalg.norm(delta)
        if norm > self.cfg.max_step:
            delta = delta / norm * self.cfg.max_step
        if self.noise_std > 0:
            delta = delta + self.rng.normal(0, self.noise_std, size=2)
        return np.clip(delta, -self.cfg.max_step, self.cfg.max_step)

    @staticmethod
    def _detour(ee_pos: np.ndarray, target: np.ndarray, obstacle_pos: np.ndarray, obstacle_radius: float) -> np.ndarray:
        """If the direct path to target would pass through the obstacle,
        redirect to a waypoint tangent to the obstacle instead."""
        clearance = obstacle_radius + 0.06
        d = target - ee_pos
        length = np.linalg.norm(d)
        if length < 1e-6:
            return target
        d_unit = d / length
        to_obstacle = obstacle_pos - ee_pos
        t = np.clip(np.dot(to_obstacle, d_unit), 0.0, length)
        closest_point = ee_pos + d_unit * t
        dist_to_obstacle = np.linalg.norm(closest_point - obstacle_pos)
        if dist_to_obstacle >= clearance:
            return target  # path is already clear
        perp = np.array([-d_unit[1], d_unit[0]])
        side = np.sign(np.dot(ee_pos - obstacle_pos, perp))
        side = side if side != 0 else 1.0
        return obstacle_pos + perp * side * clearance
