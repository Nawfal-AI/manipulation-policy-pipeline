"""Basic tests — enough for CI to catch real regressions, not exhaustive coverage."""
import numpy as np

from envs.push_env import PushEnv, PushEnvConfig, ScriptedExpert


def test_env_reset_shapes():
    env = PushEnv(PushEnvConfig(seed=0))
    obs = env.reset()
    assert obs.shape == (8,)


def test_env_step_shapes():
    env = PushEnv(PushEnvConfig(seed=0))
    env.reset()
    obs, reward, done, info = env.step(np.array([0.01, 0.01]))
    assert obs.shape == (8,)
    assert isinstance(done, (bool, np.bool_))
    assert "success" in info


def test_scripted_expert_solves_task():
    """Regression test for the exact bug caught during development: the expert
    must actually reach the object and carry it to the goal, not just move
    around near it."""
    cfg = PushEnvConfig(seed=0)
    successes = 0
    trials = 50
    for ep in range(trials):
        env = PushEnv(cfg)
        env.seed(ep)
        expert = ScriptedExpert(cfg, noise_std=0.0)
        obs = env.reset()
        done = False
        info = {}
        while not done:
            action = expert.act(env)
            obs, _, done, info = env.step(action)
        successes += info["success"]
    success_rate = successes / trials
    assert success_rate > 0.95, f"Expert success rate dropped to {success_rate:.2%} — check env dynamics"


def test_action_clipping():
    cfg = PushEnvConfig(max_step=0.05, seed=0)
    env = PushEnv(cfg)
    env.reset()
    obs, _, _, _ = env.step(np.array([10.0, 10.0]))  # wildly out-of-range action
    assert np.all(np.abs(env.ee_pos) <= cfg.world_size + 1e-6)
