"""This is the env_creator module for integrating Energym with Beobench."""

import gym
import energym
from energymGymEnv import EnergymGymEnv


def create_env(env_config: dict = None) -> gym.Env:
    """Create Energym environment.

    Args:
        env_config (dict, optional): configuration kwargs for Energym. Defaults to None.

    Returns:
        gym.Env: a configured gym environment.
    """

    # each energym environment sampels their building model at differing intervals
    env_timesteps = {
        "OfficesThermostat-v0": 15,
        "SeminarcenterThermostat-v0": 10,
        "Apartments2Thermal-v0": 3,
        "MixedUseFanFCU-v0": 15,
        # add remaining envs
    }

    if not env_config:
        env_config = {
            "energym_environment": "MixedUseFanFCU-v0",
            "weather": "GRC_A_Athens",
            "days": 365,
            "gym_kwargs": {
                "max_episode_length": 35040,
                "step_period": 15,
                "normalize": True,
                "discretize": 30,
            },
        }

    # build energym environment
    env = energym.make(
        env_config["energym_environment"],
        weather=env_config["weather"],
        simulation_days=env_config["days"],
    )

    env = EnergymGymEnv(env, **env_config["gym_kwargs"])

    return env
