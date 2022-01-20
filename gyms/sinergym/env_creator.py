"""Module to create sinergym environments. """

import gym
import sinergym
import sinergym.utils.wrappers

from sinergym.utils.common import RANGES_5ZONE, RANGES_IW, RANGES_DATACENTER


def create_env(env_config: dict = None) -> gym.Env:
    """Create sinergym environment.

    Args:
        env_config (dict, optional): configuration kwargs for sinergym. Currently,
            there is only a single key in this dictionary, "name". This sets
            the name of the environment. Defaults to None.

    Returns:
        gym.Env: a configured gym environment.
    """

    if env_config is None:
        env_config = {"name":"Eplus-5Zone-hot-continuous-v1"}


    # importing sinergym automatically nicely registers
    # sinergym envs with OpenAI gym
    env = gym.make(env_config["name"])

    # Taken from
    # https://github.com/jajimer/sinergym/blob/24a37965f4e749faf6caaa3d4ece95330a478904/DRL_battery.py#L221
    if env_config["normalize"]:
        # We have to know what dictionary ranges to use
        norm_range = None
        env_type = args.environment.split('-')[2]
        if env_type == 'datacenter':
            range = RANGES_5ZONE
        elif env_type == '5Zone':
            range = RANGES_IW
        elif env_type == 'IWMullion':
            range = RANGES_DATACENTER
        else:
            raise NameError('env_type is not valid, check environment name')
        env = sinergym.utils.wrappers.NormalizeObservation(env, ranges=range)

    if env_config["multi_observation"]:
        env = MultiObsWrapper(env)

    return env