import gym
import numpy as np
import energym
from typing import Tuple
from copy import deepcopy

class EnergymGymEnv(gym.env):
    '''
    Energym environment that follows gym interface, allowing RL agents
    to interact with building models readily.

    Attributes
    ----------
    energym_path : str
        Absolute path to the energym folder.
    runs_path : str
        Path to store the information of the simulation run.
    Methods
    -------
    step(action)
        Advances the simulation by one timestep
    render()
        Renders animation of environment. (not implemented)
    reset()
        Resets the simulation environment. (not implemented)
    seed()
        Sets seed for environment random number generator. (not implemented)
    close()
        Closes the simulation environment. (not implemented)


    '''

    def __init__(self,
                 env,
                 max_episode_length=35040,
                 step_period=15,
                 normalize=True,
                 discretize=30
                 ):
        '''
        Takes energym.env and updates class to follow gym.env framework.
        Args:
            env (energym.env, required): energym environment instance
        '''
        super().__init__()

        self.env = env
        self.max_episode_length = max_episode_length
        self.step_period = step_period
        self.normalize = normalize
        if discretize:
            self.n_bins_act = discretize
        self.start_time = self.step_period * 60 # convert minutes to seconds
        self.act_keys = [key for key in self.env.get_inputs_names()]
        self.obs_keys = [key for key in self.env.get_outputs_names()]
        self.cont_actions = []
        self.discrete_actions = []
        self.cont_obs = []
        self.discrete_obs = []
        act_space = env.input_space
        obs_space = env.output_space

        # extract gym action space from energym action space
        action_space = {}
        act_low = {}
        act_high = {}
        for key, value in act_space.items():
            if isinstance(value, energym.spaces.Box):
                action_space[key] = gym.spaces.Box(low=value.low[0], high=value.high[0],
                                                   shape=value.shape, dtype=np.float32)
                self.cont_actions.append(key)
                act_low[key] = value.low[0]
                act_high[key] = value.high[0]

            elif isinstance(value, energym.spaces.Discrete):
                action_space[key] = gym.spaces.Discrete(value.n)
                self.discrete_actions.append(key)

        if discretize:
            # Calculate dimension of action space
            self.n_act = len(self.act_keys)

            # Obtain values of discretized action space
            self.val_bins_act = [np.linspace(l, h, self.n_bins_act + 1)
                                 for l, h in zip(low.flatten(), high.flatten())]

            # Instantiate discretized action space
            self.action_space = spaces.Discrete((n_bins_act + 1) ** self.n_act)

        # extract gym observation space from energym obs space
        observation_space = {}
        obs_low = {}
        obs_high = {}
        for key, value in obs_space.items():
            if isinstance(value, energym.spaces.Box):
                observation_space[key] = gym.spaces.Box(low=value.low[0], high=value.high[0],
                                                        shape=value.shape, dtype=np.float32)
                self.cont_obs.append(key)
                obs_low[key] = value.low[0]
                obs_high[key] = value.high[0]

            elif isinstance(value, energym.spaces.Discrete):
                observation_space[key] = gym.spaces
                self.discrete_obs.append(key)

        # configure Gym attributes
        self.action_space = action_space
        self.observation_space = observation_space
        self.reward_range = (-float("inf"), float("inf"))

        # configure other attributes
        self.obs_low = obs_low
        self.obs_high = obs_high
        self.act_low = act_low
        self.act_high = act_high


        ### OBSERVATION SPACE BOUNDS ###
        upper_bound = {}
        lower_bound = {}

        default_upper = {key: self.env.output_space[key].high[0] for key in self.obs_space}
        default_lower = {key: self.env.output_space[key].low[0] for key in self.obs_space}

        self.obs_low = {**lower_bound, **default_lower}
        self.obs_high = {**upper_bound, **default_upper}

        ### ACTION SPACE

    def step(self, action: np.array) -> Tuple[np.array, float, bool, dict]:
        '''
        Takes action in Gym format, converts to Energym format and advances
        the simulation one time step, then reports results in Gym format.

        Parameters
        ----------
        action: numpy array
            List of actions computed by the agent to be implemented
            in this step

        Returns
        -------
        observations: numpy array
            Observations at the end of this time step
        reward: float
            Reward for the state-action pair implemented
        done: boolean
            True if episode is finished after this step
        info: dictionary
            Additional information for this step



        '''

        action_dict = self.action_convertor(action, norm=self.normalize)

        observations = self.env.step(action_dict)

        done = self.compute_done(observations)

        reward = self.compute_reward(observations)

        observations = self.obs_convertor(observations)

        info = {}


        return observations, reward, done, info

    def render(self):
        pass

    def reset(self):
        pass

    def seed(self):
        pass

    def obs_normaliser(self, observation: dict, norm: bool) -> np.array:
        '''
        Takes energym observation and normalises in [-1,1]
        Args:
            observation (dict): unnormalised dictionary output of energym

        Returns:
            observation (np.array): array normalised outputs of shape (obs_dim,)
        '''
        observation = deepcopy(observation)

        # normalise values
        if norm:
            for key in self.obs_keys:
                observation[key] = 2 * (observation[key] - self.obs_low[key]) / (self.obs_high[key] - self.obs_low[key]) - 1

        # convert to ndarray
        observation = np.array(list(observation.values()), dtype=np.float).reshape(len(observation.values()), )

        return observation

    def action_convertor(self, action: np.array, norm: bool) -> dict:
        '''
        Takes numpy array actions and converts to dictionary compatible with energym
        Args:
            action (np.array): Array of actions computed by the agent to be implemented in this step

        Returns:
            action (dict): Dict of actions in format recognisable to energym
        '''

        action = deepcopy(action)
        action_dict = {}

        for i, key, in enumerate(self.act_keys):
            action_dict[key] = action[i]

        if norm:
            # un-normalise values
            for key in self.cont_actions:
                action_dict[key] = [((action_dict[key] + 1) / 2) * (self.act_high[key] - self.act_low[key]) \
                         + self.act_low[key]]

            # manually interpret discrete elements of action array
            if len(self.discrete_actions) > 0:
                for key in self.discrete_actions:
                    if action_dict[key] <= 0:
                        action_dict[key] = [0]
                    else:
                        action_dict[key] = [1]

        return action_dict

    def compute_reward(self, observations):

        return reward

    def compute_done(self, observation):
        '''
        Compute whether the episode is finished or not. By default, a
        maximum episode length is defined and the episode will be finished
        only when the time exceeds this maximum episode length.

        Returns
        -------
        done: boolean
            Boolean indicating whether the episode is done or not.

        Notes
        -----
        The 'time' variable reported in energym observations is in seconds, but
        environment step period are in minutes for comprehensability. Hence, we
        convert the max_episode_length (timesteps) to seconds.
        '''

        done = observation['time'] >= self.start_time + \
                                    (self.max_episode_length * self.step_period * 60)

        return done


class NormalizedActionWrapper(gym.ActionWrapper):
    '''
    This wrapper normalizes the values of the action space to lie
    between -1 and 1. Normalization can significantly help with convergence speed.
    '''

    def __init__(self, env):

        # construct from parent class
        super().__init__(env)

class NormalizedObservationWrapper(gym.ObservationWrapper):
    '''
    This wrapper normalizes the values of the observation space to lie
    between -1 and 1. Normalization can significantly help with convergence speed.
    '''

    def __init__(self, env):

        # construct from parent class
        super().__init__(env)

class DiscretizedActionWrapper(gym.ActionWrapper):
    '''
    This wrapper normalizes the values of the observation space to lie
    between -1 and 1. Normalization can significantly help with convergence speed.
    '''

    def __init__(self, env, n_bins_act):

        # construct from parent class
        super().__init__(env)