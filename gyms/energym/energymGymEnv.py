import gym
import numpy as np
import energym
from typing import Tuple
from copy import deepcopy
from re import match

class EnergymGymEnv(gym.Env):
    '''
    Energym environment that follows gym interface.

    Attributes
    ----------
    env: energym.env
        Energym env instance to be converted
    action_space: gym.spaces.Dict
        Environment action space in gym-compatible format
    observation_space: gym.spaces.Dict
        Environment observation space in gym-compatible format
    reward_range: tuple
        Accepted reward range for environment
    max_episode_length: int
        Maximum number of timesteps in one episode
    step_period: int
        Number of real-world minutes between timesteps in building simulation
    normalize: bool
        User-provided flag to require state and action spaces to be normalized
    discretize: bool
        User-provided flag to require state and action spaces to be discretized
    n_bins: int
        Number of bins to use for state/action discretization
    start_time: int
        Start time of simulation in seconds
    act_keys: list
        List of action strings
    obs_keys: list
        List of observation strings
    n_act: int
        Number of actions in action space
    temps: list
        List of temperature-related feature strings
    power: list
        List of power-related feature strings
    cont_actions: list
        List of continuous action strings
    discrete_actions: list
        List of discrete action strings
    cont_obs: list
        List of continuous observation strings
    discrete_obs: list
        List of discrete observation strings
    act_low: dict
        Dictionary of lower bounds of actions
    act_high: dict
        Dictionary of upper bounds of actions
    obs_low: dict
        Dictionary of lower bounds of observations
    obs_high: dict
        Dictionary of upper bounds of observations
    val_bins_act: dict
        Dictionary of ndarrays for action bin values if environment is discretized
    val_bins_obs: dict
        Dictionary of ndarrays for observation bin values if environment is discretized

    Methods
    -------
    step()
        Advances the simulation by one timestep
    render()
        Renders animation of environment. (not implemented)
    reset()
        Resets the simulation environment.
    seed()
        Sets seed for environment random number generator. (not implemented)
    close()
        Closes the simulation environment.
    action_converter()
        Converts ndarray action to Dict compatable with energym
    obs_converter()
        Converts energym output Dict to ndarray compatible with RL agents
    compute_reward()
        Calculates reward given environment observation
    compute_done()
        Raises True if at end of episode
    '''

    def __init__(self,
                 env,
                 max_episode_length=35040,
                 step_period=15,
                 normalize=True,
                 discretize=False,
                 discrete_bins=30
                 ):

        super().__init__()

        if normalize and discretize:
            raise ValueError(
                'Energym cannot normalise and discretize the state/action spaces. Please choose to normalize '
                ' OR discretize, not both.'
            )

        self.env = env
        self.max_episode_length = max_episode_length
        self.step_period = step_period
        self.normalize = normalize
        self.discretize = discretize
        if self.discretize:
            self.n_bins = discrete_bins
        self.start_time = self.step_period * 60  # convert minutes to seconds
        self.act_keys = [key for key in self.env.get_inputs_names()]
        self.obs_keys = [key for key in self.env.get_outputs_names()]
        self.n_act = len(self.act_keys)
        self.temps = list(filter(lambda t: match('Z\d\d_T', t), self.obs_keys))
        self.power = ['Fa_Pw_All']

        self.cont_actions = []
        self.discrete_actions = []
        self.cont_obs = []
        self.discrete_obs = []
        act_space = env.input_space
        obs_space = env.output_space

        ### UPDATE ACTION SPACE ###

        # create gym action space from energym action space and get lower and upper bounds
        action_space = gym.spaces.Dict()
        act_low = {}
        act_high = {}
        for key in self.act_keys:
            action = act_space[key]
            if isinstance(action, energym.spaces.box.Box):
                action_space[key] = gym.spaces.Box(low=action.low[0], high=action.high[0],
                                                   shape=action.shape, dtype=np.float32)
                self.cont_actions.append(key)
                act_low[key] = action.low[0]
                act_high[key] = action.high[0]

            elif isinstance(action, energym.spaces.discrete.Discrete):
                action_space[key] = gym.spaces.Discrete(action.n)
                act_low[key] = 0
                act_low[key] = action.n
                self.discrete_actions.append(key)

        self.act_low = act_low
        self.act_high = act_high

        # normalise action space if prompted by user
        if normalize:
            for key in self.cont_actions:
                action_space[key] = gym.spaces.Box(low=-1, high=1,
                                                   shape=action_space[key].shape, dtype=np.float32)

        # discretize action spaces if prompted by user
        if discretize:
            # Obtain values of discretized action space
            val_bins_act = {}
            for key in self.act_keys:
                val_bins_act[key] = np.linspace(self.act_low[key], self.act_high[key], self.n_bins + 1)
            self.val_bins_act = val_bins_act

            # Convert Box spaces to Discrete spaces
            for key in self.cont_actions:
                action_space[key] = gym.spaces.Discrete(self.n_bins + 1)

        ### UPDATE OBSERVATION SPACE ###

        # create gym obs space from energym obs space and get lower and upper bounds
        observation_space = gym.spaces.Dict()
        obs_low = {}
        obs_high = {}
        for key in self.obs_keys:
            obs = obs_space[key]
            if isinstance(obs, energym.spaces.box.Box):
                observation_space[key] = gym.spaces.Box(low=obs.low[0], high=obs.high[0],
                                                        shape=obs.shape, dtype=np.float32)
                self.cont_obs.append(key)
                obs_low[key] = obs.low[0]
                obs_high[key] = obs.high[0]

            elif isinstance(obs, energym.spaces.discrete.Discrete):
                observation_space[key] = gym.spaces
                self.discrete_obs.append(key)

        self.obs_low = obs_low
        self.obs_high = obs_high

        # normalise obs space if prompted by user
        if normalize:
            for key in self.cont_obs:
                observation_space[key] = gym.spaces.Box(low=-1, high=1,
                                                        shape=observation_space[key].shape, dtype=np.float32)

        # discretize obs spaces if prompted by user
        if discretize:
            # Obtain values of discretized obs space
            val_bins_obs = {}
            for key in self.obs_keys:
                val_bins_obs[key] = np.linspace(self.obs_low[key], self.obs_high[key], self.n_bins + 1)
            self.val_bins_obs = val_bins_obs

            # Convert Box spaces to Discrete spaces
            for key in self.cont_obs:
                observation_space[key] = gym.spaces.Discrete(self.n_bins + 1)

        # configure Gym attributes
        self.action_space = action_space
        self.observation_space = observation_space
        self.reward_range = (-float("inf"), 0)

    def step(self, action: np.array) -> Tuple[np.array, float, bool, dict]:
        """
        Takes action in Gym format, converts to Energym format and advances
        the simulation one time step, then reports results in Gym format.

        Args
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
        """
        # convert rllib action vector to dictionary compatible with energym
        action_dict = self.action_converter(action)

        # take step in energym environment
        observations = self.env.step(action_dict)

        # determine whether episode is finshed
        done = self.compute_done(observations)

        # evaluate reward
        reward = self.compute_reward(observations)

        # convert energym output observation to obs vector compatible with rllib
        observations = self.obs_converter(observations)

        # create dummy info variable, TBC what output we pass to user
        info = {}

        return observations, reward, done, info

    def render(self):
        pass

    def reset(self) -> None:
        """
        Resets the energym simulation and return first observation
        Args:
            None
        Returns:
            obs (dict):
                first observation from reset environment
        """
        self.env.reset()
        obs = self.env.get_output()
        obs = self.obs_converter(obs)

        return obs

    def seed(self):
        pass

    def obs_converter(self, observation: dict) -> np.array:
        """
        Takes energym observation and normalises in [-1,1]

        Args:
            observation (dict): unnormalised dictionary output of energym

        Returns:
            observation (np.array): array normalised outputs of shape (obs_dim,)
        """
        observation = deepcopy(observation)
        obs = {}
        if self.normalize:
            for key in self.cont_obs:
                obs[key] = np.array(2 * (observation[key] - self.obs_low[key]) /
                                    (self.obs_high[key] - self.obs_low[key]) - 1, dtype=np.float32).reshape(1,)

        elif self.discretize:
            for key in self.cont_obs:
                obs[key] = np.digitize(observation[key], self.val_bins_obs[key], dtype=np.float32)

        # convert to ndarray
        # observation = np.array(list(observation.values()), dtype=np.float).reshape(len(observation.values()), )

        return obs

    def action_converter(self, action: np.array) -> dict:
        """
        Takes numpy array actions and converts to dictionary compatible with energym. This transformation is compatible
        with both normalized and discrete action spaces.

        Args:
            action (np.array): Array of actions computed by the agent to be implemented in this step

        Returns:
            action (dict): Dict of actions in format recognisable to energym

        Notes:
            energym expects values in action dict to be lists, hence the square brackets around each value in
            action_dict.
        """

        action = deepcopy(action)
        action_dict = {}

        for i, key in enumerate(self.act_keys):
            action_dict[key] = action[i]

        if self.normalize:
            # un-normalise values
            for key in self.cont_actions:
                action_dict[key] = [((action_dict[key] + 1) / 2) * (self.act_high[key] - self.act_low[key]) \
                                    + self.act_low[key]]

        elif self.discretize:
            # un-discretize values
            for key in self.cont_actions:
                action_dict[key] = [self.val_bins_act[key][action_dict[key]]]  # index bins vals given action selected

        else:
            # recreate action_dict with action as lists if not normalising or discretising as per function notes.
            # to-do: find more elegant way of performing this transformation
            for i, key in enumerate(self.act_keys):
                action_dict[key] = [action[i]]

        return action_dict

    def compute_reward(self, observation: dict) -> np.float:
        """
        Compute reward given observation at current timestep.

        Args:
            observation (dict): Dictionary of observation from energym simulation

        Returns:
            reward (np.float): Scalar reward from environment

        Notes
        -----
        To be updated to allow for user specified reward function.
        Currently, the reward motivates the agent to minimise energy-use
        whilst maintaining building temperature in range [19, 24].
        """

        low_discomfort_temp = 19
        high_discomfort_temp = 24
        discomfort_penalty = 1

        # discomfort term in reward
        discomfort = 0
        for t in self.temps:
            temp = observation[t]

            if (low_discomfort_temp <= temp) and (temp <= high_discomfort_temp):
                pass
            else:
                discomfort -= discomfort_penalty * min((low_discomfort_temp - temp) ** 2,
                                                        (high_discomfort_temp - temp) ** 2)

        # energy term in reward
        energy = -(observation[self.power[0]] * (self.step_period / 60)) / 1000 # kWh

        reward = energy + discomfort

        return np.float(reward)

    def compute_done(self, observation: dict):
        """
        Compute whether the episode is finished or not. By default, a
        maximum episode length is defined and the episode will be finished
        only when the time exceeds this maximum episode length.

        Args:
            observation (dict): Dictionary of observation from energym simulation

        Returns:
            done (boolean): Boolean indicating whether the episode is done or not.

        Notes
        -----
        The 'time' variable reported in energym observations is in seconds, but
        environment step period are in minutes for comprehensability. Hence, we
        convert the max_episode_length (timesteps) to seconds.
        """

        done = observation['time'] >= self.start_time + \
               (self.max_episode_length * self.step_period * 60)

        return done
