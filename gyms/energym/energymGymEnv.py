import gym
import numpy as np
import energym
from typing import Tuple
from copy import deepcopy
from re import match

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
                 discretize=False,
                 discrete_bins=30
                 ):
        '''
        Takes energym.env and updates class to follow gym.env framework.
        Args:
            env (energym.env, required): energym environment instance
        '''
        super().__init__()

        if normalize and discretize:
            raise ValueError(
                'Energym cannot normalise and discretize the state/action spaces. Please choose to normalize'
                'OR discretize, not both.'
            )



        values = ['123', '234', 'foobar']
        filtered_values = list(filter(lambda v: match('^\d+$', v), values))

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
        for key, value in act_space.items():
            if isinstance(value, energym.spaces.Box):
                action_space[key] = gym.spaces.Box(low=value.low[0], high=value.high[0],
                                                   shape=value.shape, dtype=np.float32)
                self.cont_actions.append(key)
                act_low[key] = value.low[0]
                act_high[key] = value.high[0]

            elif isinstance(value, energym.spaces.Discrete):
                action_space[key] = gym.spaces.Discrete(value.n)
                act_low[key] = 0
                act_low[key] = value.n
                self.discrete_actions.append(key)

        self.act_low = act_low
        self.act_high = act_high

        # normalise action space if prompted by user
        if normalize:
            for key in self.cont_actions:
                action_space[key] = gym.spaces.Box(low=-1,
                                                   high=1,
                                                   shape=1,
                                                   dtype=np.float32)

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

        self.obs_low = obs_low
        self.obs_high = obs_high

        # normalise obs space if prompted by user
        if normalize:
            for key in self.cont_obs:
                action_space[key] = gym.spaces.Box(low=-1,
                                                   high=1,
                                                   shape=1,
                                                   dtype=np.float32)

        # discretize obs spaces if prompted by user
        if discretize:
            # Obtain values of discretized obs space
            val_bins_obs = {}
            for key in self.obs_keys:
                val_bins_obs[key] = np.linspace(self.obs_low[key], self.obs_high[key], self.n_bins + 1)
            self.val_bins_obs = val_bins_obs

            # Convert Box spaces to Discrete spaces
            for key in self.cont_obs:
                action_space[key] = gym.spaces.Discrete(self.n_bins + 1)

        # configure Gym attributes
        self.action_space = action_space
        self.observation_space = observation_space
        self.reward_range = (-float("inf"), float("inf"))

    def step(self, action: np.array) -> Tuple[np.array, float, bool, dict]:
        """
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
        Resets the energym simulation
        Args:
            None
        Returns:
            None
        """
        self.env.reset()

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

        if self.normalize:
            for key in self.cont_obs:
                observation[key] = 2 * (observation[key] - self.obs_low[key]) / (self.obs_high[key]
                                                                                 - self.obs_low[key]) - 1
        elif self.discretize:
            for key in self.cont_obs:
                observation[key] = np.digitize(observation[key], self.val_bins_obs[key])

        # convert to ndarray
        observation = np.array(list(observation.values()), dtype=np.float).reshape(len(observation.values()), )

        return observation

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
                action_dict[key] = [self.val_bins_act[action_dict[key]]]  # index bins vals given action selected

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
                                                        (high_discomfort_temp - temps) ** 2)

        # energy term in reward
        energy = -(observation[self.power] * (self.step_period / 60)) / 1000 # kWh

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
