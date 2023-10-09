import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from ..utils import HistoryTracker, describe_act
import random

class BanditEnv(gym.Env):
    """
    Bandit environment base to allow agents to interact with the class n-armed bandit
    in different variations

    p_dist:
        A list of probabilities of the likelihood that a particular bandit will pay out
    r_dist:
        A list of either rewards (if number) or means and standard deviations (if list)
        of the payout that bandit has
    """

    default_iter = 20
    default_steps = 50

    def __init__(self, p_dist, r_dist, max_steps=50):
        if len(p_dist) != len(r_dist):
            raise ValueError("Probability and Reward distribution must be the same length")

        if min(p_dist) < 0 or max(p_dist) > 1:
            raise ValueError("All probabilities must be between 0 and 1")

        for reward in r_dist:
            if isinstance(reward, list) and reward[1] <= 0:
                raise ValueError("Standard deviation in rewards must all be greater than 0")

        self.n_bandits = len(p_dist)
        self.action_space = spaces.Discrete(self.n_bandits)
        self.action_list = ["Pull slot machine {}.".format(i+1) for i in range(self.n_bandits)]
        self.observation_space = spaces.Discrete(1)
        self.p_dist = p_dist
        self.r_dist = r_dist
        self.desc = """
You are in the casino with 2 slot machines in front of you. Your goal is to try to earn the most from those slot machines.

{}
""".format(describe_act(self.action_list)).strip()
        self.history = HistoryTracker(max_steps)

        self._seed()
        self.score_tracker = 0

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)

        reward = -1
        done = False

        if np.random.uniform() < self.p_dist[action]:
            if not isinstance(self.r_dist[action], list):
                reward = self.r_dist[action]
            else:
                reward = np.random.normal(self.r_dist[action][0], self.r_dist[action][1])

        self.score_tracker+=int(action == self.optimal)
        info = {"obs": "You pulled slot machine {}, you received reward {}.".format(action+1, reward),
                "score": self.score_tracker,
                "manual": self.desc,
                "history": self.history.describe(),
                "optimal":self.optimal,
                "completed": 0,
                }
        self.history.step(info)

        return 0, reward, done, info

    def reset(self):
        idx_list = list(range(len(self.p_dist)))
        random.shuffle(idx_list)

        self.score_tracker=0
        self.p_dist = [self.p_dist[i] for i in idx_list]
        self.r_dist = [self.r_dist[i] for i in idx_list]
        self.ev = [p*r - (1-p) for p, r in zip(self.p_dist, self.r_dist)]
        self.optimal = np.argmax(self.ev)
        self.history.reset()
        info = {"obs": "A new round begins.",
                "score": self.score_tracker,
                "manual": self.desc,
                "history": self.history.describe(),
                "optimal":self.optimal,
                "completed": 0,
                }
        self.history.step(info)
        return 0, info

    def render(self, mode='human', close=False):
        pass


class BanditTwoArmedDeterministicFixed(BanditEnv):
    """Simplest case where one bandit always pays, and the other always doesn't"""
    def __init__(self):
        BanditEnv.__init__(self, p_dist=[1, 0], r_dist=[1, 1])


class BanditTwoArmedHighLowFixed(BanditEnv):
    """Stochastic version with a large difference between which bandit pays out of two choices"""
    def __init__(self):
        BanditEnv.__init__(self, p_dist=[0.8, 0.2], r_dist=[1, 1])


class BanditTwoArmedHighHighFixed(BanditEnv):
    """Stochastic version with a small difference between which bandit pays where both are good"""
    def __init__(self):
        BanditEnv.__init__(self, p_dist=[0.8, 0.9], r_dist=[1, 1])


class BanditTwoArmedLowLowFixed(BanditEnv):
    """Stochastic version with a small difference between which bandit pays where both are bad"""
    def __init__(self):
        BanditEnv.__init__(self, p_dist=[0.1, 0.2], r_dist=[1, 1])
