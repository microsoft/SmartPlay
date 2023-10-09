import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from ..utils import HistoryTracker, describe_act
import random


class RPSEnv(gym.Env):
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

    def __init__(self, probs, reward = [1, 1, 1], max_steps=50):

        if min(probs) < 0 or max(probs) > 1 or sum(probs) != 1:
            raise ValueError("All probabilities must be between 0 and 1, and sum to 1")

        dist = list(zip(probs, reward))
        random.shuffle(dist)
        self.probs = [d[0] for d in dist]

        self.action_list = ["Rock", "Paper", "Scissor"]
        self.reward = [d[1] for d in dist]
        self._update_manual()
        self.observation_space = spaces.Discrete(1)
        self.action_space = spaces.Discrete(len(self.probs))
        self.history = HistoryTracker(max_steps)
        self.score_tracker = 0

        self._seed()

    def _update_manual(self):
        self.desc = """
For the game Rock Paper Scissors, you and the opponent choose one of three options: rock, paper, or scissors.
After both players have chosen, the winner is determined as follows:
Rock crushes scissors (Rock wins, score {})
Scissors cut paper (Scissors win, score {})
Paper covers rock (Paper wins, score {})
If you lose, your score is the negative of the winner's score.
If both players choose the same option, it's a draw (score 0).
Your goal is to maximize your score.

{}
""".strip().format(*self.reward, describe_act(self.action_list))

    def compute_optimal_action(self):
        n_actions = len(self.probs)
        expected_scores = np.zeros(n_actions)

        for action in range(n_actions):
            expected_score = 0
            for opponent_action in range(n_actions):
                expected_score += self.probs[opponent_action] * self.reward[opponent_action]
                if (action == 0 and opponent_action == 1) or (action == 1 and opponent_action == 2) or (action == 2 and opponent_action == 0):
                    expected_score -= self.probs[opponent_action] * self.reward[opponent_action]
                elif action == opponent_action:
                    pass
                else:
                    expected_score += self.probs[opponent_action] * self.reward[action]
            expected_scores[action] = expected_score

        optimal_action = np.argmax(expected_scores)
        expected_score = expected_scores[optimal_action]

        return optimal_action, expected_score

    def sample_opponent_action(self, probs):
        return np.random.choice(len(probs), p=probs)

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)
        opponent_action = self.sample_opponent_action(self.probs)
        
        if action == 0 and opponent_action == 1:
            result = "lost"
        elif action == 1 and opponent_action == 2:
            result = "lost"
        elif action == 2 and opponent_action == 0:
            result = "lost"
        elif action == opponent_action:
            result = "tied"
        else:
            result = "won"

        reward = self.reward[action] if result == "won" else -self.reward[opponent_action] if result == "lost" else 0
        done = False
        optimal_action, expected_score = self.compute_optimal_action()
        self.score_tracker += int(action == optimal_action)
        info = {
            "obs": "You chose {}, and the opponent chose {}. You {} and received score {}.\nNew round begins.".format(
                self.action_list[action], 
                self.action_list[opponent_action], 
                result, reward),
            "manual": self.desc,
            "history": self.history.describe(),
            "score": self.score_tracker,
            "result": result,
            "completed": 0,
            }
        self.history.step(info)

        return 0, reward, done, info

    def reset(self):
        probs, reward = self.probs, self.reward
        dist = list(zip(probs, reward))
        random.shuffle(dist)
        self.probs = [d[0] for d in dist]

        self.reward = [d[1] for d in dist]
        self._update_manual()
        self.history.reset()
        self.score_tracker = 0
        info = {
            "obs":"New round begins.",
            "manual": self.desc,
            "history": self.history.describe(),
            "score": self.score_tracker,
            "completed": 0,
            }
        self.history.step(info)
        return 0, info

    def render(self, mode='human', close=False):
        pass


class RockPaperScissorBasic(RPSEnv):
    """Basic version of the game with biased opponent"""
    def __init__(self):
        RPSEnv.__init__(self, [0.2, 0.2, 0.6])


class RockPaperScissorDifferentScore(RPSEnv):
    """Biased opponent and different scores"""
    def __init__(self):
        RPSEnv.__init__(self, p_dist=[0.3, 0.3, 0.4], r_dist=[1, 2, 1])
