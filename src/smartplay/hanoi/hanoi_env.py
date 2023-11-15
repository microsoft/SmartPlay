import gym
from gym import error, spaces, utils
from gym.utils import seeding
from ..utils import HistoryTracker, describe_act

import random
import itertools
import numpy as np

action_to_move = [(0, 1), (0, 2), (1, 0),
                  (1, 2), (2, 0), (2, 1)]

class HanoiEnv(gym.Env):
    default_iter = 10
    default_steps = 30

    def __init__(self, max_steps=5, num_disks=4, env_noise=0):
        self.num_disks = num_disks
        self.env_noise = env_noise
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Tuple(self.num_disks*(spaces.Discrete(3),))

        self.current_state = None
        self.goal_state = self.num_disks*(2,)
        self.history = HistoryTracker(max_steps)

        self.done = None
        self.ACTION_LOOKUP = {0 : "(0,1) - top disk of pole 0 to top of pole 1 ",
                              1 : "(0,2) - top disk of pole 0 to top of pole 2 ",
                              2 : "(1,0) - top disk of pole 1 to top of pole 0",
                              3 : "(1,2) - top disk of pole 1 to top of pole 2",
                              4 : "(2,0) - top disk of pole 2 to top of pole 0",
                              5 : "(2,1) - top disk of pole 2 to top of pole 1"}


        self.action_list = [
        "Move the top disk of rod A to the top of rod B",
        "Move the top disk of rod A to the top of rod C",
        "Move the top disk of rod B to the top of rod A",
        "Move the top disk of rod B to the top of rod C",
        "Move the top disk of rod C to the top of rod A",
        "Move the top disk of rod C to the top of rod B",
        ]
      
        self.desc = """
The game consists of three rods (A,B,C) and a number of disks of various sizes, which can go onto any rod. 
The game begins with the disks stacked on rod A in order of decreasing size, the smallest at the top (righthand side). 
The objective is to move the entire stack to rod C, obeying the following rules:

 - Only one disk may be moved at a time.
 - Each move consists of taking the top disk from one of the stacks and placing it on top of another stack or on an empty rod.
 - You cannot place a bigger disk on top of a smaller disk.

For example, considering movements from B under the following setting:
- A: |bottom, [0], top|
- B: |bottom, [1], top|
- C: |bottom, [2], top|
You are only allowed to move from B to C but not A, since the top of B (1) is smaller than the top of C (2) but bigger than the top of A (0).

Finally, the starting configuration is:
- A: |bottom, {}, top|
- B: |bottom, [], top|
- C: |bottom, [], top|

and the goal configuration is:
- A: |bottom, [], top|
- B: |bottom, [], top|
- C: |bottom, {}, top|
with top on the right and bottom on the left

{}
""".format(list(range(num_disks))[::-1],list(range(num_disks))[::-1], describe_act(self.action_list)).strip()

    def step(self, action):
        """
        * Inputs:
            - action: integer from 0 to 5 (see ACTION_LOOKUP)
        * Outputs:
            - current_state: state after transition
            - reward: reward from transition
            - done: episode state
            - info: dict of booleans (noisy?/invalid action?)
        0. Check if transition is noisy or not
        1. Transform action (0 to 5 integer) to tuple move - see Lookup
        2. Check if move is allowed
        3. If it is change corresponding entry | If not return same state
        4. Check if episode completed and return
        """
        if self.done:
            raise RuntimeError("Episode has finished. Call env.reset() to start a new episode.")

        info = {"transition_failure": False,
                "invalid_action": False}

        if self.env_noise > 0:
            r_num = random.random()
            if r_num <= self.env_noise:
                action = random.randint(0, self.action_space.n-1)
                info["transition_failure"] = True

        move = action_to_move[action]

        if self.move_allowed(move):
            disk_to_move = min(self.disks_on_peg(move[0]))
            moved_state = list(self.current_state)
            moved_state[disk_to_move] = move[1]
            self.current_state = tuple(moved_state)
        else:
            info["invalid_action"] = True

        if self.current_state == self.goal_state:
            reward = 100
            self.done = True
        elif info["invalid_action"] == True:
            reward = -1
        else:
            reward = 0

        info["state"] = (self.disks_on_peg(0), self.disks_on_peg(1), self.disks_on_peg(2))
        info["score"] = len(self.disks_on_peg(2))
        info["manual"] = self.desc
        info["obs"] = self.describe_state(info, action)
        info["history"] = self.history.describe()
        info["completed"] = 0 if not self.done else 1
        self.history.step(info)

        return self.current_state, reward, self.done, info

    def describe_state(self, state, action=None):
        index = ['A', 'B', 'C']
        if action is not None:
            result = "You tried to {}. Current configuration:".format(self.action_list[action].lower())
        else:
            result = "Current configuration:"
        for i in range(3):
            result += "\n- {}: |bottom, {}, top|".format(index[i], state['state'][i][::-1])
        return result.strip()

    def disks_on_peg(self, peg):
        """
        * Inputs:
            - peg: pole to check how many/which disks are in it
        * Outputs:
            - list of disk numbers that are allocated on pole
        """
        return [disk for disk in range(self.num_disks) if self.current_state[disk] == peg]

    def move_allowed(self, move):
        """
        * Inputs:
            - move: tuple of state transition (see ACTION_LOOKUP)
        * Outputs:
            - boolean indicating whether action is allowed from state!
        move[0] - peg from which we want to move disc
        move[1] - peg we want to move disc to
        Allowed if:
            * discs_to is empty (no disc of peg) set to true
            * Smallest disc on target pole larger than smallest on prev
        """
        disks_from = self.disks_on_peg(move[0])
        disks_to = self.disks_on_peg(move[1])

        if disks_from:
            return (min(disks_to) > min(disks_from)) if disks_to else True
        else:
            return False

    def reset(self):
        self.current_state = self.num_disks * (0,)
        self.done = False
        self.history.reset()
        info = {"state":(self.disks_on_peg(0), self.disks_on_peg(1), self.disks_on_peg(2))}
        info["score"] = len(self.disks_on_peg(2))
        info["manual"] = self.desc
        info["obs"] = self.describe_state(info)
        info["history"] = self.history.describe()
        info["completed"] = 0
        self.history.step(info)
        return self.current_state, info

    def render(self, mode='human', close=False):
        return

    def set_env_parameters(self, num_disks=4, env_noise=0, verbose=True):
        self.num_disks = num_disks
        self.env_noise = env_noise
        self.observation_space = spaces.Tuple(self.num_disks*(spaces.Discrete(3),))
        self.goal_state = self.num_disks*(2,)

        if verbose:
            print("Hanoi Environment Parameters have been set to:")
            print("\t Number of Disks: {}".format(self.num_disks))
            print("\t Transition Failure Probability: {}".format(self.env_noise))

    def get_movability_map(self, fill=False):
        # Initialize movability map
        mov_map = np.zeros(self.num_disks*(3, ) + (6,))

        if fill:
            # Get list of all states as tuples
            id_list = self.num_disks*[0] + self.num_disks*[1] + self.num_disks*[2]
            states = list(itertools.permutations(id_list, self.num_disks))

            for state in states:
                for action in range(6):
                    move = action_to_move[action]
                    disks_from = []
                    disks_to = []
                    for d in range(self.num_disks):
                        if state[d] == move[0]: disks_from.append(d)
                        elif state[d] == move[1]: disks_to.append(d)

                    if disks_from: valid = (min(disks_to) > min(disks_from)) if disks_to else True
                    else: valid = False

                    if not valid: mov_map[state][action] = -np.inf

                    move_from = [m[0] for m in action_to_move]
                    move_to = [m[1] for m in action_to_move]

        return mov_map

class Hanoi3Disk(HanoiEnv):
    """Basic 3 disk Hanoi Environment"""
    def __init__(self):
        HanoiEnv.__init__(self, num_disks=3)

class Hanoi4Disk(HanoiEnv):
    """Basic 4 disk Hanoi Environment"""
    def __init__(self):
        HanoiEnv.__init__(self, num_disks=4)
