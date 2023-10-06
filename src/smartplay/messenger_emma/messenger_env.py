import gym
from gym import error, spaces, utils
from gym.utils import seeding
from ..utils import HistoryTracker, describe_act

import random
import itertools
import numpy as np
import messenger
import messenger.envs


''' Format function passed to numpy print to make things pretty.
'''

id_map = {}

for ent in messenger.envs.config.ALL_ENTITIES:
    id_map[ent.id] = ent.name
id_map[0] = '  '
id_map[15] = 'you (agent) without the message'
id_map[16] = 'you (agent) with the message'

def describe_block(i):
    if i < 17:
        return id_map[i]
    else:
        return 'XX'

def describe_loc(ref, P):
    desc = []
    if ref[0] > P[0]:
        desc.append("north")
    elif ref[0] < P[0]:
        desc.append("south")
    if ref[1] > P[1]:
        desc.append("west")
    elif ref[1] < P[1]:
        desc.append("east")

    return "-".join(desc)

    
def describe_frame(info):

    if 15 in np.unique(info['avatar']):
        obs = "You (agent) don't have the message."
        agent = 15
    elif 16 in np.unique(info['avatar']):
        obs = "You (agent) already have the message."
        agent = 16
    else:
        print(np.unique(info['avatar']))
        raise NotImplemented("Problem with agent")
    center = np.array(np.where(info['avatar'].squeeze() == agent)).squeeze()
    info = info['entities']
    result = ""
    x = np.arange(info.shape[1])
    y = np.arange(info.shape[0])
    x1, y1 = np.meshgrid(x,y)
    loc = np.stack((y1, x1),axis=-1)
    dist = np.absolute(center-loc).sum(axis=-1)[:,:,np.newaxis]

    obj_info_list = []
    
    
    for idx in np.unique(info):
        if idx == 15 or idx == 0:
            continue
        smallest = np.unravel_index(np.argmin(np.where(info==idx, dist, np.inf)), info.shape)
        obj_info_list.append((describe_block(idx), dist[smallest[:2]][0], describe_loc(np.array([0,0]), smallest[:2]-center)))

    if len(obj_info_list)>0:
        status_str = "You see:\n{}".format("\n".join(["- {} {} steps to your {}".format(name, dist, loc) if dist>0 else "- {} 0 steps with you".format(name) for name, dist, loc in obj_info_list]))
    else:
        status_str = "You see nothing away from you."
    result += obs.strip() + "\n\n" + status_str.strip()
    
    return result.strip()

class MessengerEnv(gym.Env):
    default_iter = 100
    default_steps = None

    def __init__(self, lvl=1, max_steps=2, env_noise=0):

        lvl_to_steps = [4, 64, 128]
        self.default_steps = lvl_to_steps[lvl-1]
        env_id = 'msgr-test-v{}'.format(lvl)
        self._env = gym.make(env_id)
        self.action_list = ["Move North", "Move South", "Move West", "Move East", "Do Nothing"]

        self.history = HistoryTracker(max_steps)
        self.game_context = """In the game, MESSENGER, each entity can take on one of three roles: an enemy, message, or goal. The agent’s objective is to bring the message to the goal while avoiding the enemies. If the agent encounters an enemy at any point in the game, or the goal without first obtaining the message, it loses the game and obtains a reward of −1.""".strip()
        self.advice = """To solve a game, you may find it helpful to list the objects that you see. Then for each object, match it with an entity description, and identify whether it is good or bad to interact with the object.
The name specifications of in-game objects may not be exact matches. Please try identifying with synonyms.
""".strip()

    def _update_manual(self, manual):
        self.desc = "{}\n\n{}\n\n{}\n\n{}".format(self.game_context, "\n".join(manual), self.advice, describe_act(self.action_list)).strip()

    def describe(self, obs, action=None):
        if action is not None:
            return "You took action {}.\n\n{}".format(self.action_list[action], describe_frame(obs))
        else:
            return describe_frame(obs)

    def reset(self):
        obs, manual = self._env.reset()
        self._update_manual(manual)
        self.history.reset()
        info = {
            "obs": self.describe(obs),
            "manual": self.desc,
            "history": self.history.describe(),
            "score": 0,
            "completed": 0,
            }
        self.history.step(info)
        return obs, info

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        try:
            description = self.describe(obs, action)
        except:
            description = "Environment Error."
        info={
            "obs": description,
            "manual": self.desc,
            "history": self.history.describe(),
            "score": reward,
            "completed": int(reward==1 and done),
            }
        self.history.step(info)
        return obs, reward, done, info
