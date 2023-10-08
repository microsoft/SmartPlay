import gym
from gym import error, spaces, utils
from gym.utils import seeding
from ..utils import HistoryTracker
from .crafter import Env
import numpy as np

id_to_item = [0]*19
import itertools
dummyenv = Env()
for name, ind in itertools.chain(dummyenv._world._mat_ids.items(), dummyenv._sem_view._obj_ids.items()):
    name = str(name)[str(name).find('objects.')+len('objects.'):-2].lower() if 'objects.' in str(name) else str(name)
    id_to_item[ind] = name
player_idx = id_to_item.index('player')
del dummyenv

vitals = ["health","food","drink","energy",]

rot = np.array([[0,-1],[1,0]])
directions = ['front', 'right', 'back', 'left']

def describe_inventory(info):
    result = ""
    
    status_str = "Your status:\n{}".format("\n".join(["- {}: {}/9".format(v, info['inventory'][v]) for v in vitals]))
    result += status_str + "\n\n"
    
    inventory_str = "\n".join(["- {}: {}".format(i, num) for i,num in info['inventory'].items() if i not in vitals and num!=0])
    inventory_str = "Your inventory:\n{}".format(inventory_str) if inventory_str else "You have nothing in your inventory."
    result += inventory_str #+ "\n\n"
    
    return result.strip()


REF = np.array([0, 1])

def rotation_matrix(v1, v2):
    dot = np.dot(v1,v2)
    cross = np.cross(v1,v2)
    rotation_matrix = np.array([[dot, -cross],[cross, dot]])
    return rotation_matrix

def describe_loc(ref, P):
    desc = []
    if ref[1] > P[1]:
        desc.append("north")
    elif ref[1] < P[1]:
        desc.append("south")
    if ref[0] > P[0]:
        desc.append("west")
    elif ref[0] < P[0]:
        desc.append("east")

    return "-".join(desc)


def describe_env(info):
    assert(info['semantic'][info['player_pos'][0],info['player_pos'][1]] == player_idx)
    semantic = info['semantic'][info['player_pos'][0]-info['view'][0]//2:info['player_pos'][0]+info['view'][0]//2+1, info['player_pos'][1]-info['view'][1]//2+1:info['player_pos'][1]+info['view'][1]//2]
    center = np.array([info['view'][0]//2,info['view'][1]//2-1])
    result = ""
    x = np.arange(semantic.shape[1])
    y = np.arange(semantic.shape[0])
    x1, y1 = np.meshgrid(x,y)
    loc = np.stack((y1, x1),axis=-1)
    dist = np.absolute(center-loc).sum(axis=-1)
    obj_info_list = []
    
    facing = info['player_facing']
    target = (center[0] + facing[0], center[1] + facing[1])
    target = id_to_item[semantic[target]]
    obs = "You face {} at your front.".format(target, describe_loc(np.array([0,0]),facing))
    
    for idx in np.unique(semantic):
        if idx==player_idx:
            continue

        smallest = np.unravel_index(np.argmin(np.where(semantic==idx, dist, np.inf)), semantic.shape)
        obj_info_list.append((id_to_item[idx], dist[smallest], describe_loc(np.array([0,0]), smallest-center)))

    if len(obj_info_list)>0:
        status_str = "You see:\n{}".format("\n".join(["- {} {} steps to your {}".format(name, dist, loc) for name, dist, loc in obj_info_list]))
    else:
        status_str = "You see nothing away from you."
    result += status_str + "\n\n"
    result += obs.strip()
    
    return result.strip()


def describe_act(info):
    result = ""
    
    action_str = info['action'].replace('do_', 'interact_')
    action_str = action_str.replace('move_up', 'move_north')
    action_str = action_str.replace('move_down', 'move_south')
    action_str = action_str.replace('move_left', 'move_west')
    action_str = action_str.replace('move_right', 'move_east')
    
    act = "You took action {}.".format(action_str) 
    result+= act
    
    return result.strip()


def describe_status(info):
    if info['sleeping']:
        return "You are sleeping, and will not be able take actions until energy is full.\n\n"
    elif info['dead']:
        return "You died.\n\n"
    else:
        return ""

    
def describe_frame(info, action):
    try:
        result = ""
        
        if action is not None:
            result+=describe_act(info)
        result+=describe_status(info)
        result+="\n\n"
        result+=describe_env(info)
        result+="\n\n"
        result+=describe_inventory(info)
        
        return result.strip()
    except:
        return "Error, you are out of the map."

class Crafter(Env):

    default_iter = 10
    default_steps = 10000

    def __init__(self, area=(64, 64), view=(9, 9), size=(64, 64), reward=True, length=10000, seed=None, max_steps=2):
        self.history = HistoryTracker(max_steps)
        self.action_list = ["Noop", "Move West", "Move East", "Move North", "Move South", "Do", \
    "Sleep", "Place Stone", "Place Table", "Place Furnace", "Place Plant", \
    "Make Wood Pickaxe", "Make Stone Pickaxe", "Make Iron Pickaxe", "Make Wood Sword", \
    "Make Stone Sword", "Make Iron Sword"]
        import pickle
        import pathlib
        root = pathlib.Path(__file__).parent
        with open(root / "assets/crafter_ctxt.pkl", 'rb') as f: # Context extracted using text-davinci-003 following https://arxiv.org/abs/2305.15486
            CTXT = pickle.load(f)
        CTXT = CTXT.replace("DO NOT answer in LaTeX.", "")
        CTXT = CTXT.replace("Move Up: Flat ground above the agent.", "Move North: Flat ground north of the agent.")
        CTXT = CTXT.replace("Move Down: Flat ground below the agent.", "Move South: Flat ground south of the agent.")
        CTXT = CTXT.replace("Move Left: Flat ground left to the agent.", "Move West: Flat ground west of the agent.")
        CTXT = CTXT.replace("Move Right: Flat ground right to the agent.", "Move East: Flat ground east of the agent.")
        self.desc = CTXT
        self.score_tracker = 0
        super().__init__(area, view, size, reward, length, seed)

    def reset(self):
        self.history.reset()
        super().reset()
        obs, reward, done, info = self.step(0)
        self.score_tracker = 0 + sum([1. for k,v in info['achievements'].items() if v>0])
        info.update({'manual': self.desc,
                'obs': describe_frame(info, None),
                'history': self.history.describe(),
                'score': self.score_tracker,
                'done': done,
                'completed': 0,
                })
        self.history.step(info)
        return obs, info
    
    def step(self, action):
        obs, reward, done, info = super().step(action)
        self.score_tracker = self.score_tracker + sum([1. for k,v in info['achievements'].items() if v>0])
        info.update({'manual': self.desc,
                'obs': describe_frame(info, self.action_list[action]),
                'history': self.history.describe(),
                'score': self.score_tracker,
                'done': done,
                'completed': 0,
                })
        self.history.step(info)
        return obs, reward, done, info
