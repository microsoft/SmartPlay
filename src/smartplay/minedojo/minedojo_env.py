import gym
from gym import error, spaces, utils
from gym.utils import seeding
from ..utils import HistoryTracker, describe_act

import random
import itertools
import numpy as np
import math
import cc3d
import minedojo

biomes_dict = {
    0: 'Ocean',
    1: 'Plains',
    2: 'Desert',
    3: 'Extreme Hills',
    4: 'Forest',
    5: 'Taiga',
    6: 'Swampland',
    7: 'River',
    8: 'Hell (The Nether)',
    9: 'The End',
    10: 'FrozenOcean',
    11: 'FrozenRiver',
    12: 'Ice Plains',
    13: 'Ice Mountains',
    14: 'MushroomIsland',
    15: 'MushroomIslandShore',
    16: 'Beach',
    17: 'DesertHills',
    18: 'ForestHills',
    19: 'TaigaHills',
    20: 'Extreme Hills Edge',
    21: 'Jungle',
    22: 'JungleHills',
    23: 'JungleEdge',
    24: 'Deep Ocean',
    25: 'Stone Beach',
    26: 'Cold Beach',
    27: 'Birch Forest',
    28: 'Birch Forest Hills',
    29: 'Roofed Forest',
    30: 'Cold Taiga',
    31: 'Cold Taiga Hills',
    32: 'Mega Taiga',
    33: 'Mega Taiga Hills',
    34: 'Extreme Hills+',
    35: 'Savanna',
    36: 'Savanna Plateau',
    37: 'Mesa',
    38: 'Mesa Plateau F',
    39: 'Mesa Plateau',
    127: 'The Void',
    129: 'Sunflower Plains',
    130: 'Desert M',
    131: 'Extreme Hills M',
    132: 'Flower Forest',
    133: 'Taiga M',
    134: 'Swampland M',
    140: 'Ice Plains Spikes',
    149: 'Jungle M',
    151: 'JungleEdge M',
    155: 'Birch Forest M',
    156: 'Birch Forest Hills M',
    157: 'Roofed Forest M',
    158: 'Cold Taiga M',
    160: 'Mega Spruce Taiga',
    161: 'Redwood Taiga Hills M',
    162: 'Extreme Hills+ M',
    163: 'Savanna M',
    164: 'Savanna Plateau M',
    165: 'Mesa (Bryce)',
    166: 'Mesa Plateau F M',
    167: 'Mesa Plateau M'
}

yaw_granularity = 6
pitch_granularity = 6
FOV = 96

pitch_cnt = len(np.arange(-FOV//2, FOV//2+1, pitch_granularity))
yaw_cnt = len(np.arange(-FOV//2, FOV//2+1, yaw_granularity))

def get_direction(yaw, pitch):
    # Convert yaw and pitch to radians
    yaw_rad = math.radians(yaw)
    pitch_rad = math.radians(pitch)
    direction_list = []

    # Calculate the x, y, and z components of the direction vector
    x = math.sin(yaw_rad)* math.cos(pitch_rad)
    y = math.sin(pitch_rad)
    z = math.cos(yaw_rad) * math.cos(pitch_rad)

    # Determine the direction based on the sign of the x, y, and z components
    if y > 0.2:
        direction = "above you"
    elif y < -0.2:
        direction = "below you"
    else:
        direction = "at your level"

    if z > 0.2:
        direction_list.append("north")
    elif z < -0.2:
        direction_list.append("south")
    
    if x > 0.2:
        direction_list.append("east")
    elif x < -0.2:
        direction_list.append("west")

    return direction + " to " + ("front" if "-".join(direction_list) == "" else "-".join(direction_list))

def describe_surround(obs):
    # print("What blocks are around me?")
    # print()
    result = "Around you:\n"
    
    # Voxel size (Can be changed based on the voxel space)
    voxel_x_size = 9
    voxel_y_size = 9
    voxel_z_size = 9
    voxel_center_x = voxel_x_size // 2
    voxel_center_y = voxel_y_size // 2
    voxel_center_z = voxel_z_size // 2
    
    # Convert the 3D block name list into np array
    block_names = np.array(obs['voxels']['block_name'])
    
    # Only preserve blocks 1 meter below me
    block_names = block_names[:, voxel_z_size // 2 - 1:, :]

    # Encode the block name using its unique index
    unique_block_names, unique_block_indices = np.unique(block_names, return_inverse=True)
    block_names_int = np.array(unique_block_indices).reshape(
        block_names.shape)

    # Apply 3D connected component and extract the number of labels
    block_names_labels, labels_count = cc3d.connected_components(block_names_int, return_N = True, connectivity = 6)

    # Get the centroids of each label
    centroids = cc3d.statistics(block_names_labels)['centroids']

    # Get the map: label -> block name
    label_2_block_name = [""] * (labels_count + 1)
    for i in range(voxel_x_size):
        for j in range(block_names.shape[1]):
            for k in range(voxel_y_size):
                label = block_names_labels[i][j][k]
                block_name = block_names[i][j][k]
                label_2_block_name[label] = block_name

    # Describe the surrounding environment based on the connected component labels
    # Helper lists for printing
    x_axis_list = ["west", "east"]
    y_axis_list = ["north", "south"]
    z_axis_list = ["below you", "above you"] # The height of me is defined as the height of my feet
    
    # The map: block name -> min distance
    block2dist = dict()
    
    # The map: block name -> direction
    block2dir = dict()
    
    # Get the direction of each label based on the centroid
    for i in range(labels_count + 1):
        middle_point = centroids[i]
        block_name = label_2_block_name[i]

        # Skip the air
        if block_name == 'air': continue

        # Get the indices to the printing help list
        x_coord = 1 if middle_point[0] > voxel_center_x else 0
        z_coord = 1 if middle_point[1] > 1 else 0
        y_coord = 1 if middle_point[2] > voxel_center_y else 0

        is_x_center = math.isclose(middle_point[0], voxel_center_x)
        is_z_center = math.isclose(middle_point[1], 1)
        is_y_center = math.isclose(middle_point[2], voxel_center_y)

        # If the component is exactly centered, skip
        if is_x_center and is_y_center and is_z_center:
            continue
        
        # Update the min distance and correspoding direction for the same block name
        distance = math.sqrt((middle_point[0] - voxel_center_x)**2 + (middle_point[1] - 1)**2 + (middle_point[2] - voxel_center_y)**2)
        if block_name not in block2dist or distance < block2dist[block_name]:
            block2dist[block_name] = distance
            
            # Record the direction of this component
            direction_str_list = []

            if not is_z_center:
                direction_str_list.append(z_axis_list[z_coord])
            else:
                direction_str_list.append("")

            if not is_y_center:
                direction_str_list.append(y_axis_list[y_coord])

            # Skip when the coordinate is at the voxel center
            if not is_x_center:
                direction_str_list.append(x_axis_list[x_coord])

            direction_str = direction_str_list[0] + " to "+ ('-'.join(direction_str_list[1:]) if '-'.join(direction_str_list[1:]) != "" else "front")
            
            block2dir[block_name] = direction_str.strip()
        
    # Finally print the blocks
    for block_name in block2dist:
        direction_str = block2dir[block_name]
        result += f" - {block_name}, {'%.2f' % block2dist[block_name]} blocks away, {direction_str}\n"
        
    return result.strip()

# Describe the block the agent is facing to

def describe_cursor(obs):
    
    # Track the cursor using lidar
    cursor_block = obs["rays"]["block_name"][0]
    
    # Skip if the cursor is pointing to air
    if cursor_block == 'air':
        return "You're not aiming at any block."
    else:
        return f"You're aiming at {cursor_block}."

# Describe the surrounding entities around the agent

def describe_entity(obs):
    result = ""
    my_yaw = obs["location_stats"]["yaw"]
    my_pitch = obs["location_stats"]["pitch"]
    
    # Number of pitch/yaw rays (Can be changed based on the lidar rays)
    
    # A flag indicating where there are entities around
    see_entity = False

    # Reshape and convert the list into np array
    entity_names = np.array(obs["rays"]["entity_name"][1:].reshape(pitch_cnt,yaw_cnt))

    # Encode the block name using its unique index
    unique_entity_names, unique_entity_indices = np.unique(entity_names, return_inverse=True)
    entity_names_int = np.array(unique_entity_indices).reshape(entity_names.shape)

    # Apply 2D connected components
    entity_names_labels, labels_count = cc3d.connected_components(entity_names_int, return_N = True, connectivity = 8)

    # Get the map: label -> entity name
    label_2_entity_name = [""] * (labels_count + 1)
    for i in range(pitch_cnt):
        for j in range(yaw_cnt):
            label = entity_names_labels[i][j]
            entity_name = entity_names[i][j]
            label_2_entity_name[label] = entity_name

    # Describe each component
    for i in range(labels_count + 1):
        entity_name = label_2_entity_name[i]

        # Skip the null entity
        if entity_name == "null": 
            continue
        else: 
            see_entity = True

        # Find all the indices of this component
        all_idx = np.where(entity_names_labels == i)
        amount = np.sum(entity_names_labels == i)

        # Find the minimum distance inside each component
        min_distance = math.inf
        direction = ""
        for row, col in zip(all_idx[0], all_idx[1]):
            index = row * yaw_cnt + col
            distance = obs["rays"]["entity_distance"][index+1]
            if distance < min_distance:
                min_distance = distance
                yaw = ((col / (yaw_cnt-1)) * FOV - FOV/2 + my_yaw + 540) % 360 - 180
                pitch = (row / (pitch_cnt-1)) * 180 - 90 + my_pitch
                direction = get_direction(yaw, pitch)

        amount_description = "taking {0:.0f}% of screen".format(amount/(pitch_cnt*yaw_cnt)*100)

        result+=f" - {entity_name}, {'%.2f' % min_distance} blocks away, {direction}, {amount_description}\n"
        
    # If there are no surrounding entities that the agent can see
    if not see_entity:
        result = ""
    
    return result

# Describe the surrounding entities around the agent

def describe_obj(obs):
    result = "You see:\n"
    my_yaw = obs["location_stats"]["yaw"]
    my_pitch = obs["location_stats"]["pitch"]
    
    # Number of pitch/yaw rays (Can be changed based on the lidar rays)
    
    # A flag indicating where there are entities around
    see_entity = False

    # Reshape and convert the list into np array
    entity_names = np.array(obs["rays"]["block_name"][1:].reshape(pitch_cnt,yaw_cnt))

    # Encode the block name using its unique index
    unique_entity_names, unique_entity_indices = np.unique(entity_names, return_inverse=True)
    entity_names_int = np.array(unique_entity_indices).reshape(entity_names.shape)

    # Apply 2D connected components
    entity_names_labels, labels_count = cc3d.connected_components(entity_names_int, return_N = True, connectivity = 8)

    # Get the map: label -> entity name
    label_2_entity_name = [""] * (labels_count + 1)
    for i in range(pitch_cnt):
        for j in range(yaw_cnt):
            label = entity_names_labels[i][j]
            entity_name = entity_names[i][j]
            label_2_entity_name[label] = entity_name

    # Describe each component
    for i in range(labels_count + 1):
        entity_name = label_2_entity_name[i]

        # Skip the null entity
        if entity_name in ("null", "air"): 
            continue
        else: 
            see_entity = True

        # Find all the indices of this component
        all_idx = np.where(entity_names_labels == i)
        amount = np.sum(entity_names_labels == i)

        # Find the minimum distance inside each component
        min_distance = math.inf
        direction = ""
        for row, col in zip(all_idx[0], all_idx[1]):
            index = row * yaw_cnt + col
            distance = obs["rays"]["block_distance"][index+1]
            if distance < min_distance:
                min_distance = distance
                yaw = ((col / (yaw_cnt-1)) * FOV - FOV/2 + my_yaw + 540) % 360 - 180
                pitch = ((row / (pitch_cnt-1)) * 180 - 90) + my_pitch
                direction = get_direction(yaw, pitch)
        
        amount_description = "taking {0:.0f}% of screen".format(amount/(pitch_cnt*yaw_cnt)*100)

        result+=f" - {entity_name}, {'%.2f' % min_distance} blocks away, {direction}, {amount_description}\n"
        
    # If there are no surrounding entities that the agent can see
    if not see_entity:
        result = ""

    return result.strip()

# Describe the exact coordinate the agent is in
def describe_location(obs):
    result = ""

    coord_list = obs["location_stats"]["pos"]
    
    # Describe the direction the agent is currently facing
    yaw = (obs["location_stats"]["yaw"]+180) % 360 - 180
    
    direction_list = ["north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west"]
    direction_index = (int((yaw // 22.5) + 1)) // 2 # Calculate the index mapping to the direction_list above
    if direction_index == 4:
        direction_index *= -1
    direction_index += 4

    result = f"Coordinate ({'%.2f' % coord_list[0]},{'%.2f' % coord_list[1]},{'%.2f' % coord_list[2]}). Facing {direction_list[direction_index]}."    # Describe whether the agent is looking up or down
    pitch = obs["location_stats"]["pitch"]

    if pitch < 0:
        result += " Looking up."
    elif pitch > 0:
        result += " Looking down."

    return result.strip()

def describe_frame(obs):
    return describe_location(obs)+"\n\n"+describe_cursor(obs)+"\n"+describe_surround(obs)+"\n"+describe_entity(obs)+"\n"+describe_obj(obs)

class MineDojoEnv(gym.Env):
    default_iter = 20
    default_steps = 200

    def __init__(self, task_id, max_steps=2):

        # Creative task
        self._env = minedojo.make(
            task_id="creative:{}".format(task_id),
            fast_reset=False,
            image_size=(160, 256),
            use_voxel=True,
            voxel_size=dict(xmin=-4, ymin=-4, zmin=-4, xmax=4, ymax=4, zmax=4), # Set voxel space to be 9*9*9
            use_lidar=True,
            lidar_rays=[(0,0,3)] + [
                    (np.pi * pitch / 180, np.pi * yaw / 180, 65)
                    for pitch in np.arange(-FOV//2, FOV//2+1, pitch_granularity)
                    for yaw in np.arange(-FOV//2, FOV//2+1, yaw_granularity)
            ]   # Track the agent's cursor
        )
        goal = self._env.task_prompt.replace(".", "")[self._env.task_prompt.find('find ')+len('find '):]
        self.goal_set = {k for k,v in biomes_dict.items() if goal in v.lower()}
        self.action_list = ["Move North", "Move East", "Move South", "Move West"]
        self.noop_step=50

        self.history = HistoryTracker(max_steps)
        self.desc = """
You are in Minecraft and your goal is to find a {} biome. You are not allowed to craft anything.

In your observation, you are provided the amount of space an object takes in your field of view. Note that objects of the same size takes more space when they are closer to you.

{}
""".format(goal, describe_act(self.action_list)).strip()
        
    def get_turn_action(self, act, obs):
        yaw = obs["location_stats"]["yaw"]
        # Compute yaw delta to face the yaw indicated by the action 'act'
        action_yaw = act * 90 - 180
        yaw_delta = (action_yaw - yaw + 180) % 360 // 15
        action = self._env.action_space.no_op()
        action[4] = yaw_delta
        return action

    def go_forward_action(self):
        action = self._env.action_space.no_op()
        action[0] = 1 # Move forward
        action[1] = 0 if random.random() < 0.5 else (1 if random.random() < 0.5 else 2) # Move left/right with prob=0.5 to get unstuck
        action[2] = 3 if random.random() < 0.5 else 1 # Sprint with prob=0.5
        return action

    def describe(self, obs, action=None):
        if action is not None:
            return "You took action {}.\n\n{}".format(self.action_list[action], describe_frame(obs))
        else:
            return describe_frame(obs)

    def reset(self):
        obs = self._env.reset()
        self.history.reset()
        info = {
            "obs": self.describe(obs),
            "manual": self.desc,
            "history": self.history.describe(),
            "score": 0,
            "completed": 0,
            }
        self.history.step(info)
        self.obs = obs
        return obs, info

    def step(self, action):
        R = 0
        obs, reward, done, info = self._env.step(self.get_turn_action(action, self.obs))
        R += reward
        for _ in range(self.noop_step):
            obs, reward, done, info = self._env.step(self.go_forward_action())
            R += reward
            if obs["location_stats"]["biome_id"].item() in self.goal_set:
                break

        info.update({
            "obs": self.describe(obs, action),
            "manual": self.desc,
            "history": self.history.describe(),
            "score": 0,
            "completed": 0,
            })
        if obs["location_stats"]["biome_id"].item() in self.goal_set:
            info["completed"] = 1
            info["score"] = 1
            done = True
        self.history.step(info)
        self.obs = obs
        return obs, R, done, info
