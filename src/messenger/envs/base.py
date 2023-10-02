import random
from collections import namedtuple

import gym
from gym import spaces
import numpy as np

import messenger.envs.config as config
from messenger.envs.config import Entity

# Positions of the entities
Position = namedtuple('Position', ["x", "y"])


class MessengerEnv(gym.Env):
    '''
    Base Messenger class that defines the action and observation spaces.
    '''

    def __init__(self):
        super().__init__()
        # up, down, left, right, stay
        self.action_space = spaces.Discrete(len(config.ACTIONS))

        # observations, not including the text manual
        self.observation_space = spaces.Dict({
            "entities": spaces.Box(
                low=0,
                high=14,
                shape=(config.STATE_HEIGHT, config.STATE_WIDTH, 3)
            ),
            "avatar": spaces.Box(
                low=15,
                high=16,
                shape=(config.STATE_HEIGHT, config.STATE_WIDTH, 1)
            )
        })

    def reset(self):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError

    def render(self):
        raise NotImplementedError


class Grid:
    '''
    Class which makes it easier to build a grid observation from the dict state
    return by VGDLEnv.
    '''
    def __init__(self, layers, shuffle=True):
        '''
        layers:
            Each add() operation will place a separate entity in a new layer.
            Thus, this is the upper-limit to the number of items to be added.
        shuffle:
            Place each items in a random order.
        '''
        self.grid = np.zeros((config.STATE_HEIGHT, config.STATE_WIDTH, layers))
        self.order = list(range(layers)) # insertion order
        if shuffle:
            random.shuffle(self.order)
        self.layers = layers
        self.entity_count = 0

    def add(self, entity:Entity, position:Position):
        '''
        Add entity entity and position position.
        '''
        assert self.entity_count < self.layers, \
            f"Tried to add entity no. {self.entity_count} with {self.layers} layers."

        self.grid[position.y, position.x, self.order[self.entity_count]] = entity.id
        self.entity_count += 1