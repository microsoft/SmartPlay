'''
Default config settings used in the envs. Changing the config settings here
will have a global effect on the entire messenger package.
'''

import itertools
from pathlib import Path
import json
from collections import namedtuple

'''
An entity namedtuple consists of the entity's name (e.g. alien) and the
id (symbol) of the entity used to represent the entity in the state
'''
Entity = namedtuple("Entity", "name id")

'''
A Game namedtuple specifies an assignment of three Entity namedtuples to the roles
enemy, message, goal.
'''
Game = namedtuple("Game", "enemy message goal")

'''
ACTIONS (user set):
    Maps actions to ints that is consistent with those used in PyVGDL
STATE_HEIGHT, STATE_WIDTH (user set):
    Dimensions of the game state. Note that these must be consistent with the level
    files that PyVGDL reads.
_all_npcs (user set):
    List of all the npcs (non-player characters) in the game. Aside from the obvious 
    effects of changing the entities themselves, the order of things here are also 
    important as it determines the embedding ID.
NPCS: 
    List of Entity namedtuples consisting non-player characters (e.g. alien, bear)
ALL_ENTITIES:
    List of all Entity namedtuples. Includes NPCS, avatar, and walls.
WITH_MESSAGE:
    The avatar Entity when it has the message.
NO_MESSAGE:
    The avatar Entity without the message.
WALL:
    The wall entity.
'''

ACTIONS = namedtuple('Actions', 'up down left right stay')(0, 1, 2, 3, 4)
STATE_HEIGHT, STATE_WIDTH = 10, 10 # dimensions of the game state.

_all_npcs = ['airplane', 'mage', 'dog', 'bird', 'fish', 'scientist', 'thief',
    'ship', 'ball', 'robot', 'queen', 'sword']

NPCS = [] # non-player characters
ALL_ENTITIES = [] # all entities including wall and avatar
WITH_MESSAGE = None # avatar with the message
NO_MESSAGE = None # avatar without the message
WALL = None # the wall

# Fill in the entities
for i, name in enumerate(_all_npcs + ['wall', 'no_message', 'with_message']):
    ALL_ENTITIES.append(Entity(name, i + 2)) # 0, 1 reserved for background and dirt resp.
    
    if name not in ['wall', 'no_message', 'with_message']:
        NPCS.append(Entity(name, i + 2))

    # we also want to be able to specifically reference the non-NPC entities
    if name == 'with_message': 
        WITH_MESSAGE = Entity(name, i + 2)
    if name == 'no_message':
        NO_MESSAGE = Entity(name, i + 2)
    if name == 'wall':
        WALL = Entity(name, i + 2)