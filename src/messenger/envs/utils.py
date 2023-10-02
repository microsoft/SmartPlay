'''
Utilites for the environments
'''

import json
from pathlib import Path
from messenger.envs.config import NPCS, Game

def get_entity(name:str):
    '''
    Get the Entity object for the entity with
    name name.
    '''
    for entity in NPCS:
        if entity.name == name:
            return entity
    raise Exception("entity not found.")

def get_game(game_tuple):
    '''
    Take a tuple of strings (enemy, message, goal) and get the
    corresponding Game object.
    '''
    enemy_name, message_name, goal_name = game_tuple
    enemy = get_entity(enemy_name)
    message = get_entity(message_name)
    goal = get_entity(goal_name)
    return Game(enemy=enemy, message=message, goal=goal)

def games_from_json(json_path:str, split:str):
    '''
    Convert game strings in games.json to Game namedtuples
    '''
    json_path = Path(json_path)
    with json_path.open(mode="r") as json_file:
        games = json.load(json_file)
    converted = []
    for g in games[split]:
        converted.append(get_game(g))
    return converted