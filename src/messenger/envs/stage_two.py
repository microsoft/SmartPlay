'''
Classes that follows a gym-like interface and implements stage two of the Messenger
environment.
'''

import json
import random
from collections import namedtuple
from pathlib import Path
from os import environ
import re

# hack to stop PyGame from printing to stdout
environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

from vgdl.interfaces.gym import VGDLEnv
import numpy as np

from messenger.envs.base import MessengerEnv, Grid, Position
import messenger.envs.config as config
from messenger.envs.manual import TextManual
from messenger.envs.utils import games_from_json


# specifies the game variant (e.g. chasing enemy, fleeing message, stationary goal)
# path is path to the vgdl domain file describing the variant.
GameVariant = namedtuple(
    "GameVariant", ["path", "enemy_type", "message_type", "goal_type"]
)


class StageTwo(MessengerEnv):
    '''
    Full messenger environment with mobile sprites. Uses Py-VGDL as game engine.
    To avoid the need to instantiate a large number of games, (since there are
    P(12,3) = 1320 possible entity to role assignments) We apply a wrapper on top
    of the text and game state which masks the role archetypes (enemy, message goal)
    into entities (e.g. alien, knight, mage).
    '''

    def __init__(self, split:str, shuffle_obs=True):
        super().__init__()
        self.shuffle_obs = shuffle_obs # shuffle the entity layers

        this_folder = Path(__file__).parent
        # Get the games and manual
        games_json_path = this_folder.joinpath("games.json")
        if "train" in split and "mc" in split: # multi-combination games
            game_split = "train_multi_comb"
            text_json_path = this_folder.joinpath("texts", "text_train.json")
        elif "train" in split and "sc" in split: # single-combination games
            game_split = "train_single_comb"
            text_json_path = this_folder.joinpath("texts", "text_train.json")
        elif "val" in split:
            game_split = "val"
            text_json_path = this_folder.joinpath("texts", "text_val.json")
        elif "test" in split:
            game_split = "test"
            text_json_path = this_folder.joinpath("texts", "text_test.json")
        else:
            raise Exception(f"Split: {split} not understood.")

        # list of Game namedtuples
        self.all_games = games_from_json(json_path=games_json_path, split=game_split)
        self.text_manual = TextManual(json_path=text_json_path)

        # get the folder that has the game variants and init_states
        if "test" in split and "se" not in split: # new dynamics (se for state estimation)
            vgdl_files = this_folder.joinpath("vgdl_files", "stage_2_nd")
        else: # training dynamics
            vgdl_files = this_folder.joinpath("vgdl_files", "stage_2")
        
        # get the file paths to possible starting states
        self.init_states = [
            str(path) for path in vgdl_files.joinpath("init_states").glob("*.txt")
        ]
        # get all the game variants
        self.game_variants = [
            self._get_variant(path) for path in vgdl_files.joinpath("variants").glob("*.txt")
        ]

        # entities tracked by VGDLEnv
        self.notable_sprites = ["enemy", "message", "goal", "no_message", "with_message"]
        self.env = None # the VGDLEnv

    def _get_variant(self, variant_file:Path) -> GameVariant:
        '''
        Return the GameVariant for the variant specified by variant_file. 
        Searches through the vgdl code to find the correct type:
        {chaser, fleeing, immovable}
        '''

        code = variant_file.read_text()
        return GameVariant(
            path = str(variant_file),
            enemy_type = re.search(r'enemy > (\S+)', code)[1].lower(),
            message_type = re.search(r'message > (\S+)', code)[1].lower(),
            goal_type = re.search(r'goal > (\S+)', code)[1].lower()
        )

    def _convert_obs(self, vgdl_obs):
        '''
        Return a grid built from the vgdl observation which is a
        KeyValueObservation object (see vgdl code for details).
        '''
        entity_locs = Grid(layers=3, shuffle=self.shuffle_obs)
        avatar_locs = Grid(layers=1)

        # try to add each entity one by one, if it's not there move on.
        if 'enemy.1' in vgdl_obs:
            entity_locs.add(self.game.enemy, Position(*vgdl_obs['enemy.1']['position']))
        if 'message.1' in vgdl_obs:
            entity_locs.add(self.game.message, Position(*vgdl_obs['message.1']['position']))
        else:
            # advance the entity counter, Oracle model requires special order.
            # TODO: maybe used named layers to make this more understandable.
            entity_locs.entity_count += 1
        if 'goal.1' in vgdl_obs:
            entity_locs.add(self.game.goal, Position(*vgdl_obs['goal.1']['position']))

        if 'no_message.1' in vgdl_obs:
            '''
            Due to a quirk in VGDL, the avatar is no_message if it starts as no_message
            even if the avatar may have acquired the message at a later point.
            To check, if it has a message, check that the class vector corresponding to
            with_message is == 1.
            '''
            avatar_pos = Position(*vgdl_obs['no_message.1']['position'])
            # with_key is last position, see self.notable_sprites
            if vgdl_obs['no_message.1']['class'][-1] == 1:
                avatar = config.WITH_MESSAGE
            else:
                avatar = config.NO_MESSAGE

        elif "with_message.1" in vgdl_obs:
            # this case only occurs if avatar begins as with_message at start of episode
            avatar_pos = Position(*vgdl_obs['with_message.1']['position'])
            avatar = config.WITH_MESSAGE

        else: # the avatar is not in observation, so is probably dead
            return {"entities": entity_locs.grid, "avatar": avatar_locs.grid}

        avatar_locs.add(avatar, avatar_pos) # if not dead, add it.

        return {"entities": entity_locs.grid, "avatar": avatar_locs.grid}

    def reset(self, **kwargs):
        '''
        Resets the current environment. NOTE: We remake the environment each time.
        This is a workaround to a bug in py-vgdl, where env.reset() does not
        properly reset the environment. kwargs go to get_document().
        '''

        self.game = random.choice(self.all_games) # (e.g. enemy-alien, message-knight, goal - bear)

        # choose the game variant (e.g. enmey-chasing, message-fleeing, goal-static)
        # and initial starting location of the entities.
        variant = random.choice(self.game_variants)
        init_state = random.choice(self.init_states) # initial state file

        # args that will go into VGDL Env.
        self._envargs = {
            'game_file': variant.path,
            'level_file': init_state,
            'notable_sprites': self.notable_sprites.copy(),
            'obs_type': 'objects', # track the objects
            'block_size': 34  # rendering block size
        }
        self.env = VGDLEnv(**self._envargs)
        vgdl_obs = self.env.reset()

        manual = self.text_manual.get_document(
            enemy=self.game.enemy.name,
            message=self.game.message.name,
            goal=self.game.goal.name,
            enemy_type=variant.enemy_type,
            message_type=variant.message_type,
            goal_type=variant.goal_type,
            **kwargs
        )

        if self.shuffle_obs:
            random.shuffle(manual)
            
        return self._convert_obs(vgdl_obs), manual

    def step(self, action):
        vgdl_obs, reward, done, info = self.env.step(action)
        return self._convert_obs(vgdl_obs), reward, done, info
    
