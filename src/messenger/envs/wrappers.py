'''
Implements wrappers on top of the basic messenger environments
'''
import random

from messenger.envs.base import MessengerEnv
from messenger.envs.stage_one import StageOne
from messenger.envs.stage_two import StageTwo
from messenger.envs.stage_three import StageThree


class TwoEnvWrapper(MessengerEnv):
    '''
    Switches between two Messenger environments
    '''
    def __init__(self, stage:int, split_1:str, split_2:str, prob_env_1=0.5, **kwargs):
        super().__init__()
        if stage == 1:
            self.env_1 = StageOne(split=split_1, **kwargs)
            self.env_2 = StageOne(split=split_2, **kwargs)
        elif stage == 2:
            self.env_1 = StageTwo(split=split_1, **kwargs)
            self.env_2 = StageTwo(split=split_2, **kwargs)
        elif stage == 3:
            self.env_1 = StageThree(split=split_1, **kwargs)
            self.env_2 = StageThree(split=split_2, **kwargs)
        
        self.prob_env_1 = prob_env_1
        self.cur_env = None
        
    def reset(self, **kwargs):
        if random.random() < self.prob_env_1:
            self.cur_env = self.env_1
        else:
            self.cur_env = self.env_2
        return self.cur_env.reset(**kwargs)
    
    def step(self, action):
        return self.cur_env.step(action)