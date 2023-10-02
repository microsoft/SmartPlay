# This file is used to register the gym environments in the OpenAI gym registry.
# This allows the environments to be used with the gym.make() function.
# The gym environments are registered with the following IDs:
# from gym.envs.registration import add_group


# add_group(
#     id='smartplay',
#     name='SmartPlay',
#     description='Benchmark for LLMs'
# )

from . import rock_paper_scissors
from . import hanoi
from . import bandits
from . import crafter
from . import minedojo
from . import messenger_emma

import itertools
env_list = []
for env_name, version in itertools.chain(rock_paper_scissors.environments, hanoi.environments, bandits.environments, crafter.environments, minedojo.environments, messenger_emma.environments):
    env_list.append('{}-{}'.format(env_name, version))