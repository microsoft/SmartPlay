from gym.envs.registration import register

from .rock_paper_scissor import RockPaperScissorBasic
from .rock_paper_scissor import RockPaperScissorDifferentScore

environments = [
    ['RockPaperScissorBasic', 'v0'],
    ['RockPaperScissorDifferentScore', 'v0'],
]

for environment in environments:
    register(
        id='{}-{}'.format(environment[0], environment[1]),
        entry_point='smartplay.rock_paper_scissors:{}'.format(environment[0]),
        # group='smartplay',
    )
