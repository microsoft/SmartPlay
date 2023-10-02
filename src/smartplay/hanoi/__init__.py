from gym.envs.registration import register

from .hanoi_env import Hanoi3Disk, Hanoi4Disk

environments = [
    ['Hanoi3Disk', 'v0'],
    ['Hanoi4Disk', 'v0'],
]

for environment in environments:
    register(
        id='{}-{}'.format(environment[0], environment[1]),
        entry_point='smartplay.hanoi:{}'.format(environment[0]),
        # group='smartplay',
    )
