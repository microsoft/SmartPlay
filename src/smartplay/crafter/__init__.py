from gym.envs.registration import register

from .crafter_env import Crafter

environments = [
    ['Crafter', 'v0'],
]

for environment in environments:
    register(
        id='{}-{}'.format(environment[0], environment[1]),
        entry_point='smartplay.crafter:{}'.format(environment[0]),
        kwargs={'reward': True}
    )
