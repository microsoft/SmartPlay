from gym.envs.registration import register

from .minedojo_env import MineDojoEnv

environments = [
    ['MinedojoCreative0', 'v0'],
    ['MinedojoCreative1', 'v0'],
    ['MinedojoCreative2', 'v0'],
    ['MinedojoCreative4', 'v0'],
    ['MinedojoCreative5', 'v0'],
    ['MinedojoCreative7', 'v0'],
    ['MinedojoCreative8', 'v0'],
    ['MinedojoCreative9', 'v0'],
]

for environment in environments:
    register(
        id='{}-{}'.format(environment[0], environment[1]),
        entry_point='smartplay.minedojo:MineDojoEnv',
        kwargs={'task_id': environment[0][-1]}
    )
