from gym.envs.registration import register

from .messenger_env import MessengerEnv

environments = [
    ['MessengerL1', 'v0'],
    ['MessengerL2', 'v0'],
    ['MessengerL3', 'v0'],
]

for environment in environments:
    register(
        id='{}-{}'.format(environment[0], environment[1]),
        entry_point='smartplay.messenger_emma:MessengerEnv',
        kwargs={'lvl': int(environment[0][-1])}
        # group='smartplay',
    )
