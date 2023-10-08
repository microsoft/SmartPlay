benchmark_games_v0 = [
    'RockPaperScissorBasic',
    'BanditTwoArmedHighLowFixed',
    'MessengerL1',
    'MessengerL2',
    'Hanoi3Disk',
    'Crafter',
    'MinedojoCreative0',
]

def normalize_scores(env_name, score):
    if env_name in recorded_settings.keys():
        return (score-recorded_settings[env_name]['min score']) / (recorded_settings[env_name]['human score']-recorded_settings[env_name]['min score'])
    else:
        raise ValueError('Environment `{}` does not have recorded settings.'.format(env_name))