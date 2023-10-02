from gym.envs.registration import register

register(
    id = "msgr-train-v1",
    entry_point="messenger.envs:TwoEnvWrapper",
    kwargs = dict(
        stage=1,
        split_1="train_mc",
        split_2="train_sc",
        prob_env_1=0.75
    )
)

register(
    id = "msgr-train-sc-v1",
    entry_point="messenger.envs:StageOne",
    kwargs = dict(
        split="train_sc",
    )
)

register(
    id = "msgr-train-mc-v1",
    entry_point="messenger.envs:StageOne",
    kwargs = dict(
        split="train_mc",
    )
)

register(
    id = "msgr-val-v1",
    entry_point="messenger.envs:StageOne",
    kwargs = dict(
        split="val",
    )
)

register(
    id = "msgr-test-v1",
    entry_point="messenger.envs:StageOne",
    kwargs = dict(
        split="test",
    )
)

register(
    id = "msgr-train-v2",
    entry_point="messenger.envs:TwoEnvWrapper",
    kwargs = dict(
        stage=2,
        split_1="train_mc",
        split_2="train_sc",
        prob_env_1=0.75
    )
)

register(
    id = "msgr-train-sc-v2",
    entry_point="messenger.envs:StageTwo",
    kwargs = dict(
        split="train_sc",
    )
)

register(
    id = "msgr-train-mc-v2",
    entry_point="messenger.envs:StageTwo",
    kwargs = dict(
        split="train_mc",
    )
)

register(
    id = "msgr-val-v2",
    entry_point="messenger.envs:StageTwo",
    kwargs = dict(
        split="val",
    )
)

register(
    id = "msgr-test-v2",
    entry_point="messenger.envs:StageTwo",
    kwargs = dict(
        split="test",
    )
)

register(
    id = "msgr-test-se-v2",
    entry_point="messenger.envs:StageTwo",
    kwargs = dict(
        split="test_se",
    )
)

register(
    id = "msgr-train-v3",
    entry_point="messenger.envs:TwoEnvWrapper",
    kwargs = dict(
        stage=3,
        split_1="train_mc",
        split_2="train_sc",
        prob_env_1=0.75,
    )
)

register(
    id = "msgr-train-sc-v3",
    entry_point="messenger.envs:StageThree",
    kwargs = dict(
        split="train_sc",
    )
)

register(
    id = "msgr-train-mc-v3",
    entry_point="messenger.envs:StageThree",
    kwargs = dict(
        split="train_mc",
    )
)


register(
    id = "msgr-val-v3",
    entry_point="messenger.envs:StageThree",
    kwargs = dict(
        split="val",
    )
)

register(
    id = "msgr-test-v3",
    entry_point="messenger.envs:StageThree",
    kwargs = dict(
        split="test",
    )
)