import copy
class HistoryTracker:

    def __init__(self, max_steps) -> None:
        self.max_steps = max_steps
        self.game_step = 0
        self.reset()

    def step(self, info) -> None:
        self.info.append(copy.copy(info))
        if len(self.info) > self.max_steps:
            self.info.pop(0)
        self.game_step += 1

    def reset(self) -> None:
        self.info = []
        self.game_step = 0
    
    def describe(self, game_step=None):
        if len(self.info) == 0:
            return ""
        game_step = self.game_step if game_step is None else game_step
        result = "Most recent {} steps of the player's in-game observation:\n\n".format(len(self.info))
        for i, info in enumerate(self.info):
            result += "Player Observation Step {}:\n".format(game_step - len(self.info) + i)
            result += info["obs"] + "\n\n"
        return result.strip()
    
    def score(self):
        return sum([info["score"] for info in self.info])


def describe_act(action_list):
    return "List of all actions:\n" + "\n".join(["{}. {}".format(i+1, s) for i,s in enumerate(action_list)])
