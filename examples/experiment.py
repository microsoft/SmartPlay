import os
os.environ["MINEDOJO_HEADLESS"]="1"
import argparse
import numpy as np
from tqdm import tqdm
import gym
import smartplay

parser = argparse.ArgumentParser()
parser.add_argument('--llm_name', type=str, default='gpt-4', help='Name of the LLM')
parser.add_argument('--env_names', type=str, default=None, help='Comma separated list of environments to run')

args = parser.parse_args()

if args.env_names is None:
    args.env_names = ','.join(smartplay.benchmark_games_v0)

LLM_name = args.llm_name

# Replace with your own LLM API.
# Note: query_model takes two arguments: 1) message in openai chat completion form (list of dictionaries), 
#                                        2) an index to indicate where the message should be truncated if the length exceeds LLM context length.
from llm_api import get_query
query_model = get_query(LLM_name)

def compose_ingame_prompt(info, question, past_qa=[]):
    messages = [
        {"role": "system", "content" : "You’re a player trying to play the game."}
    ]
    
    if len(info['manual'])>0:
        messages.append({"role": "system", "content": info['manual']})

    if len(info['history'])>0:
        messages.append({"role": "system", "content": info['history']})

    messages.append({"role": "system", "content": "current step observation: {}".format(info['obs'])})

    if len(past_qa)>0:
        for q,a in past_qa:
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": a})

    messages.append({"role": "user", "content": question})

    return messages, 2 # This is the index of the history, we will truncate the history if it is too long for LLM

questions=[
        "What is the best action to take? Let's think step by step, ",
        "Choose the best executable action from the list of all actions. Write the exact chosen action."
    ]

def run(env_name):
    normalized_scores = []
    env = gym.make("smartplay:{}-v0".format(env_name))
    env_steps = env.default_steps
    num_iter = env.default_iter

    def match_act(output):
        inds = [(i, output.lower().index(act.lower())) for i, act in enumerate(env.action_list) if act.lower() in output.lower()]
        if len(inds)>0:
            # return the action with smallest index
            return sorted(inds, key=lambda x:x[1])[0][0]
        else:
            # print("LLM failed with output \"{}\", taking action 0...".format(output))
            return 0

    rewards = []
    progresses = []
    for eps in tqdm(range(num_iter), desc="Evaluating LLM {} on {}".format(LLM_name, env_name)):
        import wandb
        wandb.init(project="SmartPlay", config={"LLM": LLM_name, "env": env_name, "eps": eps, "num_iter": num_iter, "env_steps": env_steps})
        step = 0
        trajectories = []
        qa_history = []
        progress = [0]
        reward = 0
        rewards = []
        done=False

        columns=["Context", "Step", "OBS", "History", "Score", "Reward", "Total Reward"] + questions + ["Action"]
        wandb_table = wandb.Table(columns=columns)

        _, info = env.reset()
        
        while step < env_steps:

            new_row = [info['manual'], step, info['obs'], info['history'], info['score'], reward, sum(rewards)]
            wandb.log({"metric/total_reward".format(eps): sum(rewards), 
                       "metric/score".format(eps): info['score'],
                       "metric/reward".format(eps): reward,
                       })
            
            if done:
                break
            
            qa_history = []
            for question in questions:
                prompt = compose_ingame_prompt(info, question, qa_history)
                answer = query_model(*prompt)
                qa_history.append((question, answer))
                new_row.append(answer)
                answer_act = answer

            a = match_act(answer_act)
            new_row.append(env.action_list[a])
            _, reward, done, info = env.step(a)
            rewards.append(reward)
            score=info['score']

            step += 1
            wandb_table.add_data(*new_row)

        if not done:
            completion=0
        else:
            completion=info['completed']
        progresses.append(np.max(progress))
        wandb.log({"rollout/rollout".format(eps): wandb_table, 
                "final/total_reward":sum(rewards),
                "final/score":score,
                "final/normalized_score":smartplay.normalize_score(env_name, score),
                "final/completion":completion,
                "final/episodic_step":step,
                "final/eps":eps,
                })
        normalized_scores.append(smartplay.normalize_score(env_name, score))
        del wandb_table
        wandb.finish()
    return np.average(normalized_scores)

score_dict = {}
for env_name in args.env_names.split(','):
    score_dict[env_name] = run(env_name)

print("Normalized scores on each task:", score_dict)
print("Capability scores of the LLM:", smartplay.analyze_capabilities(score_dict))
