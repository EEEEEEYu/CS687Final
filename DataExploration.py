import yaml
import pandas as pd
from DataProcess import generate_episode


lists = generate_episode('data/small_dataset_0.csv','data/small_dataset_index_0.csv')
states = range(18)
actions = range(4)
all_state_action_pair = []
for state in states:
    for action in actions:
        all_state_action_pair.append((state,action))
all_state_action_pair = set(all_state_action_pair)

complete_episode_count = 0
for e in lists:
    state_action_pair = []
    for s_a in zip(e.state,e.action):
        state_action_pair.append(s_a)
    state_action_pair=set(state_action_pair)
    if state_action_pair==all_state_action_pair:
        complete_episode_count+=1
    if len(state_action_pair)>60:
        complete_episode_count+=1

print(complete_episode_count)
print("ratio:",float(complete_episode_count/100000))


