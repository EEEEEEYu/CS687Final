import pandas as pd


class episode:
    def __init__(self,state,action,reward,transition_prob):
        self.state = state
        self.action = action
        self.reward = reward
        self.transition_prob = transition_prob


def generate_episode(data_path,index_path):
    data = pd.read_csv(data_path,header = None)
    index = pd.read_csv(index_path,header = None)
    episodes = []
    start_index = 0
    end_index = 0
    for count in range(index.shape[0]):
        end_index += index.iloc[count,0]
        new_episode = episode(data.iloc[start_index:end_index,0].tolist(),data.iloc[start_index:end_index,1].tolist(),
                              data.iloc[start_index:end_index,2].tolist(),data.iloc[start_index:end_index,3].tolist())
        episodes.append(new_episode)
        start_index += index.iloc[count,0]

    print('finished')
    return episodes


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