import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from DataProcess import split_dataset
from scipy.stats import t


def Explore():
    data = pd.read_csv('data/training_episodes.csv', header=None)
    print(data[(data[0] == 6)][1].unique())


class Episode:
    def __init__(self, state, action, reward, transition, length):
        self.length = length
        self.action = action
        self.state = state
        self.reward = reward
        self.transition = transition

    def __len__(self):
        return self.length

    def get_state(self):
        return self.state

    def get_action(self):
        return self.action

    def get_reward(self):
        return self.reward

    def get_transition(self):
        return self.transition


class SafetyTestDataset:
    def __init__(self, episode_path, index_path, total_episodes, state_dim, device):
        print("Generating safety test dataset......")
        self.episodes = []
        self.index = 0
        data = pd.read_csv(episode_path, header=None)
        index = pd.read_csv(index_path, header=None)
        start_index, end_index = 0, 0
        for count in range(total_episodes):
            episode_length = index.iloc[count, 0]
            end_index += episode_length

            state = data.iloc[start_index:end_index, 0].tolist()
            action = data.iloc[start_index:end_index, 1].tolist()
            reward = data.iloc[start_index:end_index, 2].tolist()
            transition = data.iloc[start_index:end_index, 3].tolist()

            state = torch.tensor(state, dtype=torch.int64)
            action = torch.tensor(action, dtype=torch.int64).to(device)
            reward = torch.tensor(reward).float().to(device)
            transition = torch.tensor(transition).float().to(device)
            self.episodes.append(Episode(state, action, reward, transition, episode_length))

            start_index = end_index
        print("Done!")

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, item):
        return self.episodes[item]

    def __next__(self):
        if self.index > len(self.episodes):  # 退出循环的条件
            raise StopIteration();
        temp = self.episodes[self.index]
        self.index += 1
        return temp  # 返回下一个值


policy = [[0.0675, 0.9323, 0.0000, 0.0002],
          [0.0000, 1.0000, 0.0000, 0.0000],
          [0.0116, 0.9051, 0.0000, 0.0833],
          [0.0001, 0.9999, 0.0000, 0.0001],
          [0.0000, 0.0032, 0.0000, 0.9968],
          [0.0001, 0.0001, 0.0000, 0.9998],
          [0.0000, 1.0000, 0.0000, 0.0000],
          [0.1360, 0.8582, 0.0000, 0.0058],
          [0.6982, 0.0042, 0.0000, 0.2976],
          [0.0999, 0.0008, 0.0000, 0.8994],
          [0.0080, 0.9918, 0.0000, 0.0002],
          [0.0000, 0.9999, 0.0000, 0.0001],
          [0.0000, 0.5916, 0.0000, 0.4084],
          [0.9977, 0.0000, 0.0000, 0.0022],
          [0.0237, 0.9763, 0.0000, 0.0000],
          [0.0556, 0.9430, 0.0000, 0.0014],
          [0.9969, 0.0023, 0.0000, 0.0008],
          [0.0000, 0.9999, 0.0000, 0.0001]]


def PDIS(dataset, gamma):
    PDIS_array = []
    N = len(dataset)
    for sampled_episode in tqdm(dataset):
        state = sampled_episode.get_state()
        action = sampled_episode.get_action()
        reward = sampled_episode.get_reward()
        transition = sampled_episode.get_transition()

        # action_prob = agent.get_action_prob(state)
        length = len(sampled_episode)
        cur_gamma, IS_weight = 1.0, 1.0
        PDIS_H = 0.0
        for t in range(length):
            IS_weight *= policy[state[t]][action[t]] / transition[t]
            PDIS_H += cur_gamma * IS_weight * reward[t]
            cur_gamma *= gamma
        PDIS_array.append(PDIS_H)
    PDIS_array = np.array(PDIS_array)
    PDIS_hat = np.sum(PDIS_array).item() / N
    sigma = np.sqrt(np.sum(np.power(PDIS_array - PDIS_hat, 2)) / (N - 1))
    return PDIS_hat, sigma


def main():
    with open('config.yml', 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)
    split_dataset(config)
    testing_data = SafetyTestDataset(config['TEST_DATA_PATH'], config['TEST_INDEX_PATH'],
                                     int(1000000 * config['TEST_PERCENTAGE']), config['STATE_DIMENSION'],
                                     torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    PDIS_hat, sigma = PDIS(testing_data, 0.95)
    estimated_value = PDIS_hat - sigma / np.sqrt(len(testing_data)) * t.ppf(1 - 0.01, len(testing_data) - 1)
    print(estimated_value, PDIS_hat, sigma)


if __name__ == '__main__':
    Explore()
