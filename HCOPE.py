import torch
import yaml
import numpy as np
import os
from scipy.stats import t
from torch.utils.data import DataLoader
from DataProcess import OfflineDataset, SafetyTestDataset
from Agent import OfflineRandomEnsembleMixtureAgent


class HCOPE:
    def __init__(self, testing_data, device):
        self.passed = 0
        self.total = 0
        self.dataset = testing_data
        self.device = device

    def PDIS(self, agent, gamma):
        PDIS_array = []
        N = len(self.dataset)
        for sampled_episode in self.dataset:
            state = sampled_episode.get_state()
            action = sampled_episode.get_action()
            reward = sampled_episode.get_reward()
            transition = sampled_episode.get_transition()

            action_prob = agent.get_action_prob(state)
            length = len(sampled_episode)
            cur_gamma, IS_weight = 1.0, 1.0
            PDIS_H = 0.0
            for t in range(length):
                IS_weight *= action_prob[t][action[t]] / transition[t]
                PDIS_H += cur_gamma * IS_weight * reward[t]
                cur_gamma *= gamma
            PDIS_array.append(PDIS_H)
        PDIS_array = np.array(PDIS_array)
        PDIS_hat = np.sum(PDIS_array) / N
        sigma = np.sqrt(np.sum(np.power(PDIS_array - PDIS_hat, 2)) / (N - 1))
        print(PDIS_hat)
        return PDIS_hat, sigma

    def safety_test(self, threshold, gamma, agent=None):
        if agent is None:
            with open('checkpoint/checkpoint.pth', 'rb') as file:
                # agent = torch.load(file,map_location=lambda storage,loc:storage.cuda(0)) # CPU to GPU
                # agent = torch.load(file,map_location=lambda storage,loc:storage)  # GPU to CPU
                agent = torch.load(file)
        PDIS_hat, sigma = self.PDIS(agent, gamma)
        estimated_value = PDIS_hat - sigma / np.sqrt(len(self.dataset)) * t.ppf(1 - 0.01, len(self.dataset) - 1)
        self.total += 1
        if estimated_value > threshold:
            self.passed += 1
            print("Estimated J is: {}, Pass safety test! Current pass ratio: {}".format(estimated_value,float(self.passed / self.total)))
        else:
            print("Estimated J is {}, No solution found!".format(estimated_value))


def main():
    with open('config.yml', 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)
    testing_data = SafetyTestDataset(config['TEST_DATA_PATH'], config['TEST_INDEX_PATH'],
                                     int(1000000 * config['TEST_PERCENTAGE']), config['STATE_DIMENSION'],
                                     torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    evaluation = HCOPE(testing_data, torch.device('cpu'))
    evaluation.safety_test(1.4153,0.95)


if __name__ == '__main__':
    main()
