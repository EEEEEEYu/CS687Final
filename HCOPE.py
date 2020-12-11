import torch
import yaml
import numpy as np
import os
from tqdm import tqdm
from scipy.stats import t
from torch.utils.data import DataLoader
from DataProcess import OfflineDataset, SafetyTestDataset, split_dataset
from Agent import OfflineRandomEnsembleMixtureAgent


class HCOPE:
    def __init__(self, testing_data, device, config):
        self.passed = 0
        self.total = 0
        self.dataset = testing_data
        self.device = device
        self.config = config

    def PDIS(self, agent, gamma):
        PDIS_array = []
        N = len(self.dataset)
        for sampled_episode in tqdm(self.dataset):
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
        PDIS_hat = np.sum(PDIS_array).item() / N
        sigma = np.sqrt(np.sum(np.power(PDIS_array - PDIS_hat, 2)) / (N - 1))
        return PDIS_hat, sigma

    def safety_test(self, threshold, gamma, checkpoint, agent = None):
        print("Doing safety test......")
        if agent is None:
            if os.path.exists('checkpoint/' + checkpoint):
                with open('checkpoint/' + checkpoint, 'rb') as file:
                    # agent = torch.load(file,map_location=lambda storage,loc:storage.cuda(0)) # CPU to GPU
                    # agent = torch.load(file,map_location=lambda storage,loc:storage)  # GPU to CPU
                    agent = torch.load(file, map_location = lambda storage, loc: storage)
                    agent.set_cpu()
            else:
                print("Using random agent for testing......")
                agent = OfflineRandomEnsembleMixtureAgent(18, 4, self.config)
        PDIS_hat, sigma = self.PDIS(agent, gamma)
        print("Average PDIS:{} sigma:{}".format(PDIS_hat, sigma))
        estimated_value = PDIS_hat - sigma / np.sqrt(len(self.dataset)) * t.ppf(1 - 0.01, len(self.dataset) - 1)
        self.total += 1
        if estimated_value > threshold:
            self.passed += 1
            print("Estimated J is: {}, Pass safety test! Current pass ratio: {}".format(estimated_value, float(
                self.passed / self.total)))

        else:
            print("Estimated J is {}, No solution found!".format(estimated_value))

        return estimated_value

    def calculate_estimation(self):
        print("Calculating all policy evaluations......")
        file_list = os.listdir('checkpoint')
        file_list = sorted(file_list)
        if not os.path.exists('rank'):
            os.makedirs('rank')
        if not os.path.exists('rank/already_done.pth'):
            with open('rank/already_done.pth','wb') as file:
                torch.save(0,file)
        if not os.path.exists('rank/rank.pth'):
            with open('rank/rank.pth','wb') as file:
                torch.save({},file)
        with open('rank/already_done.pth', 'rb') as file:
            already_done = torch.load(file)
        with open('rank/rank.pth','rb') as file:
            estimation_dict = torch.load(file)
        for index, checkpoint in enumerate(file_list):
            if index < already_done:
                continue
            print("Estimating policy{}......".format(index))
            estimated_value = self.safety_test(self.config['LOWER_BOUND'], self.config['GAMMA'], checkpoint)
            estimation_dict[checkpoint] = estimated_value
            with open("rank/rank.pth", 'wb') as file:
                torch.save(estimation_dict, file)
            with open("rank/already_done.pth",'wb') as file:
                torch.save(index+1,file)

    def generate_policy(self):
        print("Dumping policy......")
        with open('rank/rank.pth', 'rb') as file:
            estimation_dict = torch.load(file)
            desc_sorted_agents = sorted(list(estimation_dict.items()), key = lambda x: -x[1])
            desc_sorted_agents = desc_sorted_agents[0:100]
            for i in range(len(desc_sorted_agents) if len(desc_sorted_agents)<100 else 100):
                agent = torch.load('checkpoint/'+desc_sorted_agents[i][0], map_location=lambda storage, loc: storage)
                agent.set_cpu()
                agent.dump_policy(self.config['STATE_DIMENSION'], self.config['ACTION_DIMENSION'])
        print("Done!")


def main():
    with open('config.yml', 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader = yaml.FullLoader)
    split_dataset(config)
    testing_data = SafetyTestDataset(config['TEST_DATA_PATH'], config['TEST_INDEX_PATH'],
                                     int(1000000 * config['TEST_PERCENTAGE']), config['STATE_DIMENSION'],
                                     torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    evaluation = HCOPE(testing_data, torch.device('cuda' if torch.cuda.is_available() else 'cpu'), config)
    evaluation.calculate_estimation()
    evaluation.generate_policy()


if __name__ == '__main__':
    main()
