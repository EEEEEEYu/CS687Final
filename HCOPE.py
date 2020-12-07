import torch
import yaml
from torch.utils.data import DataLoader
from DataProcess import OfflineDataset, SafetyTestDataset
from Agent import OfflineRandomEnsembleMixtureAgent


def PDIS(data_loader, agent, gamma):
    for sampled_episode in data_loader:
        state = sampled_episode.get_state()
        action = sampled_episode.get_action()
        reward = sampled_episode.get_reward()
        transition = sampled_episode.get_transition()

        action_prob = agent.get_action_prob(state)
        length = len(sampled_episode)
        cur_gamma,IS_weight = 1.0,1.0
        J_estimated = 0.0
        for t in range(length):
            IS_weight *= action_prob[t][action[t]]/transition[t]
            J_estimated += cur_gamma * IS_weight * reward[t]
            cur_gamma *= gamma
        return J_estimated


def SafetyTest(agent, dataset, threshold):
    pass


def HCOPE(config):
    testing_data = SafetyTestDataset(config['TEST_DATA_PATH'], config['TEST_INDEX_PATH'],
                                     int(1000000 * config['TEST_PERCENTAGE']), config['STATE_DIMENSION'])
    agent = OfflineRandomEnsembleMixtureAgent(config['STATE_DIMENSION'], config['ACTION_DIMENSION'], config)
    PDIS(testing_data, agent, config['GAMMA'])


def main():
    with open('config.yml', 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)
    HCOPE(config)


if __name__ == '__main__':
    main()
