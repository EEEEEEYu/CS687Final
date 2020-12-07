import torch
import yaml
from torch.utils.data import DataLoader
from DataProcess import OfflineDataset, SafetyTestDataset
from Agent import OfflineRandomEnsembleMixtureAgent


def PDIS(data_loader, agent):
    for sampled_batch in data_loader:
        # action_prob = agent.get_action_prob(sampled_batch)
        print(sampled_batch[0])


def SafetyTest(agent, dataset, threshold):
    pass


def HCOPE(config):
    testing_data = SafetyTestDataset(config['TEST_DATA_PATH'], config['TEST_INDEX_PATH'],
                                     int(1000000 * config['TEST_PERCENTAGE']), config['STATE_DIMENSION'])
    data_loader = DataLoader(dataset=testing_data,
                             batch_size=config['BATCH_SIZE'],
                             shuffle=True,
                             pin_memory=True,
                             num_workers=8)
    agent = torch.load(config['CHECKPOINT_PATH'])
    PDIS(data_loader, agent)


def main():
    with open('config.yml', 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)
    HCOPE(config)


if __name__ == '__main__':
    main()
