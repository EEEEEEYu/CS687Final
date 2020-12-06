import torch
from torch.utils.data import DataLoader
from DataProcess import OfflineDataset


def PDIS(data_loader,agent):
    for sampled_batch in data_loader:
        action_prob = agent.get_action_prob(sampled_batch)


def SafetyTest(agent,dataset,threshold):
    pass


def HCOPE(config):
    testing_data = OfflineDataset(config['TEST_DATA_PATH'], config['TEST_INDEX_PATH'],
                                  int(1000000 * config['TEST_PERCENTAGE']),
                                  config['STATE_DIMENSION'], config['ORIGINAL_DATA_DIRECTORY'])
    data_loader = DataLoader(dataset = testing_data,
                             batch_size = config['BATCH_SIZE'],
                             shuffle = True,
                             pin_memory = True,
                             num_workers = 8)
