import yaml
import torch
import numpy as np
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from DataProcess import *

from Agent import OfflineRandomEnsembleMixtureAgent
from HCOPE import HCOPE


def train(config: dict):
    """
    Trains an agent in an offline setting. Parameters are passed via a config dict.
    :param config: Dictionary with parameters.
    """
    observation_space = config['STATE_DIMENSION']
    action_space = config['ACTION_DIMENSION']

    if os.path.exists(config['CHECKPOINT_PATH']):
        print("Checkpoint found, loading previous agent......")
        with open(config['CHECKPOINT_PATH'], 'rb') as file:
            agent = torch.load(file)
    else:
        print("No checkpoint found, initializing new agent......")
        agent = OfflineRandomEnsembleMixtureAgent(observation_space, action_space, config)
    agent.print_model()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Utilizing device {}'.format(device))

    print('Initializing Training Data......')
    training_data = OfflineDataset(config['TRAIN_DATA_PATH'], config['TRAIN_INDEX_PATH'],
                                   int(1000000 * config['TRAIN_PERCENTAGE']),
                                   config['STATE_DIMENSION'], config['TRAIN_DATA_DIRECTORY'])
    print('Initializing Testing Data......')
    testing_data = SafetyTestDataset(config['TEST_DATA_PATH'], config['TEST_INDEX_PATH'],
                                     int(1000000 * config['TEST_PERCENTAGE']), config['STATE_DIMENSION'],
                                     torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    evaluation = HCOPE(testing_data)
    data_loader = DataLoader(dataset=training_data,
                             batch_size=config['BATCH_SIZE'],
                             shuffle=True,
                             pin_memory=True,
                             num_workers=8)

    print('Start training with {} epochs'.format(config['EPOCHS']))
    if not os.path.exists('checkpoint'):
        os.makedirs('checkpoint')
    for epoch in range(1, config['EPOCHS'] + 1):
        print("############### EPOCH:"+str(epoch)+" ###############")
        for i_batch, sample_batched in enumerate(tqdm(data_loader, leave=False)):
            agent.learn(sample_batched)
            if i_batch == len(data_loader)-1 or (i_batch > 0 and i_batch % config['CHECKPOINT_SAVE_INTERVAL'] == 0):
                print("Batch trained:" + str(i_batch) + " Loss for new batches:" + str(agent.get_total_loss()),"Checkpoint......")
                # Save agent
                with open(config['CHECKPOINT_PATH'], 'wb') as file:
                    torch.save(agent, file)
                # Do evaluation and safety test
                if epoch > 10:
                    evaluation.safety_test(config['LOWER_BOUND'], config['GAMMA'], agent)


def main():
    with open('config.yml', 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)
    split_dataset(config)
    train(config)


if __name__ == '__main__':
    main()
