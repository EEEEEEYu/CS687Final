import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from DataProcess import *

from Agent import OfflineRandomEnsembleMixtureAgent



def train(config: dict):
    """
    Trains an agent in an offline setting. Parameters are passed via a config dict.
    :param config: Dictionary with parameters.
    """
    observation_space = config['STATE_DIMENSION']
    action_space = config['ACTION_DIMENSION']

    print('Creating Agent.')
    agent = OfflineRandomEnsembleMixtureAgent(observation_space,action_space,config)
    agent.print_model()

    print('Initializing Dataloader.')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Utilizing device {}'.format(device))
    training_data = OfflineDataset(config['TRAIN_DATA_PATH'],config['TRAIN_INDEX_PATH'],int(1000000*config['TRAIN_PERCENTAGE']),
                                   config['STATE_DIMENSION'],config['ORIGINAL_DATA_DIRECTORY'])
    data_loader = DataLoader(dataset=training_data,
                             batch_size=config['BATCH_SIZE'],
                             shuffle=True,
                             pin_memory=True,
                             num_workers = 8)

    print('Start training with {} epochs'.format(config['EPOCHS']))
    for e in range(1, config['EPOCHS'] + 1):
        for i_batch, sample_batched in enumerate(tqdm(data_loader, leave=False)):
            agent.learn(sample_batched)
            if i_batch % 500==0:
                print("Batch trained:"+str(i_batch)+" Loss for new 500 batches:"+str(agent.get_total_loss()))
        agent.save(e)


def main():
    with open('config.yml', 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)
    split_dataset(config)
    train(config)


if __name__ == '__main__':
    main()