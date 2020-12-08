import pandas as pd
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
from torch.nn.functional import one_hot
import yaml
import pickle
import datetime


# Split original dataset into 2 groups(training set and safety test set), each group has a episode file and a index file
def split_dataset(config):
    print("Splitting Dataset......")
    with open(config['ORIGINAL_DATA_PATH'], 'r') as original_file:
        # Read total count 1000000
        total_episode_count = int(original_file.readline())
        train_episode_count = int(total_episode_count * float(config['TRAIN_PERCENTAGE']))
        test_episode_count = total_episode_count - train_episode_count

        # Split dataset into 2 groups. Each group has two files storing index and episodes
        # Do split every call
        if os.path.exists(config['TRAIN_DATA_PATH']):
            os.remove(config['TRAIN_DATA_PATH'])
        if os.path.exists(config['TRAIN_INDEX_PATH']):
            os.remove(config['TRAIN_INDEX_PATH'])
        if os.path.exists(config['TEST_DATA_PATH']):
            os.remove(config['TEST_DATA_PATH'])
        if os.path.exists(config['TEST_INDEX_PATH']):
            os.remove(config['TEST_INDEX_PATH'])
        current_train_episode = 0
        with open(config['TRAIN_DATA_PATH'], 'a') as train_episode_file:
            with open(config['TRAIN_INDEX_PATH'], 'a') as train_index_file:
                while current_train_episode < train_episode_count:
                    episode_length = int(original_file.readline())

                    # Write index
                    train_index_file.write(str(episode_length) + '\n')
                    current_train_episode += 1

                    # Write episode
                    for row in range(episode_length):
                        train_episode_file.write(original_file.readline())

        current_test_episode = 0
        with open(config['TEST_DATA_PATH'], 'a') as test_episode_file:
            with open(config['TEST_INDEX_PATH'], 'a') as test_index_file:
                while current_test_episode < test_episode_count:
                    episode_length = int(original_file.readline())

                    # Write index
                    test_index_file.write(str(episode_length) + '\n')
                    current_test_episode += 1

                    # Write episode
                    for row in range(episode_length):
                        test_episode_file.write(original_file.readline())
    print("Successfully splitted dataset!")


class OfflineDataset(Dataset):
    """
    PyTorch Dataset class for trajectories in vector form.
    """

    def __init__(self, episode_path, index_path, total_episodes, state_dim, folder):
        self.total_steps = 0
        print("Generating Training Dataset...")
        state, action, reward, transition, next_state, done = [], [], [], [], [], []
        data = pd.read_csv(episode_path, header=None)
        index = pd.read_csv(index_path, header=None)
        start_index, end_index = 0, 0
        for count in range(total_episodes):
            episode_length = index.iloc[count, 0]
            end_index += episode_length

            cur_state = data.iloc[start_index:end_index, 0].tolist()
            state += cur_state
            action += data.iloc[start_index:end_index, 1].tolist()
            reward += data.iloc[start_index:end_index, 2].tolist()
            transition += data.iloc[start_index:end_index, 3].tolist()
            next_state += cur_state[1:] + [0]
            done += [False] * episode_length
            done[-1] = True

            start_index = end_index
        self.state = one_hot(torch.tensor(state, dtype=torch.int64), state_dim).float()
        self.action = torch.tensor(action, dtype=torch.int64)
        self.reward = torch.tensor(reward, dtype=torch.float)
        self.transition = torch.tensor(transition, dtype=torch.float)
        self.next_state = one_hot(torch.tensor(next_state, dtype=torch.int64), state_dim).float()
        self.done = torch.tensor(done, dtype=torch.bool)
        print("Done!")

    def __len__(self):
        return len(self.reward)

    def __getitem__(self, item):
        sample = {
            'state': self.state[item, :],
            'action': self.action[item],
            'reward': self.reward[item],
            'done': self.done[item],
            'new_state': self.next_state[item, :],
            'transition': self.transition[item]
        }
        return sample


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
    def __init__(self, episode_path, index_path, total_episodes, state_dim):
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

            state = one_hot(torch.tensor(state, dtype=torch.int64), state_dim).float()
            action = torch.tensor(action, dtype = torch.int64)
            reward = torch.tensor(reward).float()
            transition = torch.tensor(transition).float()
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


class Summary(object):
    """
    Logs metrics to tensorboard files
    """

    def __init__(self, directory, agent_name, cfg):
        """
        Initializes a summary object.
        :param directory: Saving directory of dirs
        :param agent_name: Subfolder for the logs
        :param cfg: Optional dictionary with parameters to be saved.
        """
        self.directory = os.path.join(directory,
                                      agent_name,
                                      datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        self.writer = SummaryWriter(log_dir=self.directory)
        self.step = 1
        self.episode = 1

        if cfg is not None:
            params = {
                'AGENT': cfg['AGENT'],
                'TRAIN_DATA_PATH': cfg['TRAIN_DATA_PATH'],
                'EPOCHS': int(cfg['EPOCHS']),
                'BATCH_SIZE': int(cfg['BATCH_SIZE']),
                'EVAL_EPISODES': int(cfg['EVAL_EPISODES']),
                'LEARNING_RATE': cfg['LEARNING_RATE'],
                'GAMMA': cfg['GAMMA'],
                'NUM_HEADS': int(cfg['NUM_HEADS']),
                'TARGET_UPDATE_INTERVAL': int(cfg['TARGET_UPDATE_INTERVAL']),
                'SUMMARY_CHECKPOINT': int(cfg['SUMMARY_CHECKPOINT'])
            }
            self.writer.add_hparams(hparam_dict=params, metric_dict={})

    def add_scalar(self, tag: str, value, episode: bool = False):
        """
        Add a scalar to the summary
        :param tag: Tag of scalar
        :param value: Value of scalar
        :param episode: Is the scalar accountable for a step or episode
        """
        step = self.step
        if episode:
            step = self.episode

        self.writer.add_scalar(tag, value, step)

    def adv_step(self):
        """
        Increase step counter
        """
        self.step += 1

    def adv_episode(self):
        """
        Increase episode counter
        """
        self.episode += 1

    def close(self):
        """
        Flush the cached metrics and close writer.
        """
        self.writer.flush()
        self.writer.close()


def main():
    with open('config.yml', 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)
        a = OfflineDataset(config['TEST_DATA_PATH'], config['TEST_INDEX_PATH'])


if __name__ == '__main__':
    main()
