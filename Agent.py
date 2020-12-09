import torch
import torch.nn as nn
import torchsummary
import os
import torch.optim as optim
import numpy as np
from torch.nn.functional import one_hot


class DQNMultiHead(nn.Module):
    """
    DQN Multi-Head Network for vector inputs. Network uses 1D-Convolutions with
    kernel size 1 and grouped convolutions to simulate parallel computation
    of the respective heads.
    """

    def __init__(self, observation_space: int, action_space: int, num_heads: int):
        super(DQNMultiHead, self).__init__()
        self.num_heads = num_heads
        self.action_space = action_space

        self.fc1 = nn.Linear(in_features=observation_space, out_features=128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(in_features=64, out_features=32)
        self.bn3 = nn.BatchNorm1d(32)

        self.heads = nn.Conv1d(in_channels=32 * num_heads,
                               out_channels=action_space * num_heads,
                               kernel_size=1,
                               groups=num_heads)

        self.elu = nn.ELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.elu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.elu(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.elu(x)

        x = x.repeat(1, self.num_heads).unsqueeze(-1)
        x = self.heads(x)
        return x.view(-1, self.action_space, self.num_heads).permute(2, 0, 1)


class OfflineRandomEnsembleMixtureAgent:
    """
    Implementation of REM-DQN
    """

    def __init__(self, observation_space: int, action_space: int, config: dict):
        self.observation_space = observation_space
        self.action_space = action_space
        self.name = 'RandomEnsembleMixtureAgent'
        self.batches_done = 0

        self.target_update_steps = config['TARGET_UPDATE_INTERVAL']
        self.gamma = config['GAMMA']
        self.num_heads = config['NUM_HEADS']

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('Utilizing device {}'.format(self.device))
        self.policy = DQNMultiHead(observation_space, action_space, self.num_heads).to(self.device)
        self.target = DQNMultiHead(observation_space, action_space, self.num_heads).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config['LEARNING_RATE'])
        self.loss = nn.SmoothL1Loss()
        self.loss_total = 0

    def act(self, state):
        with torch.no_grad():
            self.policy.eval()
            state = torch.tensor(state).float().unsqueeze(0).to(self.device)
            avg_q_values = torch.mean(self.policy(state), dim=0)
            action = torch.argmax(avg_q_values)
            return action.cpu().detach().numpy()

    def learn(self, batch):
        self.policy.train()

        state = batch['state'].to(self.device)
        action = batch['action'].to(self.device)
        reward = batch['reward'].to(self.device)
        done = batch['done'].to(self.device)
        new_state = batch['new_state'].to(self.device)

        # Get random alpha values where sum(alphas) = 1
        alpha = torch.rand(self.num_heads).to(self.device)
        alpha = alpha / torch.sum(alpha)
        alpha = alpha.unsqueeze(-1).expand(-1, len(action))

        # Get Q(s,a) for actions taken and weigh
        actions = action.unsqueeze(-1).expand(self.num_heads, -1, -1)
        state_action_values = self.policy(state).gather(2, actions).squeeze()
        state_action_values = torch.sum(alpha * state_action_values, dim=0)

        # Get V(s') for the new states w/ mask for final state
        with torch.no_grad():
            all_next_states = self.target(new_state)
            all_next_states = torch.sum(all_next_states * alpha.unsqueeze(-1).expand(-1, -1, self.action_space),
                                        dim=0)
            next_state_values, _ = torch.max(all_next_states, dim=1)
            next_state_values[done] = 0

        # Get expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward

        # Compute loss
        loss = self.loss(state_action_values, expected_state_action_values)

        # Optimize w/ Clipping
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        self.loss_total += loss.item()

        # Update target every n steps
        if self.batches_done % self.target_update_steps == 0:
            self.target.load_state_dict(self.policy.state_dict())

        self.batches_done += 1

    def print_model(self):
        torchsummary.summary(self.policy, input_size=(self.observation_space,))

    def get_action_prob(self, state):
        self.target.eval()
        with torch.no_grad():
            state.to(self.device)
            avg_q_values = torch.mean(self.target(state), dim=0)
            return avg_q_values.softmax(dim=1)

    def get_total_loss(self):
        temp = self.loss_total
        self.loss_total = 0
        return temp

    def get_batches_done(self):
        return self.batches_done

    def set_cpu(self):
        self.policy.to(torch.device('cpu'))
        self.target.to(torch.device('cpu'))

    def set_gpu(self):
        self.policy.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.target.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    def dump_policy(self, state_dim, action_dim):
        self.target.eval()
        if not os.path.exists('policy'):
            os.mkdir('policy')
        filename = "policy" + str(len(os.listdir('policy')) + 1)

        with open(filename,'w') as file:
            with torch.no_grad():
                for i in range(state_dim):
                    state = torch.tensor((i, i)).float()
                    action_q_values = torch.mean(self.target(one_hot(state)), dim=0)
                    action_prob = action_q_values.softmax(dim=1)
                    for j in range(action_dim):
                        file.write(action_prob[0][j])