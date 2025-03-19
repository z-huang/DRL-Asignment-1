from typing import Tuple
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from tqdm import tqdm
from simple_custom_taxi_env import SimpleTaxiEnv
import matplotlib.pyplot as plt
from state import StateManager, ACTION_SIZE, reward_shaping
from taxi import TaxiEnv

WIN_SIZE = 100


class PolicyTable(nn.Module):
    def __init__(self, state_size, num_actions, lr=0.1):
        super().__init__()
        self.table = nn.Parameter(
            torch.zeros(state_size + (num_actions,)))
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, state):
        state = tuple(state.long().tolist())
        logits = self.table[state]
        probs = F.softmax(logits, dim=-1)
        return probs

    @torch.no_grad()
    def get_action(self, state: Tuple):
        probs = self(state)
        action = torch.multinomial(probs, 1).item()
        return action

    def update(self, state: Tuple, action: int, reward: float):
        logits = self.table[state]
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -log_probs[action] * reward

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 16)
        self.fc2 = nn.Linear(16, action_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state):
        x = torch.tensor(state, dtype=torch.float32)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)

    def get_action(self, state):
        action_probs = self(torch.tensor(state))
        action_dist = torch.distributions.Categorical(action_probs)
        return action_dist.sample().item()


def train(
    episodes: int = 8000,
    lr: float = 0.01,
    gamma: float = 0.99,
    output_path: str = 'checkpoints/pg.pth'
):
    state_manager = StateManager()
    states_visited = set()
    policy = PolicyNetwork(state_manager.state_size, ACTION_SIZE)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    rewards_per_episode = []
    success_per_episode = []
    steps_per_episode = []

    with tqdm(range(episodes)) as pbar:
        for episode in pbar:
            env = SimpleTaxiEnv(
                fuel_limit=1000,
                grid_size=np.random.randint(5, 11)
            )

            log_probs = []
            rewards = []
            total_reward = 0

            obs, _ = env.reset()
            state_manager.reset()
            state, info = state_manager.get_state(obs)
            states_visited.add(state)

            done, success = False, False
            step = 0
            while not done:
                action_probs = policy(state)
                action_dist = torch.distributions.Categorical(action_probs)
                action = action_dist.sample()
                log_probs.append(action_dist.log_prob(action))

                # policy.update_action(action)
                state_manager.update_action(action.item())

                obs, reward, done, _ = env.step(action.item())
                next_state, next_info = state_manager.get_state(obs)
                states_visited.add(next_state)

                if done and reward == 49.9:
                    success = True

                reward = reward_shaping(reward, info, next_info)

                total_reward += reward
                rewards.append(reward)
                state = next_state
                info = next_info
                step += 1

            rewards_per_episode.append(total_reward)
            success_per_episode.append(success)
            steps_per_episode.append(step)

            returns = []
            R = 0
            for r in reversed(rewards):
                R = r + gamma * R
                returns.insert(0, R)

            returns = torch.tensor(returns)
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)

            policy_loss = []
            for log_prob, R in zip(log_probs, returns):
                policy_loss.append(-log_prob * R)
            loss = torch.stack(policy_loss).sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({
                'avg reward': np.mean(rewards_per_episode[-WIN_SIZE:]).item(),
                'win rate': np.mean(success_per_episode[-WIN_SIZE:]).item(),
                'step': np.mean(steps_per_episode[-WIN_SIZE:]).item(),
                'state': len(states_visited)
            })

            if (episode + 1) % WIN_SIZE == 0:
                moving_avg = np.convolve(rewards_per_episode,
                                         np.ones(WIN_SIZE) / WIN_SIZE,
                                         mode='valid')
                plt.clf()
                plt.plot(rewards_per_episode, alpha=0.5)
                plt.plot(range(WIN_SIZE - 1, len(rewards_per_episode)),
                         moving_avg, color='red')
                plt.xlabel("Episodes")
                plt.ylabel("Total Reward")
                plt.title("Training Progress")
                plt.grid()
                plt.savefig('graph/pg.png')

                torch.save(policy.state_dict(), output_path)


if __name__ == '__main__':
    train()
