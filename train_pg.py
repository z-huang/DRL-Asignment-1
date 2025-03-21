from typing import Optional
import numpy as np
import torch
from torch import optim
from tqdm import tqdm
from model import PolicyNetwork
from simple_custom_taxi_env import SimpleTaxiEnv
import matplotlib.pyplot as plt
from state import StateManager, ACTION_SIZE, reward_shaping
import argparse

WIN_SIZE = 100


def train(
    episodes: int = 8000,
    lr: float = 0.01,
    gamma: float = 0.8,
    output_path: str = 'checkpoints/pg.pth',
    checkpoint_path: Optional[str] = None
):
    state_manager = StateManager()
    states_visited = set()
    policy = PolicyNetwork(state_manager.state_size, ACTION_SIZE)
    if checkpoint_path is not None:
        policy.load_state_dict(torch.load(checkpoint_path))
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    rewards_per_episode = []
    success_per_episode = []
    steps_per_episode = []

    with tqdm(range(episodes)) as pbar:
        for episode in pbar:
            env = SimpleTaxiEnv(
                fuel_limit=1000,
                grid_size=np.random.randint(5, 11),
                difficulty='normal' if episode < episodes / 3 else 'hard'
                # difficulty='hard'
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
                'reward': np.mean(rewards_per_episode[-WIN_SIZE:]).item(),
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
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output',
                        type=str,
                        default='checkpoints/pg.pth')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--gamma', type=float, default=0.99)
    args = parser.parse_args()
    train(
        lr=args.lr,
        gamma=args.gamma,
        output_path=args.output,
        checkpoint_path=args.checkpoint,
    )
