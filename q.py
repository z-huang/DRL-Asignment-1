from collections import defaultdict
import pickle
from typing import Mapping, Tuple
import numpy as np
from tqdm import tqdm
from simple_custom_taxi_env import SimpleTaxiEnv
import matplotlib.pyplot as plt
from state import StateManager, ACTION_SIZE, reward_shaping


WIN_SIZE = 1000

def q_learning(
    episodes=1000000,
    alpha=0.01,
    gamma=0.999,
    epsilon_start=1.0,
    epsilon_end=0.01,
    decay_episodes=400000,
) -> Mapping[Tuple, np.ndarray]:
    state_manager = StateManager()
    env = SimpleTaxiEnv(fuel_limit=100)

    q_table = defaultdict(lambda: np.zeros(ACTION_SIZE))

    epsilon = epsilon_start
    rewards_per_episode = []
    success_per_episode = []

    with tqdm(range(episodes)) as pbar:
        for episode in pbar:
            obs, _ = env.reset()
            state_manager.reset()
            
            state, info = state_manager.get_state(obs)
            total_reward = 0
            done = success = False

            while not done:
                if np.random.rand() < epsilon:
                    action = np.random.randint(ACTION_SIZE)
                else:
                    action = np.argmax(q_table[state])

                obs, reward, done, _ = env.step(action)
                state_manager.update_action(action)
                next_state, next_info = state_manager.get_state(obs)

                if done and reward == 49.9:
                    success = True
                reward = reward_shaping(reward, info, next_info)

                # update Q table
                best_next_action = q_table[next_state].max()
                q_table[state][action] += alpha * (
                    reward + gamma * best_next_action -
                    q_table[state][action])
                
                total_reward += reward
                state = next_state
                info = next_info

            rewards_per_episode.append(total_reward)
            success_per_episode.append(success)
            epsilon = max(epsilon_end,
                          epsilon_start - episode * (epsilon_start - epsilon_end) / decay_episodes)
            # epsilon = max(epsilon_end, epsilon * decay_rate)

            pbar.set_postfix({
                'avg reward': np.mean(rewards_per_episode[-WIN_SIZE:]).item(),
                'win rate': np.mean(success_per_episode[-WIN_SIZE:]).item(),
                'epsilon': epsilon,
            })

    moving_avg = np.convolve(rewards_per_episode,
                             np.ones(WIN_SIZE) / WIN_SIZE,
                             mode='valid')

    plt.plot(rewards_per_episode, alpha=0.5)
    plt.plot(range(WIN_SIZE - 1, len(rewards_per_episode)),
             moving_avg,
             color='red')
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Training Progress")
    plt.savefig('graph/q_learning.png')

    return dict(q_table)


if __name__ == '__main__':
    q_table = q_learning()

    with open('q_table.pkl', 'wb') as f:
        pickle.dump(q_table, f)
