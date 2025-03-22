import gym
import importlib.util

import imageio
import numpy as np
from env import DynamicTaxiEnv


class TaxiEnv(gym.Wrapper):
    def __init__(
        self,
        grid_size=5,
        fuel_limit=50,
        randomize_passenger=True,
        randomize_destination=True
    ):
        super().__init__(DynamicTaxiEnv(
            grid_size=grid_size,
            fuel_limit=fuel_limit,
            randomize_passenger=randomize_passenger,
            randomize_destination=randomize_destination
        ))

    def render(self, mode='human'):
        grid_size = self.unwrapped.grid_size

        grid = [['.'] * grid_size for _ in range(grid_size)]

        for label, (x, y) in self.unwrapped.station_map.items():
            grid[x][y] = label

        # Place passenger
        py, px = self.unwrapped.passenger_loc
        if 0 <= px < grid_size and 0 <= py < grid_size:
            grid[py][px] = 'P'

        # Place destination
        dy, dx = self.unwrapped.destination
        if 0 <= dx < grid_size and 0 <= dy < grid_size:
            grid[dy][dx] = 'D'

        # Place obstaclees
        for row, col in self.unwrapped.obstacles:
            grid[row][col] = "X"

        # Place taxi
        ty, tx = self.unwrapped.taxi_pos
        if 0 <= tx < grid_size and 0 <= ty < grid_size:
            grid[ty][tx] = 'T'

        if mode == "human":
            for row in grid:
                print(" ".join(row))
            print("\n")
        elif mode == "rgb_array":
            # Define color mapping
            colors = {
                '.': [255, 255, 255],  # White (empty space)
                'X': [0, 0, 0],        # Black (obstacle)
                'P': [0, 255, 0],      # Green (passenger)
                'D': [255, 0, 0],      # Red (destination)
                'T': [255, 255, 0],      # Yellow (taxi)
                'R': [0, 0, 255],
                'G': [0, 0, 255],
                'Y': [0, 0, 255],
                'B': [0, 0, 255],
            }
            img = np.zeros((grid_size, grid_size, 3), dtype=np.uint8)
            for i in range(grid_size):
                for j in range(grid_size):
                    img[i, j] = colors.get(grid[i][j], [200, 200, 200])
            return img


def get_action_name(action):
    actions = ["Move South", "Move North", "Move East",
               "Move West", "Pick Up", "Drop Off"]
    return actions[action] if action is not None else "None"


def run_agent(agent_file, render=False):
    spec = importlib.util.spec_from_file_location("student_agent", agent_file)
    student_agent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(student_agent)

    env = TaxiEnv(
        fuel_limit=5000,
        grid_size=np.random.randint(5, 11)
    )
    obs, _ = env.reset()
    total_reward = 0
    done = False
    step_count = 0
    frames = []

    if render:
        frames.append(env.render(mode="rgb_array"))

    while not done:
        action = student_agent.get_action(obs)

        obs, reward, done, _ = env.step(action)
        total_reward += reward
        step_count += 1

        print('obs:', obs)
        print('reward:', reward)
        print('action:', get_action_name(action))
        if render:
            frames.append(env.render(mode="rgb_array"))

    print(f"Agent Finished in {step_count} steps, Score: {total_reward}")

    imageio.mimsave('graph/out.gif', frames, fps=3)

    return total_reward


if __name__ == "__main__":
    agent_score = run_agent("student_agent.py", render=True)
    print(f"Final Score: {agent_score}")
