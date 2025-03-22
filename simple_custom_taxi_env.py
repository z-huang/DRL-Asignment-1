from collections import defaultdict
from typing import Literal, Tuple
import numpy as np
import importlib.util
import time
from IPython.display import clear_output
import random


class SimpleTaxiEnv():
    def __init__(
        self,
        grid_size: int = 5,
        fuel_limit: int = 50,
        difficulty: Literal['easy', 'normal', 'hard'] = 'normal'
    ):
        """
        Custom Taxi environment supporting different grid sizes.
        """
        self.grid_size = grid_size
        self.fuel_limit = fuel_limit
        self.difficulty = difficulty

    def reset(self):
        """Reset the environment, ensuring Taxi, passenger, and destination are not overlapping obstacles"""

        obstacle_num = {
            'easy': defaultdict(lambda: 0),
            'normal': {
                10: 10,
                9: 8,
                8: 6,
                7: 4,
                6: 3,
                5: 2
            },
            'hard': {
                10: 20,
                9: 12,
                8: 8,
                7: 6,
                6: 5,
                5: 5
            }
        }
        self.n_obstacle = obstacle_num[self.difficulty][self.grid_size]

        while True:
            available_positions = [
                (x, y)
                for x in range(self.grid_size)
                for y in range(self.grid_size)
            ]

            self.stations = []
            for _ in range(4):
                x, y = random.choice(available_positions)
                self.stations.append((x, y))
                for dx, dy in [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1)]:
                    if (x + dx, y + dy) in available_positions:
                        available_positions.remove((x + dx, y + dy))

            # self.stations = [(0, 0), (0, self.grid_size - 1), (self.grid_size - 1, 0), (self.grid_size - 1, self.grid_size - 1)]

            self.current_fuel = self.fuel_limit
            self.passenger_picked_up = False

            available_positions = [
                (x, y)
                for x in range(self.grid_size)
                for y in range(self.grid_size)
                if (x, y) not in self.stations
            ]

            self.taxi_pos = random.choice(available_positions)

            self.passenger_loc = random.choice(self.stations)

            self.destination = random.choice([s for s in self.stations
                                              if s != self.passenger_loc])

            available_positions = [
                (x, y)
                for x in range(self.grid_size)
                for y in range(self.grid_size)
                if (x, y) not in self.stations and (x, y) != self.taxi_pos
            ]

            self.obstacles = set(random.sample(
                available_positions, self.n_obstacle))

            if self.is_valid():
                break

        return self.get_state(), {}

    def is_valid(self):
        return self.is_reachable(self.taxi_pos, self.passenger_loc) and \
            self.is_reachable(self.passenger_loc, self.destination)

    def is_reachable(self, p: Tuple[int, int], q: Tuple[int, int]):
        if p == q:
            return True

        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        queue = [p]
        visited = set()

        while queue:
            x, y = queue.pop(0)
            if (x, y) == q:
                return True

            if (x, y) in visited:
                continue
            visited.add((x, y))

            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.grid_size and
                    0 <= ny < self.grid_size and
                        (nx, ny) not in self.obstacles):
                    queue.append((nx, ny))

        return False

    def step(self, action):
        """Perform an action and update the environment state."""
        taxi_row, taxi_col = self.taxi_pos
        next_row, next_col = taxi_row, taxi_col
        reward = 0
        if action == 0:  # Move Down
            next_row += 1
        elif action == 1:  # Move Up
            next_row -= 1
        elif action == 2:  # Move Right
            next_col += 1
        elif action == 3:  # Move Left
            next_col -= 1

        if action in [0, 1, 2, 3]:  # Only movement actions should be checked
            if (next_row, next_col) in self.obstacles or not (0 <= next_row < self.grid_size and 0 <= next_col < self.grid_size):
                reward -= 5
            else:
                self.taxi_pos = (next_row, next_col)
                if self.passenger_picked_up:
                    self.passenger_loc = self.taxi_pos
        else:
            if action == 4:  # PICKUP
                if self.taxi_pos == self.passenger_loc:
                    self.passenger_picked_up = True
                    self.passenger_loc = self.taxi_pos
                else:
                    reward = -10
            elif action == 5:  # DROPOFF
                if self.passenger_picked_up:
                    if self.taxi_pos == self.destination:
                        reward += 50
                        return self.get_state(), reward - 0.1, True, {}
                    else:
                        reward -= 10
                    self.passenger_picked_up = False
                    self.passenger_loc = self.taxi_pos
                else:
                    reward -= 10

        reward -= 0.1

        self.current_fuel -= 1
        if self.current_fuel <= 0:
            return self.get_state(), reward, True, {}

        return self.get_state(), reward, False, {}

    def get_state(self):
        """Return the current environment state."""
        taxi_row, taxi_col = self.taxi_pos
        passenger_row, passenger_col = self.passenger_loc
        destination_row, destination_col = self.destination

        obstacle_north = int(taxi_row == 0 or (
            taxi_row-1, taxi_col) in self.obstacles)
        obstacle_south = int(taxi_row == self.grid_size -
                             1 or (taxi_row+1, taxi_col) in self.obstacles)
        obstacle_east = int(taxi_col == self.grid_size -
                            1 or (taxi_row, taxi_col+1) in self.obstacles)
        obstacle_west = int(taxi_col == 0 or (
            taxi_row, taxi_col-1) in self.obstacles)

        passenger_loc_north = int(
            (taxi_row - 1, taxi_col) == self.passenger_loc)
        passenger_loc_south = int(
            (taxi_row + 1, taxi_col) == self.passenger_loc)
        passenger_loc_east = int(
            (taxi_row, taxi_col + 1) == self.passenger_loc)
        passenger_loc_west = int(
            (taxi_row, taxi_col - 1) == self.passenger_loc)
        passenger_loc_middle = int((taxi_row, taxi_col) == self.passenger_loc)
        passenger_look = passenger_loc_north or passenger_loc_south or passenger_loc_east or passenger_loc_west or passenger_loc_middle

        destination_loc_north = int(
            (taxi_row - 1, taxi_col) == self.destination)
        destination_loc_south = int(
            (taxi_row + 1, taxi_col) == self.destination)
        destination_loc_east = int(
            (taxi_row, taxi_col + 1) == self.destination)
        destination_loc_west = int(
            (taxi_row, taxi_col - 1) == self.destination)
        destination_loc_middle = int((taxi_row, taxi_col) == self.destination)
        destination_look = destination_loc_north or destination_loc_south or destination_loc_east or destination_loc_west or destination_loc_middle

        state = (taxi_row, taxi_col, self.stations[0][0], self.stations[0][1], self.stations[1][0], self.stations[1][1], self.stations[2][0], self.stations[2]
                 [1], self.stations[3][0], self.stations[3][1], obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look)
        return state

    def render_env(self, taxi_pos, action=None, step=None, fuel=None):
        clear_output(wait=True)

        grid = [['.'] * self.grid_size for _ in range(self.grid_size)]

        for (x, y), label in zip(self.stations, ['R', 'G', 'Y', 'B']):
            grid[x][y] = label

        # Place passenger
        py, px = self.passenger_loc
        if 0 <= px < self.grid_size and 0 <= py < self.grid_size:
            grid[py][px] = 'P'

        # Place destination
        dy, dx = self.destination
        if 0 <= dx < self.grid_size and 0 <= dy < self.grid_size:
            grid[dy][dx] = 'D'

        for row, col in self.obstacles:
            grid[row][col] = "X"

        # Place taxi
        ty, tx = taxi_pos
        if 0 <= tx < self.grid_size and 0 <= ty < self.grid_size:
            grid[ty][tx] = 'T'

        # Print step info
        print(f"\nStep: {step}")
        print(f"Taxi Position: ({tx}, {ty})")
        # print(f"Passenger Position: ({px}, {py}) {'(In Taxi)' if (px, py) == (tx, ty) else ''}")
        # print(f"Destination: ({dx}, {dy})")
        print(f"Fuel Left: {fuel}")
        print(f"Last Action: {self.get_action_name(action)}\n")

        # Print grid
        for row in grid:
            print(" ".join(row))
        print("\n")

    def get_action_name(self, action):
        """Returns a human-readable action name."""
        actions = ["Move South", "Move North", "Move East",
                   "Move West", "Pick Up", "Drop Off"]
        return actions[action] if action is not None else "None"


def run_agent(agent_file, render=False):
    spec = importlib.util.spec_from_file_location("student_agent", agent_file)
    student_agent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(student_agent)

    env = SimpleTaxiEnv(
        fuel_limit=5000,
        grid_size=np.random.randint(5, 11),
    )
    obs, _ = env.reset()
    total_reward = 0
    done = False
    step_count = 0
    stations = [(0, 0), (0, 4), (4, 0), (4, 4)]

    taxi_row, taxi_col, _, _, _, _, _, _, _, _, obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look = obs

    if render:
        env.render_env((taxi_row, taxi_col),
                       action=None, step=step_count, fuel=env.current_fuel)
        time.sleep(0.5)
    while not done:

        action = student_agent.get_action(obs)

        obs, reward, done, _ = env.step(action)
        print('obs=', obs)
        total_reward += reward
        step_count += 1

        taxi_row, taxi_col, _, _, _, _, _, _, _, _, obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look = obs
        print(f'reward: {reward}')
        if render:
            env.render_env((taxi_row, taxi_col),
                           action=action, step=step_count, fuel=env.current_fuel)

    print(f"Agent Finished in {step_count} steps, Score: {total_reward}")
    return total_reward


if __name__ == "__main__":
    agent_score = run_agent("student_agent.py", render=True)
    print(f"Final Score: {agent_score}")
