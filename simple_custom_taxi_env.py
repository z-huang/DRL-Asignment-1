import gym
import numpy as np
import importlib.util
import time
from IPython.display import clear_output

# This environment allows you to verify whether your program runs correctly during testing, 
# as it follows the same observation format from `env.reset()` and `env.step()`. 
# However, keep in mind that this is just a simplified environment. 
# The full specifications for the real testing environment can be found in the provided spec.
# 
# You are free to modify this file to better match the real environment and train your own agent. 
# Good luck!


class SimpleTaxiEnv(gym.Wrapper):
    def __init__(self, fuel_limit=5000):
        self.grid_size = 5
        env = gym.make("Taxi-v3", render_mode="ansi") #🚨 Taxi-v3 is **always 5x5**. If you want a different grid size, you must create a custom environment.
        super().__init__(env)
        
        self.fuel_limit = fuel_limit
        self.current_fuel = fuel_limit

        self.stations = [(0, 0), (0, self.grid_size - 1), (self.grid_size - 1, 0), (self.grid_size - 1, self.grid_size - 1)]
        self.passenger_loc = None
        self.passenger_picked_up = False  
        self.obstacles = set()  # No obstacles in simple version
        self.destination = None
    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.current_fuel = self.fuel_limit

        taxi_row, taxi_col, pass_idx, dest_idx = self.env.unwrapped.decode(obs)

        taxi_row = min(taxi_row, self.grid_size - 1)
        taxi_col = min(taxi_col, self.grid_size - 1)
        self.passenger_loc = self.stations[pass_idx] 
        self.destination = self.stations[dest_idx] 

        self.passenger_picked_up = False  

        return self.get_state(), info
    def get_state(self):
        
        taxi_row, taxi_col, _, _ = self.env.unwrapped.decode(self.env.unwrapped.s)
        passenger_x, passenger_y = self.passenger_loc
        destination_x, destination_y = self.destination
        obstacle_north = int(taxi_row == 0)
        obstacle_south = int(taxi_row == self.grid_size - 1)
        obstacle_east  = int(taxi_col == self.grid_size - 1)
        obstacle_west  = int(taxi_col == 0)
        passenger_loc_north = int( (taxi_row - 1, taxi_col) == self.passenger_loc)
        passenger_loc_south = int( (taxi_row + 1, taxi_col) == self.passenger_loc)
        passenger_loc_east  = int( (taxi_row, taxi_col + 1) == self.passenger_loc)
        passenger_loc_west  = int( (taxi_row, taxi_col - 1) == self.passenger_loc)
        passenger_loc_middle  = int( (taxi_row, taxi_col) == self.passenger_loc)
        passenger_look = passenger_loc_north or passenger_loc_south or passenger_loc_east or passenger_loc_west or passenger_loc_middle
        destination_loc_north = int( (taxi_row - 1, taxi_col) == self.destination)
        destination_loc_south = int( (taxi_row + 1, taxi_col) == self.destination)
        destination_loc_east  = int( (taxi_row, taxi_col + 1) == self.destination)
        destination_loc_west  = int( (taxi_row, taxi_col - 1) == self.destination)
        destination_loc_middle  = int( (taxi_row, taxi_col) == self.destination)
        destination_look = destination_loc_north or destination_loc_south or destination_loc_east or destination_loc_west or destination_loc_middle

        
        state = (taxi_row, taxi_col, self.stations[0][0],self.stations[0][1] ,self.stations[1][0],self.stations[1][1],self.stations[2][0],self.stations[2][1],self.stations[3][0],self.stations[3][1],obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look)
        return state
    def step(self, action):
        """Perform an action and update the environment state."""
        taxi_row, taxi_col, pass_idx, dest_idx = self.env.unwrapped.decode(self.env.unwrapped.s)

        next_row, next_col = taxi_row, taxi_col
        if action == 0 :  # Move South
            next_row += 1
        elif action == 1:  # Move North
            next_row -= 1
        elif action == 2 :  # Move East
            next_col += 1
        elif action == 3 :  # Move West
            next_col -= 1

        if action in [0, 1, 2, 3]:  
            if not (0 <= next_row < self.grid_size and 0 <= next_col < self.grid_size):
                reward = -5  
                self.current_fuel -= 1
                if self.current_fuel <= 0:
                    return self.get_state(), reward -10, True, False, {}  
                return self.get_state(), reward, False, False, {}

        taxi_row, taxi_col = next_row, next_col

        self.current_fuel -= 1  
        obs, reward, terminated, truncated, info = super().step(action)

        if reward == 20:  
            reward = 50
        elif reward == -1:  
            reward = -0.1
        elif reward == -10:  
            reward = -10

        if action == 4:  
            if pass_idx == 4:  
                self.passenger_picked_up = True  
                self.passenger_loc = (taxi_row, taxi_col)  
            else:
                self.passenger_picked_up = False  

        elif action == 5:  
            if self.passenger_picked_up:  
                if (taxi_row, taxi_col)   == self.destination:
                    reward += 50
                    return self.get_state(), reward -0.1, True, {},{}
                else:
                    reward -=10

        if self.passenger_picked_up:
            self.passenger_loc = (taxi_row, taxi_col)  
        if self.current_fuel <= 0:
            return self.get_state(), reward -10, True, False, {}  
        return self.get_state(), reward, False, truncated, info

    def render_env(self, taxi_pos,   action=None, step=None, fuel=None):
        clear_output(wait=True)

        grid = [['.'] * self.grid_size for _ in range(self.grid_size)]
        
        '''
        # Place passenger
        py, px = passenger_pos
        if 0 <= px < self.grid_size and 0 <= py < self.grid_size:
            grid[py][px] = 'P'
        '''
        
        
        grid[0][0]='R'
        grid[0][4]='G'
        grid[4][0]='Y'
        grid[4][4]='B'
        '''
        # Place destination
        dy, dx = destination_pos
        if 0 <= dx < self.grid_size and 0 <= dy < self.grid_size:
            grid[dy][dx] = 'D'
        '''
        # Place taxi
        ty, tx = taxi_pos
        if 0 <= tx < self.grid_size and 0 <= ty < self.grid_size:
            grid[ty][tx] = '🚖'

        # Print step info
        print(f"\nStep: {step}")
        print(f"Taxi Position: ({tx}, {ty})")
        #print(f"Passenger Position: ({px}, {py}) {'(In Taxi)' if (px, py) == (tx, ty) else ''}")
        #print(f"Destination: ({dx}, {dy})")
        print(f"Fuel Left: {fuel}")
        print(f"Last Action: {self.get_action_name(action)}\n")

        # Print grid
        for row in grid:
            print(" ".join(row))
        print("\n")

    def get_action_name(self, action):
        """Returns a human-readable action name."""
        actions = ["Move South", "Move North", "Move East", "Move West", "Pick Up", "Drop Off"]
        return actions[action] if action is not None else "None"


def run_agent(agent_file, env_config, render=False):
    spec = importlib.util.spec_from_file_location("student_agent", agent_file)
    student_agent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(student_agent)

    env = SimpleTaxiEnv(**env_config)
    obs, _ = env.reset()
    total_reward = 0
    done = False
    step_count = 0
    stations = [(0, 0), (0, 4), (4, 0), (4,4)]
    
    taxi_row, taxi_col, _,_,_,_,_,_,_,_,obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look, destination_look = obs

    if render:
        env.render_env((taxi_row, taxi_col),
                       action=None, step=step_count, fuel=env.current_fuel)
        time.sleep(0.5)
    while not done:
        
        
        action = student_agent.get_action(obs)

        obs, reward, done, _, _ = env.step(action)
        print('obs=',obs)
        total_reward += reward
        step_count += 1

        taxi_row, taxi_col, _,_,_,_,_,_,_,_,obstacle_north, obstacle_south, obstacle_east, obstacle_west, passenger_look,destination_look = obs

        if render:
            env.render_env((taxi_row, taxi_col),
                           action=action, step=step_count, fuel=env.current_fuel)

    print(f"Agent Finished in {step_count} steps, Score: {total_reward}")
    return total_reward

if __name__ == "__main__":
    env_config = {
        "fuel_limit": 5000
    }
    
    agent_score = run_agent("student_agent.py", env_config, render=True)
    print(f"Final Score: {agent_score}")