import pickle
from abc import ABC, abstractmethod
import numpy as np
from pg import PolicyNetwork
import torch
from state import ACTION_SIZE, StateManager


class Agent(ABC):
    @abstractmethod
    def get_action(self):
        ...


class HumanAgent(Agent):
    def __init__(self):
        self.state_manager = StateManager()

    def get_action(self, obs):
        state, info = self.state_manager.get_state(obs)
        print(state, info)
        action = int(input())
        self.state_manager.update_action(action)
        return action


class QTableAgent(Agent):
    def __init__(self, path):
        self.state_manager = StateManager()
        with open(path, "rb") as f:
            self.q_table = pickle.load(f)

    def get_action(self, obs):
        state = self.state_manager.get_state(obs)

        if state in self.q_table:
            action = np.argmax(self.q_table[state])
        else:
            action = np.random.randint(ACTION_SIZE)

        self.state_manager.update_action(action)

        return action


class PolicyGradientAgent(Agent):
    def __init__(self, path):
        self.state_manager = StateManager()
        self.table = PolicyNetwork(self.state_manager.state_size, ACTION_SIZE)
        self.table.load_state_dict(torch.load(path, map_location='cpu'))

    @torch.no_grad()
    def get_action(self, obs):
        state, _ = self.state_manager.get_state(obs)
        action = self.table.get_action(state)
        self.state_manager.update_action(action)
        return action
