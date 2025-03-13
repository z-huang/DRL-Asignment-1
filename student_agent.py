import numpy as np
import pickle
from util import StateManager, ACTION_SIZE

with open("q_table.pkl", "rb") as f:
    q_table = pickle.load(f)

state_manager = StateManager()
manual = 0

def get_action(obs):
    if manual:
        state = state_manager.get_state(obs)
        print(state)
        action = int(input())
        state_manager.update_action(action)
        return action

    state = state_manager.get_state(obs)

    if state in q_table:
        action = np.argmax(q_table[state])
    else:
        action = np.random.randint(ACTION_SIZE)

    state_manager.update_action(action)

    return action
