from enum import IntEnum

import numpy as np


ACTION_SIZE = 6


class Action(IntEnum):
    SOUTH = 0
    NORTH = 1
    EAST = 2
    WEST = 3
    PICKUP = 4
    DROPOFF = 5


class StationType(IntEnum):
    UNK = 0
    NONE = 1
    PASSENGER = 2
    DESTINATION = 3


class StateManager:
    def __init__(self):
        self.reset()

    def reset(self):
        self.cur_state = None
        self.picked_up = False
        self.station_visited = [0, 0, 0, 0]
        self.station_type = [0, 0, 0, 0]

    def get_state(self, obs):
        self.taxi_pos = (obs[0], obs[1])
        self.stations = [
            (obs[2], obs[3]),
            (obs[4], obs[5]),
            (obs[6], obs[7]),
            (obs[8], obs[9]),
        ]
        self.obstacle = (obs[10], obs[11], obs[12], obs[13])
        self.near_passenger = obs[14]
        self.near_destination = obs[15]
        self.is_passenger = self.near_passenger and self.taxi_pos in self.stations
        self.is_destination = self.near_destination and self.taxi_pos in self.stations

        nearest_station = min(self.stations,
                              key=lambda s: (abs(s[0] - self.taxi_pos[0]) +
                                             abs(s[1] - self.taxi_pos[1])))
        
        if distance(self.taxi_pos, nearest_station) <= 1:
            i = self.stations.index(nearest_station)
            if self.picked_up:
                if self.near_destination:
                    self.station_type[i] = StationType.DESTINATION
                elif self.station_type[i] == StationType.UNK:
                    self.station_type[i] = StationType.NONE
            else:
                if self.near_passenger:
                    self.station_type[i] = StationType.PASSENGER
                elif self.near_destination:
                    self.station_type[i] = StationType.DESTINATION
                else:
                    self.station_type[i] = StationType.NONE

        unknown_station_count = sum(x != 0 for x in self.station_type)
        if unknown_station_count == 3:
            i = self.station_type.index(0)
            self.station_type[i] = StationType(7 - sum(self.station_type))
        elif unknown_station_count == 2 and \
                StationType.PASSENGER in self.station_type and \
                StationType.DESTINATION in self.station_type:
            for i, t in enumerate(self.station_type):
                if t == StationType.UNK:
                    self.station_type[i] = StationType.NONE

        self.cur_state = (
            self.taxi_pos,
            self.picked_up,
            tuple(self.station_type)
        )

        return self.cur_state

    def update_action(self, action):
        if action == Action.PICKUP and \
                self.taxi_pos in self.stations and \
                self.near_passenger:
            self.picked_up = True


def distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def reward_shaping(reward, state, next_state, obs) -> float:
    if reward == 50 - 0.1:  # finish
        reward = 50
    elif reward == -10.1 or reward == -5.1:  # Incorrect PICKUP or DROPOFF or game over
        reward = -5

    if not state[1] and next_state[1]:  # pickup
        reward += 10
    if sum(x == 0 for x in state[2]) > sum(x == 0 for x in next_state[2]):
        reward += 5

    return reward


def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Numeric stability
    return exp_x / exp_x.sum()
