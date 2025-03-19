from enum import IntEnum
from types import SimpleNamespace
from typing import Optional, Tuple
import torch

from util import distance

ACTION_SIZE = 6


class Action(IntEnum):
    SOUTH = 0
    NORTH = 1
    EAST = 2
    WEST = 3
    PICKUP = 4
    DROPOFF = 5

    def to_onehot(self):
        onehot = [0] * (len(self.__class__))
        onehot[self.value] = 1
        return onehot

    @classmethod
    def empty_onehot(cls):
        return [0] * (len(cls))


class Direction(IntEnum):
    HERE = 0
    LEFT = 1
    RIGHT = 2
    UP = 3
    DOWN = 4
    FIRST_Q = 5
    SECOND_Q = 6
    THIRD_Q = 7
    FOURTH_Q = 8

    def get(src, dest):
        def cmp(a, b):
            if a == b:
                return 0
            if a < b:
                return 1
            else:
                return -1
        mapping = {
            (0, 0): Direction.HERE,
            (1, 0): Direction.RIGHT,
            (0, 1): Direction.UP,
            (-1, 0): Direction.LEFT,
            (0, -1): Direction.DOWN,
            (1, 1): Direction.FIRST_Q,
            (-1, 1): Direction.SECOND_Q,
            (-1, -1): Direction.THIRD_Q,
            (1, -1): Direction.FOURTH_Q
        }
        return mapping[(cmp(src[0], dest[0]), cmp(src[1], dest[1]))]

    def to_onehot(self):
        onehot = [0] * (len(self.__class__))
        onehot[self.value] = 1
        return onehot


class StationType(IntEnum):
    UNKNOWN = 0
    NONE = 1
    PASSENGER = 2
    DESTINATION = 3

    def to_onehot(self):
        onehot = torch.zeros(len(self.__class__))
        onehot[self.value] = 1
        return onehot


class StateManager:
    def __init__(self):
        self.reset()

    def reset(self):
        self.picked_up = False
        self.station_type = [StationType.UNKNOWN] * 4
        self.passenger_pos = None
        self.prev_action: Optional[Action] = None
        self.visited_states = set()

    def get_state(self, obs) -> Tuple[Tuple, SimpleNamespace]:
        # Read observation
        self.taxi_pos = (obs[0], obs[1])
        self.stations = [
            (obs[2], obs[3]),
            (obs[4], obs[5]),
            (obs[6], obs[7]),
            (obs[8], obs[9]),
        ]
        self.obstacles = (obs[10], obs[11], obs[12], obs[13])
        self.near_passenger = obs[14]
        self.near_destination = obs[15]

        # update station information
        if self.passenger_pos is None and \
                self.near_passenger and \
                self.taxi_pos in self.stations:
            self.passenger_pos = self.taxi_pos
            self.station_type[
                self.stations.index(self.taxi_pos)] = StationType.PASSENGER
        if self.taxi_pos in self.stations:
            i = self.stations.index(self.taxi_pos)
            if self.near_destination:
                self.station_type[i] = StationType.DESTINATION
            elif self.station_type[i] == StationType.UNKNOWN:
                self.station_type[i] = StationType.NONE

        # inference type of unknown stations
        known_station_count = sum(x != 0 for x in self.station_type)
        if known_station_count == 3:
            i = self.station_type.index(0)
            self.station_type[i] = StationType(7 - sum(self.station_type))
            if self.station_type[i] == StationType.PASSENGER:
                self.passenger_pos = self.stations[i]

        elif known_station_count == 2 and \
                StationType.PASSENGER in self.station_type and \
                StationType.DESTINATION in self.station_type:
            for i, t in enumerate(self.station_type):
                if t == StationType.UNKNOWN:
                    self.station_type[i] = StationType.NONE

        # can pickup/dropoff
        self.can_pickup = int(not self.picked_up and
                              self.taxi_pos == self.passenger_pos)
        self.can_dropoff = int(self.picked_up and
                               self.near_destination and
                               self.taxi_pos in self.stations)

        if self.picked_up:
            self.passenger_pos = self.taxi_pos

        # target station
        if not self.picked_up:
            if self.passenger_pos is not None:
                self.target = self.passenger_pos
            else:
                # find the nearest unknown station
                self.target = min(
                    (s for s, t in zip(self.stations, self.station_type)
                     if t == StationType.UNKNOWN),
                    key=lambda x: distance(self.taxi_pos, x)
                )
        else:
            if StationType.DESTINATION in self.station_type:
                self.target = self.stations[self.station_type.index(
                    StationType.DESTINATION)]
            else:
                # find the nearest unknown station
                self.target = min(
                    (s for s, t in zip(self.stations, self.station_type)
                     if t == StationType.UNKNOWN),
                    key=lambda x: distance(self.taxi_pos, x)
                )

        up = int(self.target[1] > self.taxi_pos[1])
        down = int(self.target[1] < self.taxi_pos[1])
        left = int(self.target[0] < self.taxi_pos[0])
        right = int(self.target[0] > self.taxi_pos[0])

        self.visited_states.add((self.taxi_pos, self.picked_up))

        x, y = self.taxi_pos
        visited = [int(((x + dx, y + dy), self.picked_up)
                       in self.visited_states)
                   for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1)]]

        state = (
            *self.obstacles,
            self.can_pickup,
            self.can_dropoff,
            up,
            down,
            left,
            right,
        )
        info = SimpleNamespace(**vars(self))

        return state, info

    def update_action(self, action: int):
        self.prev_action = Action(action)

        if action == Action.PICKUP and self.taxi_pos == self.passenger_pos:
            self.picked_up = True
        elif action == Action.DROPOFF:
            self.picked_up = False

    @property
    def state_shape(self):
        return (*[2, 2, 2, 2], 2, 2, *[2, 2, 2, 2])

    @property
    def state_size(self):
        return len(self.state_shape)


def reward_shaping(reward, info: SimpleNamespace, next_info: SimpleNamespace) -> float:
    if reward == 50 - 0.1:  # finish
        reward = 50
    elif reward == -10.1:  # Incorrect PICKUP or DROPOFF
        reward = -10
    elif reward == -5.1:  # hit an obstacle
        reward = -10
    elif reward == -0.1:  # move
        reward = -0.1

    if not info.picked_up and next_info.picked_up:  # pickup
        reward += 10
    elif info.picked_up and not next_info.picked_up and not info.can_dropoff:
        reward -= 15
    if sum(x == StationType.UNKNOWN for x in info.station_type) > \
            sum(x == StationType.UNKNOWN for x in next_info.station_type):
        # visit new station
        reward += 10
    if info.target == next_info.target:
        reward += 0.1 * (distance(info.taxi_pos, info.target) -
                         distance(next_info.taxi_pos, next_info.target))

    return reward
