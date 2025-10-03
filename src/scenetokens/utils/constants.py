from enum import Enum


MILLION = 1e6
SMALL_EPSILON = 1e-10
BIG_EPSILON = 1e10
POSITION_DIMS = [2, 3]
MIN_VALID_POINTS = 2
# NOTE: borrowed from UniTraj: https://arxiv.org/pdf/2403.15098. They don't explain why these ranges were selected.
KALMAN_DIFFICULTY = {"easy": [0, 30], "medium": [30, 60], "hard": [60, 9999999]}


class DataSplits(Enum):
    TRAINING = 0
    VALIDATION = 1
    TESTING = 2


class TrajectoryType(Enum):
    STATIONARY = 0
    STRAIGHT = 1
    STRAIGHT_RIGHT = 2
    STRAIGHT_LEFT = 3
    RIGHT_U_TURN = 4
    RIGHT_TURN = 5
    LEFT_U_TURN = 6
    LEFT_TURN = 7


class AgentBehaviorType(Enum):
    NON_CAUSAL = 0
    CAUSAL = 1


class CausalOutputType(Enum):
    GROUND_TRUTH = 0
    PREDICTION = 1
