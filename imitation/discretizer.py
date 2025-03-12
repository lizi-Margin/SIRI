import numpy as np
import gymnasium.spaces as spaces
from typing import Union, List
from siri.utils.logger import lprint
from .filter import mouse_filter, mouse_pos_filter

class wasd_Discretizer():
    """
        None
        w, a, s, d
        wa, wd, sa, sd
    """
    def __init__(self):
        self.n_actions = 9
        self.coverter = np.array([
            [0, 0, 0, 0],  # None
            [1, 0, 0, 0],  # w
            [0, 1, 0, 0],  # a
            [0, 0, 1, 0],  # s
            [0, 0, 0, 1],  # d
            [1, 1, 0, 0],  # wa
            [1, 0, 0, 1],  # wd
            [0, 1, 1, 0],  # sa
            [0, 0, 1, 1]  # sd
        ])

    def index_to_action_(self, index):
        assert index >=0 and index < self.n_actions
        assert not isinstance(index, (list, np.ndarray,))
        return self.coverter[index].copy()

    def action_to_index_(self, action):
        assert isinstance(action, np.ndarray) and len(action.shape) == 1 and len(action) == 4
        ret = None
        for i in range(len(self.coverter)):
            if np.array_equal(action, self.coverter[i]):
                ret = i
        if ret is None:
            lprint(self, f"ws or ad action {str(action)}, aborted")
            ret = 0
        return ret
    
    def action_to_index(self, action):
        assert isinstance(action, np.ndarray)
        if len(action.shape) == 2:
            ret = np.zeros(action.shape[0], dtype=np.int32)
            for i in range(len(ret)):
                ret[i] = self.action_to_index_(action[i])
            return ret
        else:
            return self.action_to_index_(ret)

    def get_discrete_space(self):
        return spaces.Discrete(self.n_actions)


class SimpleDiscretizer:
    def __init__(self, box, **FILTER):
        if isinstance(box, list):
            box = np.array(box)
        assert isinstance(box, np.ndarray)
        assert np.all(np.diff(box) <= 0), f"box 必须是单向下降的, {repr(box)}"
        assert len(box.shape) == 1
        self.box = box
        self.n_actions = len(box)
        self.filter = mouse_filter(**FILTER)

    def index_to_action_(self, index: int):
        if index < 0 or index >= self.n_actions:
            raise IndexError(f"索引 {index} 超出范围 [0, {self.n_actions - 1}]")
        return self.box[index]

    def discretize_(self, continuous):
        assert isinstance(continuous, (int, float, np.float32, np.float64))
        continuous = self.filter.step(continuous)
        diffs = np.abs(self.box - continuous)
        return np.argmin(diffs).astype(np.int32)
    
    def discretize(self, continuous):
        if isinstance(continuous, (list, np.ndarray)):
            assert len(continuous.shape) == 1
            ret = np.zeros(continuous.shape, dtype=np.int32)
            for i in range(len(ret)):
                ret[i] = self.discretize_(continuous[i])
            return ret
        else:
            return self.discretize_(continuous)

class ActionDiscretizer():
    def __init__(self, raw_action_space: spaces.Box, n_bins_per_dim: Union[np.ndarray, List]):
        assert isinstance(raw_action_space, spaces.Box)
        assert len(n_bins_per_dim) == len(raw_action_space.low)

        self.raw_action_space = raw_action_space
        self.n_bins = np.array(n_bins_per_dim) - 1  # Adjust bins to include both lower and upper boundaries

        self.low = self.raw_action_space.low
        self.high = self.raw_action_space.high

        self.intervals = (self.high - self.low) / self.n_bins

        self.n_actions = np.prod(self.n_bins + 1)  # Adjust to include original number of bins
        assert isinstance(self.n_actions, np.int32)
        self.n_actions = int(self.n_actions)

    def index_to_action(self, index):
        indices = np.unravel_index(int(index), self.n_bins + 1)  # Adjust to include original number of bins
        values = self.low + np.array(indices) * self.intervals
        return values

    def get_discrete_space(self):
        return spaces.Discrete(self.n_actions)

    def get_multidiscrete_space(self):
        return spaces.MultiDiscrete(self.n_bins + 1)
