from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class DatasetBundle:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    class_names: list[str]
    metadata: dict = field(default_factory=dict)

    @property
    def input_features(self) -> int:
        return int(self.x_train.shape[-1])

    @property
    def num_classes(self) -> int:
        return len(self.class_names)
