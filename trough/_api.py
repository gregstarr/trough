import dataclasses
import numpy as np


@dataclasses.dataclass
class TroughData:
    time: np.ndarray = np.zeros(10)
    tec: np.ndarray = np.zeros(10)
    trough: np.ndarray = np.zeros(10)


def get_data(tec_dir, trough_dir):
    data = TroughData()
    return data
