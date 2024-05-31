import numpy as np


def get_nearest_index(array: np.array, value: float) -> int:
    array = np.asarray(array)
    index = (np.abs(array - value)).argmin()
    return index
