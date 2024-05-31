import numpy as np
from utils.find_nearest_index import get_nearest_index


def get_delta(original_x: np.array, original_y: np.array, approxed_x: np.array, approxed_y: np.array) -> np.array:
    delta_list = []
    for i_x, i_y in zip(original_x, original_y):
        i = get_nearest_index(array=approxed_x, value=i_x)
        delta_list.append(i_y - approxed_y[i])
        print(i_y - 273, approxed_y[i] - 273)
    return np.array(delta_list)


if __name__ == '__main__':
    res = get_delta(
        original_x=[1, 2, 3], original_y=[1, 2, 3], approxed_x=np.arange(1, 3.1, 0.1), approxed_y=np.arange(1, 3.1, 0.1)
    )
    print(f"{res = }")
