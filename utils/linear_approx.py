import numpy as np
from typing import Tuple


def get_linear_approx_coefficients(x, y, step=None) -> Tuple[np.array, np.array, float, float, float]:
    """
    y = ax + b
    :param x:
    :param y:
    :param step:
    :return: approx_x, approx_y, a, b, x_min
    """
    coefficients = np.polyfit(x, y, 1)
    a, b = coefficients
    x_min = -b / a

    if step is None:
        step = (max(x) - min(x)) / 1000
    approx_x = np.arange(min(x), max(x) + step, step)
    approx_y = a * approx_x + b

    return approx_x, approx_y, a, b, x_min


a = 1.50843142e-03  # из kalib.py (p)
b = 1.83585587e-04  # из kalib.py (p)
c = 4.76118152e-07  # из kalib.py (p)


if __name__ == '__main__':
    # 3.903 - значение k при температуре T = 25 C
    # 4.771 - значение k при температуре T = 39.69 C

    _, _, a, b, _ = get_linear_approx_coefficients(x=[25, 40], y=[3.903, 4.771])
    k1 = a*30 + b
    k2 = a*59.3 + b
    print(f"k1 = {k1 = }")
    print(f"k2 = {k2 = }")
