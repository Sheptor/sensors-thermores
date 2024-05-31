import numpy as np
from utils.half_division import half_division_method


def steinhart_hart_model(r, a, b, c):
    return 1 / (a + b * np.log(r) + c * (np.log(r)) ** 3)


if __name__ == '__main__':
    a = 1.50843142e-03  # из kalib.py (p)
    b = 1.83585587e-04  # из kalib.py (p)
    c = 4.76118152e-07  # из kalib.py (p)

    # r1 = np.arange(4772.4, 4773, 0.1)  # -> T = 25.25
    # r1 = np.arange(4000, 5000, 0.1)  # -> T = 25.25

    res = half_division_method([1, 5000], 30+273+0.1, 0.001, steinhart_hart_model, a, b, c)
    I01 = ((4.2 / 1000) * 0.01 / res) ** (1 / 2)
    print(f"{res = }, при T + 0.1 = {steinhart_hart_model(res, a, b, c)-273} C")
    print(f"{I01 = }")

    res = half_division_method([1, 5000], 60+273+0.1, 0.001, steinhart_hart_model, a, b, c)
    I02 = ((5.9 / 1000) * 0.01 / res) ** (1 / 2)
    print(f"{res = }, при T + 0.1 = {steinhart_hart_model(res, a, b, c) - 273} C")
    print(f"{I02 = }")
