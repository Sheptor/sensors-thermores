import numpy as np
from typing import List, Callable, Union


def half_division_method(x: Union[List[float], np.array], ytarget: float, accuracy, func: Callable, *args) -> float:
    y_list = list(func(x, *args))
    if y_list == sorted(y_list):
        pass
    elif y_list == sorted(y_list, reverse=True):
        x = x[::-1]
        y_list = y_list[::-1]
    else:
        raise ValueError("Функция должна быть монотонной!")

    if not y_list[0] <= ytarget <= y_list[-1]:
        raise ValueError(
            f"Функция должна быть монотонной и целевое значение должно входить в диапазон! {y_list[0]} <= {ytarget} <= {y_list[-1]}"
        )

    x_start = x[0]
    x_end = x[-1]
    y_start = func(x_start, *args)
    y_end = func(x_end, *args)
    try:
        y_start = y_start[0]
        y_end = y_end[0]
    except IndexError:
        pass
    middle_x = (x_start + x_end) / 2
    error = float("inf")

    while error > accuracy:
        middle_x = (x_start + x_end) / 2
        middle_y = func(middle_x, *args)
        # print(f"{middle_x = :.3f}, {middle_y:.3f}")
        error = abs(ytarget - middle_y)
        if ytarget < middle_y:
            x_end = middle_x
        elif ytarget > middle_y:
            x_start = middle_x
        else:
            break

    return middle_x
