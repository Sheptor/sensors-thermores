import numpy as np
import os
import matplotlib.pyplot as plt
import config.config  # noqa
from scipy.optimize import curve_fit
from utils.approx_difference import get_delta
from utils.streinhart_model import steinhart_hart_model


T_LIST = np.array([25, 40, 55, 70, 85, 100])  # C
T1_LIST = np.array([24, 39.03, 53.78, 68.47, 83.32, 97.63])  # C
T1_KELV_LIST = T1_LIST + 273  # K
U_LIST = np.array([0.25, 0.1417, 0.0814, 0.0488, 0.0301, 0.0189])  # V
I0 = 5e-5  # A
R_LIST = U_LIST / I0  # Om


if __name__ == '__main__':
    if not os.path.exists("images"):
        os.makedirs("images")
    print(f"{R_LIST = }")
    r_t1 = np.vstack((R_LIST, T1_KELV_LIST))
    p = curve_fit(steinhart_hart_model, xdata=R_LIST, ydata=T1_KELV_LIST, p0=[0.0, 0.0005001, -0.0000015], method="lm")
    print(f"{p[0]=} (Коэффициенты A, B, C)")

    new_r = np.arange(min(R_LIST), max(R_LIST), 0.01)
    new_t = steinhart_hart_model(r=new_r, a=p[0][0], b=p[0][1], c=p[0][2])

    if "" == "":
        plt.figure(figsize=(16, 9))
        plt.xlabel("Температура, °C")
        plt.ylabel("Сопротивление, кОм")
        plt.plot(T1_LIST, R_LIST / 1000, color="k", linestyle="", marker="o", label="Эксперимент")
        plt.plot(new_t-273, new_r / 1000, color="k", label="расчет")
        plt.grid()
        plt.legend()
        plt.savefig("images/r_t.png")
        print("saved to images/r_t.png")
        plt.show()

    delta_t = get_delta(original_x=R_LIST, original_y=T1_KELV_LIST, approxed_x=new_r, approxed_y=new_t)
    print(f"{delta_t = }")

    if "" == "":
        plt.figure(figsize=(16, 9))
        plt.xlabel("Температура, °C")
        plt.ylabel("Отклонение, °C")
        plt.stem(T1_LIST, delta_t)
        plt.grid()
        plt.savefig("images/diff_t.png")
        print("saved to images/diff_t.png")
        plt.show()

    delta_t_max = max(np.abs(delta_t))
    print(f"{delta_t_max = :.3f}")

    error_lim = delta_t_max / (max(T1_LIST) - min(T1_LIST))
    print(f"eps = {error_lim * 100:.3f} %")
