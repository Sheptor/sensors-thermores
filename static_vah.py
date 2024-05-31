import os
import numpy as np
import matplotlib.pyplot as plt
import config.config  # noqa
from utils.streinhart_model import steinhart_hart_model
from utils.linear_approx import get_linear_approx_coefficients
from utils.half_division import half_division_method

ID_TO_ROTATION = {
    0: 28,
    1: 32
}

I_LIST = np.arange(1, 21, 1) / 1000  # A
T1 = 25 + 273  # К
U1_LIST = np.array([
    4.61, 8.3, 10.6, 11.9,
    12.72, 13.13, 13.36, 13.43,
    13.39, 13.33, 13.17, 13,
    12.83, 12.66, 12.49, 12.3,
    12.13, 11.96, 11.79, 11.61
])  # V

T2 = 39.69 + 273  # К
U2_LIST = np.array([
    2.92, 5.23, 7.17, 8.7,
    9.71, 10.43, 10.95, 11.3,
    11.52, 11.66, 11.73, 11.76,
    11.74, 11.71, 11.65, 11.58,
    11.5, 11.41, 11.31, 11.21
])  # V


if __name__ == '__main__':
    if not os.path.exists("images"):
        os.makedirs("images")
    if not os.path.exists("results"):
        os.makedirs("results")

    p1_list = U1_LIST * I_LIST  # Вт
    p2_list = U2_LIST * I_LIST  # Вт

    r1_list = U1_LIST / I_LIST  # Ом
    r2_list = U2_LIST / I_LIST  # Ом

    print(f"{r1_list = }")
    print(f"{r2_list = }")
    np.savetxt("results/r1.csv", r1_list, delimiter=",")
    print("saved to results/r1.csv")
    np.savetxt("results/r2.csv", r2_list, delimiter=",")
    print("saved to results/r2.csv")

    a = 1.50843142e-03  # из kalib.py (p)
    b = 1.83585587e-04  # из kalib.py (p)
    c = 4.76118152e-07  # из kalib.py (p)

    new_t1 = steinhart_hart_model(r=r1_list, a=a, b=b, c=c)  # К
    new_t2 = steinhart_hart_model(r=r2_list, a=a, b=b, c=c)  # К

    np.savetxt("results/new_t1.csv", new_t1 - 273, delimiter=",")
    np.savetxt("results/new_t2.csv", new_t2 - 273, delimiter=",")

    print(f"{new_t1 = }")
    print(f"{new_t2 = }")

    _, _, k1, a1, t0_1 = get_linear_approx_coefficients(x=new_t1, y=p1_list)
    _, _, k2, a2, t0_2 = get_linear_approx_coefficients(x=new_t2, y=p2_list)

    print("K:")
    print(f"{k1*1000 = :.3f} мВт/C")
    print(f"{k2*1000 = :.3f} мВт/C")

    print("t0:")
    print(f"{t0_1 = }")
    print(f"{t0_2 = }")

    approxed_t1 = np.arange(t0_1, max(new_t1) + 0.01, 0.01)
    approxed_p1 = k1 * approxed_t1 + a1
    approxed_t2 = np.arange(t0_2, max(new_t2) + 0.01, 0.01)
    approxed_p2 = k2 * approxed_t2 + a2

    if "" == "":

        plt.figure(figsize=(16, 9))
        plt.xlabel("Температура, °C")
        plt.ylabel("Мощность, мВт")
        plt.text(
            float(50),
            float(105),
            f"P = {a1:.3f} + {k1:.3f}·T",
            ha='left',
            rotation=ID_TO_ROTATION[0]
        )
        plt.plot(
            new_t1 - 273, p1_list * 1000,
            color="k", linestyle="", marker="o", mfc='none', label=f"Температура среды {T1 - 273} °C"
        )
        plt.plot(approxed_t1 - 273, approxed_p1 * 1000, color="k")

        plt.text(
            float(50),
            float(63),
            f"P = {a2:.3f} + {k2:.3f}·T",
            ha='left',
            rotation=ID_TO_ROTATION[1]
        )
        plt.plot(
            new_t2 - 273, p2_list * 1000,
            color="gray", linestyle="", marker="s", label=f"Температура среды {T2 - 273} °C"
        )
        plt.plot(approxed_t2 - 273, approxed_p2 * 1000, color="gray")
        plt.grid()
        plt.legend()
        plt.savefig("images/p_t.png")
        print("saved to images/p_t.png")
        plt.show()

    r1 = half_division_method((4000, 5000), t0_1, 0.001, steinhart_hart_model, a, b, c)
    r2 = half_division_method((2000, 5000), t0_2, 0.001, steinhart_hart_model, a, b, c)
    print(f"\nСопротивление при нулевой мощности")
    print(f"{r1:.3f} Ом")
    print(f"{r2:.3f} Ом")

    I0_1 = np.sqrt((k1 * 0.1) / r1)
    I0_2 = np.sqrt((k2 * 0.1) / r2)

    print(f"\nТок нулевой мощности:")
    print(f"{I0_1 = :.6f} А")
    print(f"{I0_2 = :.6f} А")

    if "" == "":
        new_i = np.arange(1, 20.1, 0.1) / 1000  # А
        new_approxed_t1 = []
        new_approxed_r1 = []
        for i in new_i:
            tt = T1
            R = float("inf")

            while i**2 * R > k1 * (tt - T1):
                v = (1/c) * (a - (1/tt))
                u = ((b/(3*c))**3 + (v/2)**2) ** (1/2)
                R = np.exp((u - (v/2))**(1/3) - (u + (v/2))**(1/3))  # Ом
                tt += 0.01
            new_approxed_t1.append(tt)
            new_approxed_r1.append(R)
        new_approxed_u1 = new_i * new_approxed_r1

        new_approxed_t2 = []
        new_approxed_r2 = []
        for i in new_i:
            tt = T2
            R = float("inf")

            while i**2 * R > k2 * (tt - T2):
                v = (1/c) * (a - (1/tt))
                u = ((b/(3*c))**3 + (v/2)**2) ** (1/2)
                R = np.exp((u - (v/2))**(1/3) - (u + (v/2))**(1/3))  # Ом
                tt += 0.01
            new_approxed_t2.append(tt)
            new_approxed_r2.append(R)
        new_approxed_u2 = new_i * new_approxed_r2

        if "" == "":
            plt.figure(figsize=(16, 9))
            plt.xlabel("Напряжение, В")
            plt.ylabel("Ток, мА")
            plt.plot(
                U1_LIST, I_LIST * 1000,
                color="k", linestyle="", marker="o", mfc='none', label=f"Температура среды {T1 - 273} °C"
            )
            plt.plot(new_approxed_u1, new_i * 1000, color="k")
            plt.plot(
                U2_LIST, I_LIST * 1000,
                color="gray", linestyle="", marker="s", label=f"Температура среды {T2 - 273} °C"
            )
            plt.plot(new_approxed_u2, new_i * 1000, color="gray")
            plt.grid()
            plt.legend()
            plt.savefig("images/static_vah.png")
            print("saved to images/static_vah.png")
            plt.show()
    


