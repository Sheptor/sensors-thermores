import numpy as np
import matplotlib.pyplot as plt
import config.config  # noqa
from utils.linear_approx import get_linear_approx_coefficients
from utils.streinhart_model import steinhart_hart_model
from scipy.optimize import curve_fit


def fix_start_values(x, y):

    new_x = np.arange(int(x1_min), x[0] + 1, 1)
    new_y = a1 * new_x + b1
    return new_x, new_y


def temp_by_time(t, l_tau):
    return T + (t1_list[0] - T) * np.exp(-t / l_tau)


TIME_LIST = np.arange(0, 121, 1)  # с

T1 = 30 + 273  # К
#   1       2       3       4       5       6       7        8       9       10
U1_LIST = np.array([
    10.3,  # 0
    10.005, 9.6300, 9.5000, 9.2240, 8.9770, 8.4770, 7.8320, 6.8370, 5.4820, 3.2030,  # 1 - 10
    0.1255, 0.1874, 0.1996, 0.2185, 0.2247, 0.2331, 0.2441, 0.2506, 0.2594, 0.2646,  # 11 - 20
    0.2718, 0.2760, 0.2801, 0.2855, 0.2890, 0.2937, 0.2967, 0.3008, 0.3033, 0.3058,  # 21 - 30
    0.3092, 0.3115, 0.3144, 0.3164, 0.3192, 0.3210, 0.3227, 0.3251, 0.3267, 0.3289,  # 31 - 40
    0.3304, 0.3324, 0.3337, 0.3355, 0.3367, 0.3378, 0.3397, 0.3408, 0.3424, 0.3433,  # 41 - 50
    0.3448, 0.3457, 0.3472, 0.3481, 0.3494, 0.3503, 0.3511, 0.3523, 0.3531, 0.3543,  # 51 - 60
    0.3550, 0.3557, 0.3567, 0.3574, 0.3585, 0.3591, 0.3601, 0.3606, 0.3615, 0.3621,  # 61 - 70
    0.3630, 0.3636, 0.3645, 0.3650, 0.3658, 0.3663, 0.3670, 0.3675, 0.3681, 0.3687,  # 71 - 80
    0.3691, 0.3696, 0.3703, 0.3708, 0.3714, 0.3718, 0.3724, 0.3728, 0.3732, 0.3739,  # 81 - 90
    0.3744, 0.3748, 0.3753, 0.3757, 0.3762, 0.3766, 0.3769, 0.3774, 0.3777, 0.3782,  # 91 - 100
    0.3785, 0.3788, 0.3792, 0.3795, 0.3801, 0.3803, 0.3806, 0.3810, 0.3813, 0.3816,  # 100 - 110
    0.3820, 0.3824, 0.3826, 0.3830, 0.3833, 0.3836, 0.3839, 0.3842, 0.3844, 0.3846,  # 111 - 120
])  # В

T2 = 59.3 + 273  # К
#   1       2       3       4       5       6       7       8       9       10
U2_LIST = np.array([
    7.48,  # 0
    7.2450, 6.9420, 6.5600, 6.3330, 6.1060, 5.5550, 5.2310, 4.5100, 4.0180, 2.8910,  # 1 - 10
    0.8900, 0.0963, 0.0916, 0.0928, 0.0942, 0.0979, 0.1000, 0.1028, 0.1044, 0.1065,  # 11 - 20
    0.1078, 0.1095, 0.1105, 0.1119, 0.1128, 0.1140, 0.1147, 0.1157, 0.1164, 0.1170,  # 21 - 30
    0.1178, 0.1183, 0.1188, 0.1196, 0.1200, 0.1207, 0.1211, 0.1216, 0.1220, 0.1224,  # 31 - 40
    0.1229, 0.1234, 0.1237, 0.1240, 0.1247, 0.1248, 0.1252, 0.1255, 0.1258, 0.1261,  # 41 - 50
    0.1263, 0.1267, 0.1269, 0.1272, 0.1275, 0.1279, 0.1280, 0.1281, 0.1285, 0.1287,  # 51 - 60
    0.1289, 0.1291, 0.1294, 0.1295, 0.1298, 0.1300, 0.1302, 0.1304, 0.1306, 0.1308,  # 61 - 70
    0.1309, 0.1318, 0.1313, 0.1314, 0.1316, 0.1317, 0.1319, 0.1320, 0.1322, 0.1324,  # 71 - 80
    0.1326, 0.1327, 0.1328, 0.1329, 0.1330, 0.1332, 0.1333, 0.1335, 0.1336, 0.1337,  # 81 - 90
    0.1338, 0.1339, 0.1341, 0.1342, 0.1343, 0.1344, 0.1345, 0.1346, 0.1347, 0.1348,  # 91 - 100
    0.1350, 0.1350, 0.1352, 0.1353, 0.1353, 0.1355, 0.1355, 0.1357, 0.1357, 0.1358,  # 101 - 110
    0.1359, 0.1360, 0.1360, 0.1362, 0.1362, 0.1363, 0.1364, 0.1365, 0.1366, 0.1366,  # 111 - 120
])  # В

a = 1.50843142e-03  # из kalib.py (p)
b = 1.83585587e-04  # из kalib.py (p)
c = 4.76118152e-07  # из kalib.py (p)

k1 = 0.0042  # экстраполированное значение из utils.linear_approx.py
k2 = 0.0058  # экстраполированное значение из utils.linear_approx.py

I0 = 1e-4
# U0_1 = 3965.05078125 * I0  # 464.78 <- st
# U0_2 = 1314.2138671875 * I0  # 464.78 <- st


if __name__ == '__main__':
    _, _, a1, b1, x1_min = get_linear_approx_coefficients(x=TIME_LIST[12:17], y=U1_LIST[12:17])
    # _, _, a1, b1, x1_min = get_linear_approx_coefficients(x=[0, TIME_LIST[12]], y=[0.01, U1_LIST[12]])
    new_start_t1 = np.arange(0, 13, 1)
    new_start_u1 = a1 * new_start_t1 + b1
    # print(a1, b1)

    # _, _, a2, b2, x2_min = get_linear_approx_coefficients(x=[0, TIME_LIST[15]], y=[0.01, U2_LIST[15]])
    _, _, a2, b2, x2_min = get_linear_approx_coefficients(x=[0, TIME_LIST[15]], y=[b1, U2_LIST[15]])
    new_start_t2 = np.arange(0, 16, 1)
    new_start_u2 = a2 * new_start_t2 + b2
    # print(a2, b2)
    # new_start_t1, new_start_u1 = fix_start_values(x=TIME_LIST[12:15], y=U1_LIST[12:15])
    # new_start_t2, new_start_u2 = fix_start_values(x=[0, TIME_LIST[15]], y=[0, U2_LIST[15]])

    new_u1 = np.hstack([new_start_u1, U1_LIST[13:]])
    new_u2 = np.hstack([new_start_u2, U2_LIST[16:]])

    r1 = new_u1 / I0  # Ом
    r2 = new_u2 / I0  # Ом

    t1_list = steinhart_hart_model(r1, a, b, c)  # К
    t2_list = steinhart_hart_model(r2, a, b, c)  # К

    T = T1
    new_time = np.arange(0, 120.1, 0.1)
    p = curve_fit(temp_by_time, xdata=TIME_LIST, ydata=t1_list, p0=[1.0], method="lm")
    tau1 = p[0][0]
    approxed_t1 = temp_by_time(t=new_time, l_tau=p[0][0])

    T = T2
    p = curve_fit(temp_by_time, xdata=TIME_LIST, ydata=t2_list, p0=[1.0], method="lm")
    tau2 = p[0][0]
    approxed_t2 = temp_by_time(t=new_time, l_tau=p[0][0])

    c1 = k1 / tau1
    c2 = k2 / tau2

    print(f"{tau1 = :.6f} с")
    print(f"{tau2 = :.6f} с\n")
    print(f"{c1*1000 = :.3f} мДж/C")
    print(f"{c2*1000 = :.3f} мДж/C")

    if "" == "":  # Wrong measurements
        plt.figure(figsize=(16, 9))
        plt.xlabel("Время, с")
        plt.ylabel("Напряжение, В")
        plt.plot(
            TIME_LIST, U1_LIST, color="k", marker="o", mfc='none', label=f"Температура среды {T1 - 273} °C", linestyle=""
        )
        plt.plot(
            TIME_LIST, U2_LIST, color="gray", marker="s", label=f"Температура среды {T2 - 273:.2f} °C", linestyle=""
        )
        plt.grid()
        plt.legend()
        plt.savefig("images/wrong_u_time.png")
        print("saved to images/wrong_u_time.png")
        plt.show()

    if "" == "":
        plt.figure(figsize=(16, 9))
        plt.xlabel("Время, с")
        plt.ylabel("Напряжение, В")
        plt.plot(
            TIME_LIST, new_u1, color="k", marker="o", mfc='none', label=f"Температура среды {T1 - 273} °C", linestyle=""
        )
        plt.plot(TIME_LIST, new_u1, color="k")
        plt.plot(
            TIME_LIST, new_u2, color="gray", marker="s", label=f"Температура среды {T2 - 273:.2f} °C", linestyle=""
        )
        plt.plot(TIME_LIST, new_u2, color="gray")
        plt.grid()
        plt.legend()
        plt.savefig("images/u_time.png")
        print("saved to images/u_time.png")
        plt.show()

    if "" == "":
        plt.figure(figsize=(16, 9))
        plt.xlabel("Время, с")
        plt.ylabel("Температура, °C")
        plt.plot(
            TIME_LIST, t1_list - 273,
            color="k", marker="o", mfc='none', label=f"Температура среды {T1 - 273} °C", linestyle=""
        )
        plt.plot(new_time, approxed_t1 - 273, color="k")
        plt.plot(
            TIME_LIST, t2_list - 273,
            color="gray", marker="s", label=f"Температура среды {T2 - 273:.2f} °C", linestyle=""
        )
        plt.plot(new_time, approxed_t2 - 273, color="gray")
        plt.axhline(y=T1 - 273, color=(0.3, 0.3, 0.3), linestyle='--')
        plt.axhline(y=T2 - 273, color=(0.3, 0.3, 0.3), linestyle='--')
        plt.grid()
        plt.legend()
        plt.savefig("images/temp_time.png")
        print("saved to images/temp_time.png")
        plt.show()
