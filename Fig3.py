import numpy as np
import matplotlib.pyplot as plt
a_in, a_mid, a_out = (1, -0.5, 0.8)
xi = 1
alpha = 3
b_in = 0
b_mid = -a_mid * xi ** 2
b_out = -a_out * alpha ** 2
T = 1
r_m = xi * np.sqrt(-a_mid / (a_in - a_mid))
r_p = np.sqrt((a_out * alpha ** 2 - a_mid * xi ** 2) / (a_out - a_mid))
r_max = 5
C_in = 0.0
C_mid = (a_in - a_mid) / 2 * r_m ** 2 - b_mid * np.log(r_m) + C_in
C_out = (a_mid - a_out) / 2 * r_p ** 2 + (b_mid - b_out) * np.log(r_p) + C_mid

def U(r):
    if r < r_m:
        return a_in / 2 * r ** 2 + C_in
    elif r < r_p:
        return a_mid / 2 * r ** 2 + b_mid * np.log(r) + C_mid
    else:
        return a_out / 2 * r ** 2 + b_out * np.log(r) + C_out

def V_S(r):
    if r < r_m:
        return -a_in + a_in * b_in / (2 * T) + a_in ** 2 / (4 * T) * r ** 2 + b_in ** 2 / (4 * T * r ** 2)
    elif r < r_p:
        return -a_mid + a_mid * b_mid / (2 * T) + a_mid ** 2 / (4 * T) * r ** 2 + b_mid ** 2 / (4 * T * r ** 2)
    else:
        return -a_out + a_out * b_out / (2 * T) + a_out ** 2 / (4 * T) * r ** 2 + b_out ** 2 / (4 * T * r ** 2)
r = np.linspace(0.0001, r_max, 1200)
U_list = np.array([U(ri) for ri in r])
VS_list = np.array([V_S(ri) for ri in r])
plt.figure(figsize=(8, 5))
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['legend.fontsize'] = 18
plt.plot(r, U_list, ls='-', lw=3, label='$V(r)$')
plt.plot(r, VS_list, ls='-.', lw=3, label='$V_S(r)$')
plt.axvline(r_m, ls='dotted', lw=2, color='black')
plt.axvline(r_p, ls='dotted', lw=2, color='black')
plt.xlim(0, r_max)
plt.ylim(-1.5, 2)
plt.tick_params(labelsize=18)
plt.grid(linestyle='solid', alpha=0.5)
plt.xlabel('$r$', fontsize='28', math_fontfamily='cm')
plt.ylabel('$V(r),\\ V_S(r)$', fontsize='28', math_fontfamily='cm')
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig(f'Fig3.eps', bbox_inches='tight')
