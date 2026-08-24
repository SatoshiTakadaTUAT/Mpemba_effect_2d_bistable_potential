import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
a_in, a_mid, a_out = (1, -0.5, 0.8)
xi = 1
alpha = 3
b_mid = -a_mid * xi ** 2
b_out = -a_out * alpha ** 2
r_m = xi * np.sqrt(-a_mid / (a_in - a_mid))
r_p = np.sqrt((a_out * alpha ** 2 - a_mid * xi ** 2) / (a_out - a_mid))
r_max = 3.8
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
N = 600
x = np.linspace(-r_max, r_max, N)
y = np.linspace(-r_max, r_max, N)
X, Y = np.meshgrid(x, y)
R = np.sqrt(X ** 2 + Y ** 2)
U_vec = np.vectorize(U)
Z = U_vec(R)
plt.figure(figsize=(8, 7))
ax = plt.axes(projection='3d')
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['legend.fontsize'] = 18
plt.tick_params(labelsize=18)
ax.zaxis.set_major_locator(MultipleLocator(1))
surf = ax.plot_surface(X, Y, Z, cmap='jet', linewidth=0, antialiased=True, rcount=100, ccount=100)
ax.set_xlabel('$x$', fontsize=18, math_fontfamily='cm')
ax.set_ylabel('$y$', fontsize=18, math_fontfamily='cm')
ax.set_xlim(-r_max, r_max)
ax.set_ylim(-r_max, r_max)
cbar = plt.colorbar(surf, shrink=0.6)
cbar.set_label('$V(r)$', fontsize=18, math_fontfamily='cm')
cbar.ax.tick_params(labelsize=18)
plt.tight_layout()
plt.savefig(f'Fig1.eps', bbox_inches='tight')
