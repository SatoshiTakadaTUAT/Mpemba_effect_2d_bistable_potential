import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
k_vals = np.linspace(-1.999, -1e-06, 300)

def V(alpha, k):
    return 0.5 * alpha ** 2 * np.log((alpha ** 2 - k) / (alpha ** 2 * (1 - k))) + 0.5 * k * np.log(-k / (alpha ** 2 - k))
alpha_solutions = []
for k in k_vals:
    try:
        root = brentq(lambda a: V(a, k), 1.0, 5.0)
        alpha_solutions.append(root)
    except ValueError:
        alpha_solutions.append(np.nan)
alpha_solutions = np.array(alpha_solutions)
plt.figure()
plt.plot(-k_vals, alpha_solutions, lw=2)
plt.xlabel('$-k_\\mathrm{mid}$', fontsize=24, math_fontfamily='cm')
plt.ylabel('$\\alpha$', fontsize=24, math_fontfamily='cm')
plt.tick_params(labelsize=18)
plt.grid(linestyle='solid', alpha=0.5)
plt.xlim(0, 2)
plt.ylim(1, 3)
plt.text(0.8, 1.3, '$V(\\alpha)>0$', fontsize=24, math_fontfamily='cm')
plt.text(1, 2.3, '$V(\\alpha)<0$', fontsize=24, math_fontfamily='cm')
plt.tight_layout()
plt.savefig(f'Fig2.eps', bbox_inches='tight')
