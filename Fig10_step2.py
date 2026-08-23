import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Affine2D
filename = 'Fig10.dat'
data = np.loadtxt(filename, comments='#')
minus_kmid = data[:, 0]
alpha = data[:, 1]
cls = data[:, 2].astype(int)
beta_peak = data[:, 3]
V_alpha = data[:, 8]
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.size'] = 12
fig, ax = plt.subplots(figsize=(7.2, 6.0))
mask = (cls == 2) & (beta_peak < 1.0)
ax.scatter(minus_kmid[mask], alpha[mask], marker='$✓$', s=45, color='tab:blue', linewidths=0.8, label='Mpemba effect', zorder=4)
mask = (cls == 2) & (beta_peak >= 1.0)
marker_inverse = MarkerStyle('$✓$', transform=Affine2D().rotate_deg(180))
ax.scatter(minus_kmid[mask], alpha[mask], marker=marker_inverse, s=45, color='tab:green', linewidths=0.8, label='Inverse Mpemba effect', zorder=4)
mask = cls == 1
ax.scatter(minus_kmid[mask], alpha[mask], marker='^', s=38, color='tab:orange', edgecolors='none', label='Peak in $a_2$, no crossing', zorder=5)
mask = cls == 0
ax.scatter(minus_kmid[mask], alpha[mask], marker='_', s=35, color='0.35', linewidths=1.2, label='No peak in $a_2$', zorder=2)
mask = cls == -1
ax.scatter(minus_kmid[mask], alpha[mask], marker='$=$', s=24, color='0.35', edgecolors='none', label='Unstable', zorder=3)
xvals = np.sort(np.unique(minus_kmid))
yvals = np.sort(np.unique(alpha))
Vgrid = np.full((len(yvals), len(xvals)), np.nan)
for x, y, v in zip(minus_kmid, alpha, V_alpha):
    ix = np.argmin(np.abs(xvals - x))
    iy = np.argmin(np.abs(yvals - y))
    Vgrid[iy, ix] = v
X, Y = np.meshgrid(xvals, yvals)
ax.contour(X, Y, Vgrid, levels=[0.0], colors='black', linewidths=1.5, zorder=6)
handles, labels = ax.get_legend_handles_labels()
handles.append(Line2D([0], [0], color='black', lw=1.5))
labels.append('$V(\\alpha)=0$')
ax.legend(handles, labels, loc='lower right', frameon=True, fontsize=10)
ax.set_xlabel('$-k_{\\rm mid}$', fontsize=16)
ax.set_ylabel('$\\alpha$', fontsize=16)
ax.set_xlim(xvals.min() - 0.03, xvals.max() + 0.03)
ax.set_ylim(yvals.min() - 0.03, yvals.max() + 0.03)
ax.tick_params(direction='in', labelsize=12)
fig.tight_layout()
fig.savefig('Fig10.eps', bbox_inches='tight')
