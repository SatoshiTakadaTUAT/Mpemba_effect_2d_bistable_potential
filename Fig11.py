import numpy as np
import scipy.special as sc
from scipy import integrate
from scipy.optimize import brentq
import matplotlib.pyplot as plt
T = 1
a_in = 1.5
a_mid = -1.5
a_out = 1.5
zeta = 1.0
alpha = 1.5
Rmax = 100.0
beta = 1 / T
b_in = 0.0
b_mid = -a_mid * zeta ** 2
b_out = -a_out * alpha ** 2
r_m = zeta * np.sqrt(-a_mid / (a_in - a_mid))
r_p = np.sqrt((a_out * alpha ** 2 - a_mid * zeta ** 2) / (a_out - a_mid))
C_in = 0
C_mid = (a_in - a_mid) / 2 * r_m ** 2 - b_mid * np.log(r_m) + C_in
C_out = (a_mid - a_out) / 2 * r_p ** 2 + (b_mid - b_out) * np.log(r_p) + C_mid

def Pot(r):
    if r < r_m:
        return a_in / 2 * r ** 2 + C_in
    elif r < r_p:
        return a_mid / 2 * r ** 2 + b_mid * np.log(r) + C_mid
    else:
        return a_out / 2 * r ** 2 + b_out * np.log(r) + C_out
nu_in = abs(b_in) / (2 * T)
nu_mid = abs(b_mid) / (2 * T)
nu_out = abs(b_out) / (2 * T)
gamma_in = abs(a_in) / (2 * T)
gamma_mid = abs(a_mid) / (2 * T)
gamma_out = abs(a_out) / (2 * T)

def mu_value(lam, a, b, nu, gamma):
    return (1.0 + nu) / 2.0 - beta * (lam + a - a * b / (2.0 * T)) / (4.0 * gamma)

def mu_in(lam):
    return mu_value(lam, a_in, b_in, nu_in, gamma_in)

def mu_mid(lam):
    return mu_value(lam, a_mid, b_mid, nu_mid, gamma_mid)

def mu_out(lam):
    return mu_value(lam, a_out, b_out, nu_out, gamma_out)

def M(a, b, z):
    return sc.hyp1f1(a, b, z)

def U(a, b, z):
    return sc.hyperu(a, b, z)

def Phi1_inM(lam, r):
    return pow(r, nu_in) * np.exp(-gamma_in * r ** 2 / 2) * M(mu_in(lam), 1 + nu_in, gamma_in * r ** 2)

def Phi1_midM(lam, r):
    return pow(r, nu_mid) * np.exp(-gamma_mid * r ** 2 / 2) * M(mu_mid(lam), 1 + nu_mid, gamma_mid * r ** 2)

def Phi1_midU(lam, r):
    return pow(r, nu_mid) * np.exp(-gamma_mid * r ** 2 / 2) * U(mu_mid(lam), 1 + nu_mid, gamma_mid * r ** 2)

def Phi1_outU(lam, r):
    return pow(r, nu_out) * np.exp(-gamma_out * r ** 2 / 2) * U(mu_out(lam), 1 + nu_out, gamma_out * r ** 2)

def Phi3_inM(lam, r):
    return pow(r, nu_in) * np.exp(-gamma_in * r ** 2 / 2) * M(1 + mu_in(lam), 2 + nu_in, gamma_in * r ** 2)

def Phi3_midM(lam, r):
    return pow(r, nu_mid) * np.exp(-gamma_mid * r ** 2 / 2) * M(1 + mu_mid(lam), 2 + nu_mid, gamma_mid * r ** 2)

def Phi3_midU(lam, r):
    return pow(r, nu_mid) * np.exp(-gamma_mid * r ** 2 / 2) * U(1 + mu_mid(lam), 2 + nu_mid, gamma_mid * r ** 2)

def Phi3_outU(lam, r):
    return pow(r, nu_out) * np.exp(-gamma_out * r ** 2 / 2) * U(1 + mu_out(lam), 2 + nu_out, gamma_out * r ** 2)

def Phi2_inM(lam, r):
    return 1 / r * ((nu_in - gamma_in * r ** 2) * Phi1_inM(lam, r) + 2 * mu_in(lam) / (1 + nu_in) * gamma_in * r ** 2 * Phi3_inM(lam, r))

def Phi2_midM(lam, r):
    return 1 / r * ((nu_mid - gamma_mid * r ** 2) * Phi1_midM(lam, r) + 2 * mu_mid(lam) / (1 + nu_mid) * gamma_mid * r ** 2 * Phi3_midM(lam, r))

def Phi2_midU(lam, r):
    return 1 / r * ((nu_mid - gamma_mid * r ** 2) * Phi1_midU(lam, r) - 2 * mu_mid(lam) * gamma_mid * r ** 2 * Phi3_midU(lam, r))

def Phi2_outU(lam, r):
    return 1 / r * ((nu_out - gamma_out * r ** 2) * Phi1_outU(lam, r) - 2 * mu_out(lam) * gamma_out * r ** 2 * Phi3_outU(lam, r))

def MatrixM(lam):
    return np.array([[Phi1_inM(lam, r_m), -Phi1_midM(lam, r_m), -Phi1_midU(lam, r_m), 0], [Phi2_inM(lam, r_m), -Phi2_midM(lam, r_m), -Phi2_midU(lam, r_m), 0], [0, Phi1_midM(lam, r_p), Phi1_midU(lam, r_p), -Phi1_outU(lam, r_p)], [0, Phi2_midM(lam, r_p), Phi2_midU(lam, r_p), -Phi2_outU(lam, r_p)]], dtype=float)

def detM(lam):
    return np.linalg.det(MatrixM(lam))

def smallest_singular_value(lam):
    return np.linalg.svd(MatrixM(lam), compute_uv=False)[-1]
lam = 2
r_list = np.linspace(1e-08, 5, 500, endpoint=True)
lam_max = 12
lam_vals = np.linspace(0.0001, lam_max, 5000)
det_vals = np.array([detM(l) for l in lam_vals])
intervals = []
for i in range(len(lam_vals) - 1):
    if det_vals[i] * det_vals[i + 1] < 0:
        intervals.append((lam_vals[i], lam_vals[i + 1]))
roots = []
for a, b in intervals:
    try:
        r = brentq(detM, a, b)
        roots.append(r)
    except ValueError:
        pass
tol = 1e-08
physical_roots = []
for lam in sorted(roots):
    if abs(lam - round(lam)) < tol:
        continue
    physical_roots.append(lam)
new_roots = physical_roots


def MatrixM_(lam):
    return np.array([[Phi1_midM(lam, r_m), Phi1_midU(lam, r_m), 0], [Phi2_midM(lam, r_m), Phi2_midU(lam, r_m), 0], [Phi1_midM(lam, r_p), Phi1_midU(lam, r_p), -Phi1_outU(lam, r_p)]], dtype=float)

def VectorA(lam):
    return np.array([[Phi1_inM(lam, r_m)], [Phi2_inM(lam, r_m)], [0]], dtype=float)

def A_B(lam):
    return np.linalg.solve(MatrixM_(lam), VectorA(lam))

def R(lam, r, A_in, A_mid, B_mid, B_out):
    if r < r_m:
        return A_in * Phi1_inM(lam, r)
    elif r < r_p:
        return A_mid * Phi1_midM(lam, r) + B_mid * Phi1_midU(lam, r)
    else:
        return B_out * Phi1_outU(lam, r)

def integrand_rR2(lam, r, A_in, A_mid, B_mid, B_out):
    return r * R(lam, r, A_in, A_mid, B_mid, B_out) ** 2
roots = sorted(set((round(root, 12) for root in roots)))
physical_roots = [root for root in roots if abs(root - round(root)) > 1e-06]
if len(physical_roots) < 4:
    raise RuntimeError('Fewer than four noninteger candidate roots were found. Increase lam_max/resolution and inspect detM and the singular values.')
new_roots = physical_roots[:4]
A_in = [0 for _ in range(len(new_roots))]
A_mid = [0 for _ in range(len(new_roots))]
B_mid = [0 for _ in range(len(new_roots))]
B_out = [0 for _ in range(len(new_roots))]
for n in range(len(new_roots)):
    lam = new_roots[n]
    A_in[n] = 1
    A_mid[n] = A_B(lam)[0, 0]
    B_mid[n] = A_B(lam)[1, 0]
    B_out[n] = A_B(lam)[2, 0]
    I = integrate.quad(lambda x: integrand_rR2(lam, x, A_in[n], A_mid[n], B_mid[n], B_out[n]), 0.0, Rmax, limit=600, epsabs=1e-09, epsrel=1e-09)[0]
    A_in[n] /= np.sqrt(2 * np.pi * I)
    A_mid[n] /= np.sqrt(2 * np.pi * I)
    B_mid[n] /= np.sqrt(2 * np.pi * I)
    B_out[n] /= np.sqrt(2 * np.pi * I)
    I = integrate.quad(lambda x: integrand_rR2(lam, x, A_in[n], A_mid[n], B_mid[n], B_out[n]), 0.0, Rmax, limit=600, epsabs=1e-09, epsrel=1e-09)[0]
for n in range(len(new_roots)):
    R_list = []
    lam = new_roots[n]
    for r in r_list:
        R_list.append(R(lam, r, A_in[n], A_mid[n], B_mid[n], B_out[n]))
    Phi1_list = []
    Phi2_list = []
    Phi3_list = []
    Phi4_list = []
    for r in r_list:
        Phi1_list.append(Phi1_inM(lam, r))
        Phi2_list.append(Phi1_midM(lam, r))
        Phi3_list.append(Phi1_midU(lam, r))
        Phi4_list.append(Phi1_outU(lam, r))

def integrand_rR(lam, r, beta_ini, A_in, A_mid, B_mid, B_out):
    return r * R(lam, r, A_in, A_mid, B_mid, B_out) * np.exp((beta / 2 - beta_ini) * Pot(r))

def integrand_Z(r, beta_ini):
    return r * np.exp(-beta_ini * Pot(r))

def Z(beta_ini):
    val = integrate.quad(lambda x: integrand_Z(x, beta_ini), 0.0, Rmax, limit=600, epsabs=1e-09, epsrel=1e-09)[0]
    return 2 * np.pi * val

def raw_mode_projection(lam, beta_prime, A_in, A_mid, B_mid, B_out):
    integral = integrate.quad(lambda x: integrand_rR(lam, x, beta_prime, A_in, A_mid, B_mid, B_out), 0.0, Rmax, limit=600, epsabs=1e-09, epsrel=1e-09)[0]
    return 2.0 * np.pi * integral / Z(beta_prime)

def am0(lam, beta_ini, A_in, A_mid, B_mid, B_out):
    initial_projection = raw_mode_projection(lam, beta_ini, A_in, A_mid, B_mid, B_out)
    final_projection = raw_mode_projection(lam, beta, A_in, A_mid, B_mid, B_out)
    return initial_projection - final_projection
beta_ini_list = np.linspace(0.1, 3.0, 300, endpoint=True)
a_list = [[0 for _ in range(len(beta_ini_list))] for _ in range(len(new_roots))]
for n in range(len(new_roots)):
    lam = new_roots[n]
    Z_list = []
    for beta_ini in beta_ini_list:
        Z_list.append(Z(beta_ini))
    for i in range(len(beta_ini_list)):
        beta_ini = beta_ini_list[i]
        a_list[n][i] = am0(lam, beta_ini, A_in[n], A_mid[n], B_mid[n], B_out[n])
for i in range(len(beta_ini_list)):
    beta_ini = beta_ini_list[i]


plt.figure(figsize=(8, 5))
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['legend.fontsize'] = 18
plt.xlabel('$\\beta_\\mathrm{ini}$', fontsize='24', math_fontfamily='cm')
plt.ylabel('$a_2$', fontsize='24', math_fontfamily='cm')
plt.tick_params(labelsize=18)
plt.grid(linestyle='solid', alpha=0.5)
plt.xlim(0, 3)
plt.plot(beta_ini_list, a_list[0], '-', lw=2, color='r', label='$a_2$')
plt.axvline(1.0, color='black', linestyle='--', linewidth=2)
plt.axhline(0.0, color='black', linestyle=':', linewidth=2)
plt.legend(loc='lower right', fontsize=18, ncol=2)
plt.grid()
plt.tight_layout()
plt.savefig(f'Fig11a.eps', bbox_inches='tight')

plt.figure(figsize=(8, 5))
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['legend.fontsize'] = 18
plt.xlabel('$\\beta_\\mathrm{ini}$', fontsize='24', math_fontfamily='cm')
plt.ylabel('$a_3, a_4, a_5$', fontsize='24', math_fontfamily='cm')
plt.tick_params(labelsize=18)
plt.grid(linestyle='solid', alpha=0.5)
plt.xlim(0, 3)
plt.plot(beta_ini_list, a_list[1], '--', lw=2, color='b', label='$a_3$')
plt.plot(beta_ini_list, a_list[2], ':', lw=2, color='k', label='$a_4$')
plt.plot(beta_ini_list, a_list[3], '-.', lw=2, color='m', label='$a_5$')
plt.axvline(1.0, color='black', linestyle='--', linewidth=2)
plt.axhline(0.0, color='black', linestyle=':', linewidth=2)
plt.legend(loc='lower right', fontsize=18, ncol=2)
plt.grid()
plt.tight_layout()
plt.savefig(f'Fig11b.eps', bbox_inches='tight')
