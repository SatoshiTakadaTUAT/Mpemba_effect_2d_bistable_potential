import numpy as np
import scipy.special as sc
import matplotlib.pyplot as plt
T = 1
a_in = 1.0
a_mid = -0.5
a_out = 0.8
zeta = 1.0
alpha = 3.0
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
lam_max = 5
lam_list = np.linspace(0.0001, lam_max, 300, endpoint=True)
detM_list = np.array([detM(lam) for lam in lam_list])
zero = np.zeros_like(lam_list)
plt.figure(figsize=(8, 6))
plt.xlim(0, lam_max)
plt.ylim(-10, 20)
plt.xlabel('$\\lambda$', fontsize='24', math_fontfamily='cm')
plt.ylabel('$\\det \\mathcal{M}(\\lambda)$', fontsize='24', math_fontfamily='cm')
plt.tick_params(labelsize=18)
plt.grid(linestyle='solid', alpha=0.5)
plt.plot(lam_list, detM_list, '-', lw=2, color='r')
plt.plot(lam_list, zero, '--', lw=2, color='k')
plt.grid()
plt.tight_layout()
plt.savefig(f'Fig4.eps', bbox_inches='tight')
