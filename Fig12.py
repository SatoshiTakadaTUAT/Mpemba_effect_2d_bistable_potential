import numpy as np
import scipy.special as sc
from scipy import integrate
from scipy.optimize import brentq, minimize_scalar
import matplotlib.pyplot as plt
T = 1.0
beta = 1.0 / T
a_in = 1.0
a_mid = -1.0
a_out = 1.0
zeta = 1.0
alpha = 1.84485714
Rmax = 100.0
b_in = 0.0
b_mid = -a_mid * zeta ** 2
b_out = -a_out * alpha ** 2
r_m = zeta * np.sqrt(-a_mid / (a_in - a_mid))
r_p = np.sqrt((a_out * alpha ** 2 - a_mid * zeta ** 2) / (a_out - a_mid))
C_in = 0.0
C_mid = (a_in - a_mid) * r_m ** 2 / 2.0 - b_mid * np.log(r_m) + C_in
C_out = (a_mid - a_out) * r_p ** 2 / 2.0 + (b_mid - b_out) * np.log(r_p) + C_mid

def Pot(r):
    if r < r_m:
        return 0.5 * a_in * r ** 2 + C_in
    if r < r_p:
        return 0.5 * a_mid * r ** 2 + b_mid * np.log(r) + C_mid
    return 0.5 * a_out * r ** 2 + b_out * np.log(r) + C_out
nu_in = abs(b_in) / (2.0 * T)
nu_mid = abs(b_mid) / (2.0 * T)
nu_out = abs(b_out) / (2.0 * T)
gamma_in = abs(a_in) / (2.0 * T)
gamma_mid = abs(a_mid) / (2.0 * T)
gamma_out = abs(a_out) / (2.0 * T)

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
    z = gamma_in * r ** 2
    return r ** nu_in * np.exp(-z / 2.0) * M(mu_in(lam), 1.0 + nu_in, z)

def Phi1_midM(lam, r):
    z = gamma_mid * r ** 2
    return r ** nu_mid * np.exp(-z / 2.0) * M(mu_mid(lam), 1.0 + nu_mid, z)

def Phi1_midU(lam, r):
    z = gamma_mid * r ** 2
    return r ** nu_mid * np.exp(-z / 2.0) * U(mu_mid(lam), 1.0 + nu_mid, z)

def Phi1_outU(lam, r):
    z = gamma_out * r ** 2
    return r ** nu_out * np.exp(-z / 2.0) * U(mu_out(lam), 1.0 + nu_out, z)

def Phi3_inM(lam, r):
    z = gamma_in * r ** 2
    return r ** nu_in * np.exp(-z / 2.0) * M(1.0 + mu_in(lam), 2.0 + nu_in, z)

def Phi3_midM(lam, r):
    z = gamma_mid * r ** 2
    return r ** nu_mid * np.exp(-z / 2.0) * M(1.0 + mu_mid(lam), 2.0 + nu_mid, z)

def Phi3_midU(lam, r):
    z = gamma_mid * r ** 2
    return r ** nu_mid * np.exp(-z / 2.0) * U(1.0 + mu_mid(lam), 2.0 + nu_mid, z)

def Phi3_outU(lam, r):
    z = gamma_out * r ** 2
    return r ** nu_out * np.exp(-z / 2.0) * U(1.0 + mu_out(lam), 2.0 + nu_out, z)

def Phi2_inM(lam, r):
    return ((nu_in - gamma_in * r ** 2) * Phi1_inM(lam, r) + 2.0 * mu_in(lam) / (1.0 + nu_in) * gamma_in * r ** 2 * Phi3_inM(lam, r)) / r

def Phi2_midM(lam, r):
    return ((nu_mid - gamma_mid * r ** 2) * Phi1_midM(lam, r) + 2.0 * mu_mid(lam) / (1.0 + nu_mid) * gamma_mid * r ** 2 * Phi3_midM(lam, r)) / r

def Phi2_midU(lam, r):
    return ((nu_mid - gamma_mid * r ** 2) * Phi1_midU(lam, r) - 2.0 * mu_mid(lam) * gamma_mid * r ** 2 * Phi3_midU(lam, r)) / r

def Phi2_outU(lam, r):
    return ((nu_out - gamma_out * r ** 2) * Phi1_outU(lam, r) - 2.0 * mu_out(lam) * gamma_out * r ** 2 * Phi3_outU(lam, r)) / r

def MatrixM(lam):
    return np.array([[Phi1_inM(lam, r_m), -Phi1_midM(lam, r_m), -Phi1_midU(lam, r_m), 0.0], [Phi2_inM(lam, r_m), -Phi2_midM(lam, r_m), -Phi2_midU(lam, r_m), 0.0], [0.0, Phi1_midM(lam, r_p), Phi1_midU(lam, r_p), -Phi1_outU(lam, r_p)], [0.0, Phi2_midM(lam, r_p), Phi2_midU(lam, r_p), -Phi2_outU(lam, r_p)]], dtype=float)

def detM(lam):
    return np.linalg.det(MatrixM(lam))
lam_search_max = 20.0
lam_vals = np.linspace(1e-05, lam_search_max, 10000)
det_vals = np.array([detM(lam) for lam in lam_vals])
intervals = []
for i in range(len(lam_vals) - 1):
    if np.isfinite(det_vals[i]) and np.isfinite(det_vals[i + 1]) and (det_vals[i] * det_vals[i + 1] < 0.0):
        intervals.append((lam_vals[i], lam_vals[i + 1]))
raw_roots = []
for left, right in intervals:
    try:
        raw_roots.append(brentq(detM, left, right, xtol=1e-13, rtol=1e-13, maxiter=300))
    except (ValueError, RuntimeError):
        pass
raw_roots = sorted(set((round(root, 12) for root in raw_roots)))
physical_roots = [root for root in raw_roots if abs(root - round(root)) > 1e-06]
number_of_modes = 6
if len(physical_roots) < number_of_modes:
    raise RuntimeError('Fewer than six physical-root candidates were found.')
new_roots = physical_roots[:number_of_modes]

def integrate_piecewise(func, upper=Rmax, epsabs=1e-10, epsrel=1e-10):
    total = 0.0
    for left, right in [(0.0, r_m), (r_m, r_p), (r_p, upper)]:
        total += integrate.quad(func, left, right, limit=500, epsabs=epsabs, epsrel=epsrel)[0]
    return total

def coefficients_from_svd(lam):
    matrix = MatrixM(lam)
    _, singular_values, vh = np.linalg.svd(matrix)
    coeffs = vh[-1, :].copy()
    if coeffs[0] < 0.0:
        coeffs *= -1.0
    return (coeffs, singular_values[-1], np.linalg.norm(matrix @ coeffs))

def R(lam, r, A_in_value, A_mid_value, B_mid_value, B_out_value):
    if r < r_m:
        return A_in_value * Phi1_inM(lam, r)
    if r < r_p:
        return A_mid_value * Phi1_midM(lam, r) + B_mid_value * Phi1_midU(lam, r)
    return B_out_value * Phi1_outU(lam, r)
A_in_values = np.zeros(number_of_modes)
A_mid_values = np.zeros(number_of_modes)
B_mid_values = np.zeros(number_of_modes)
B_out_values = np.zeros(number_of_modes)
for j, lam in enumerate(new_roots):
    coeffs, sigma_min, residual = coefficients_from_svd(lam)
    A_in_values[j], A_mid_values[j], B_mid_values[j], B_out_values[j] = coeffs
    norm_integral = integrate_piecewise(lambda r: r * R(lam, r, A_in_values[j], A_mid_values[j], B_mid_values[j], B_out_values[j]) ** 2)
    scale = np.sqrt(2.0 * np.pi * norm_integral)
    A_in_values[j] /= scale
    A_mid_values[j] /= scale
    B_mid_values[j] /= scale
    B_out_values[j] /= scale

def Z(beta_value):
    return 2.0 * np.pi * integrate_piecewise(lambda r: r * np.exp(-beta_value * Pot(r)))

def Peq(r, beta_value):
    return np.exp(-beta_value * Pot(r)) / Z(beta_value)

def raw_mode_projection(lam, beta_prime, A_in_value, A_mid_value, B_mid_value, B_out_value):
    integral = integrate_piecewise(lambda r: r * R(lam, r, A_in_value, A_mid_value, B_mid_value, B_out_value) * np.exp((beta / 2.0 - beta_prime) * Pot(r)))
    return 2.0 * np.pi * integral / Z(beta_prime)

def a_hat(lam, beta_ini, A_in_value, A_mid_value, B_mid_value, B_out_value):
    return raw_mode_projection(lam, beta_ini, A_in_value, A_mid_value, B_mid_value, B_out_value) - raw_mode_projection(lam, beta, A_in_value, A_mid_value, B_mid_value, B_out_value)

def a2_of_beta(beta_ini):
    return a_hat(new_roots[0], beta_ini, A_in_values[0], A_mid_values[0], B_mid_values[0], B_out_values[0])
result = minimize_scalar(lambda beta_ini: -a2_of_beta(beta_ini), bounds=(0.1, 3.0), method='bounded', options={'xatol': 1e-06})
beta_ini_peak = result.x
beta_fixed = 1.15
lambda_2 = new_roots[0]
lambda_3 = new_roots[1]

def mode_amplitude(j, beta_ini):
    return a_hat(new_roots[j], beta_ini, A_in_values[j], A_mid_values[j], B_mid_values[j], B_out_values[j])
a2_fixed = mode_amplitude(0, beta_fixed)
a3_fixed = mode_amplitude(1, beta_fixed)
beta_min = 0.1
beta_max = 3.0
N_beta = 1000
beta_scan = np.linspace(beta_min, beta_max, N_beta)
a2_scan = np.zeros(N_beta)
a3_scan = np.zeros(N_beta)
Delta2_scan = np.zeros(N_beta)
Delta3_scan = np.zeros(N_beta)
F_scan = np.full(N_beta, np.nan)
for i, beta_ini in enumerate(beta_scan):
    a2_scan[i] = mode_amplitude(0, beta_ini)
    a3_scan[i] = mode_amplitude(1, beta_ini)
    Delta2_scan[i] = a2_scan[i] ** 2 - a2_fixed ** 2
    Delta3_scan[i] = a3_scan[i] ** 2 - a3_fixed ** 2
    if abs(Delta2_scan[i]) > 1e-12:
        F_scan[i] = Delta3_scan[i] / Delta2_scan[i]
plt.figure(figsize=(8, 6))
plt.plot(beta_scan, F_scan, linewidth=2.5, linestyle='-')
plt.fill_between(beta_scan, F_scan, -1.0, where=np.isfinite(F_scan) & (F_scan < -1.0), interpolate=True, color='orange', alpha=0.35)
plt.axhline(-1.0, linestyle='--', linewidth=1.5, color='black')
plt.axvline(beta, linestyle=':', linewidth=1.5, color='black')
plt.axvline(beta_fixed, linestyle='-.', linewidth=1.5, color='black')
plt.xlim(beta_min, beta_max)
plt.xlabel('$\\beta_{\\rm ini}^\\sharp$', fontsize=24, math_fontfamily='cm')
plt.ylabel('$\\mathcal{F}(\\beta_{\\rm ini}^\\sharp, \\beta_{\\rm ini}^*)$', fontsize=24, math_fontfamily='cm')
plt.xlim(1, 2)
plt.ylim(-2, 2)
plt.tick_params(labelsize=18)
plt.grid(linestyle='solid', alpha=0.35)
plt.tight_layout()
plt.savefig('Fig12.eps', bbox_inches='tight')