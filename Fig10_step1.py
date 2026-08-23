import numpy as np
import scipy.special as sc
from scipy import integrate
from scipy.optimize import brentq, minimize_scalar
T = 1.0
beta = 1.0 / T
Rmax = 100.0
beta_min = 0.1
beta_max = 3.0
lam_search_max = 15.0
N_lam_search = 5000
N_beta_search = 250
number_of_modes = 2

def solve_parameter_set(kmid_abs, alpha_value, verbose=False):
    a_in = 1.0
    a_mid = -float(kmid_abs)
    a_out = 1.0
    zeta = 1.0
    alpha = float(alpha_value)
    if kmid_abs <= 0.0:
        return {'class': -1, 'reason': 'kmid_abs <= 0'}
    if alpha <= zeta:
        return {'class': -1, 'reason': 'alpha <= zeta'}
    b_in = 0.0
    b_mid = -a_mid * zeta ** 2
    b_out = -a_out * alpha ** 2
    r_m = zeta * np.sqrt(-a_mid / (a_in - a_mid))
    r_p = np.sqrt((a_out * alpha ** 2 - a_mid * zeta ** 2) / (a_out - a_mid))
    if not (0.0 < r_m < zeta and zeta < r_p < alpha):
        return {'class': -1, 'reason': 'invalid matching radii'}
    C_in = 0.0
    C_mid = (a_in - a_mid) * r_m ** 2 / 2.0 - b_mid * np.log(r_m) + C_in
    C_out = (a_mid - a_out) * r_p ** 2 / 2.0 + (b_mid - b_out) * np.log(r_p) + C_mid

    def Pot(r):
        if r < r_m:
            return 0.5 * a_in * r ** 2 + C_in
        if r < r_p:
            return 0.5 * a_mid * r ** 2 + b_mid * np.log(r) + C_mid
        return 0.5 * a_out * r ** 2 + b_out * np.log(r) + C_out
    V_alpha = Pot(alpha)
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
        try:
            value = np.linalg.det(MatrixM(lam))
        except Exception:
            return np.nan
        return value
    lam_vals = np.linspace(1e-05, lam_search_max, N_lam_search)
    det_vals = np.empty_like(lam_vals)
    for i, lam in enumerate(lam_vals):
        try:
            det_vals[i] = detM(lam)
        except Exception:
            det_vals[i] = np.nan
    intervals = []
    for i in range(len(lam_vals) - 1):
        f1 = det_vals[i]
        f2 = det_vals[i + 1]
        if not (np.isfinite(f1) and np.isfinite(f2)):
            continue
        if f1 * f2 < 0.0:
            intervals.append((lam_vals[i], lam_vals[i + 1]))
    raw_roots = []
    for left, right in intervals:
        try:
            root = brentq(detM, left, right, xtol=1e-12, rtol=1e-12, maxiter=300)
            raw_roots.append(root)
        except Exception:
            pass
    raw_roots = sorted(set((round(root, 10) for root in raw_roots if np.isfinite(root))))
    physical_roots = [root for root in raw_roots if abs(root - round(root)) > 1e-06]
    if len(physical_roots) < 2:
        return {'class': -1, 'reason': 'fewer than two physical roots', 'V_alpha': V_alpha}
    new_roots = np.asarray(physical_roots[:2], dtype=float)

    def integrate_piecewise(func, upper=Rmax, epsabs=1e-09, epsrel=1e-09):
        total = 0.0
        for left, right in [(0.0, r_m), (r_m, r_p), (r_p, upper)]:
            value = integrate.quad(func, left, right, limit=400, epsabs=epsabs, epsrel=epsrel)[0]
            total += value
        return total

    def coefficients_from_svd(lam):
        matrix = MatrixM(lam)
        _, singular_values, vh = np.linalg.svd(matrix)
        coeffs = vh[-1, :].copy()
        if coeffs[0] < 0.0:
            coeffs *= -1.0
        return coeffs

    def R(lam, r, A_in_value, A_mid_value, B_mid_value, B_out_value):
        if r < r_m:
            return A_in_value * Phi1_inM(lam, r)
        if r < r_p:
            return A_mid_value * Phi1_midM(lam, r) + B_mid_value * Phi1_midU(lam, r)
        return B_out_value * Phi1_outU(lam, r)
    A_in_values = np.zeros(2)
    A_mid_values = np.zeros(2)
    B_mid_values = np.zeros(2)
    B_out_values = np.zeros(2)
    for j, lam in enumerate(new_roots):
        coeffs = coefficients_from_svd(lam)
        A_in_values[j], A_mid_values[j], B_mid_values[j], B_out_values[j] = coeffs
        norm_integral = integrate_piecewise(lambda r: r * R(lam, r, A_in_values[j], A_mid_values[j], B_mid_values[j], B_out_values[j]) ** 2)
        if not np.isfinite(norm_integral) or norm_integral <= 0.0:
            return {'class': -1, 'reason': 'normalization failed', 'V_alpha': V_alpha}
        scale = np.sqrt(2.0 * np.pi * norm_integral)
        A_in_values[j] /= scale
        A_mid_values[j] /= scale
        B_mid_values[j] /= scale
        B_out_values[j] /= scale
    Z_cache = {}

    def Z(beta_value):
        key = round(float(beta_value), 12)
        if key not in Z_cache:
            Z_cache[key] = 2.0 * np.pi * integrate_piecewise(lambda r: r * np.exp(-beta_value * Pot(r)))
        return Z_cache[key]

    def raw_mode_projection(j, beta_prime):
        lam = new_roots[j]
        integral = integrate_piecewise(lambda r: r * R(lam, r, A_in_values[j], A_mid_values[j], B_mid_values[j], B_out_values[j]) * np.exp((beta / 2.0 - beta_prime) * Pot(r)))
        return 2.0 * np.pi * integral / Z(beta_prime)
    final_projection = np.array([raw_mode_projection(j, beta) for j in range(2)])

    def mode_amplitude(j, beta_ini):
        return raw_mode_projection(j, beta_ini) - final_projection[j]

    def a2_of_beta(beta_ini):
        return mode_amplitude(0, beta_ini)
    peak_result = minimize_scalar(lambda x: -a2_of_beta(x), bounds=(beta_min, beta_max), method='bounded', options={'xatol': 1e-05})
    if not peak_result.success:
        return {'class': -1, 'reason': 'peak optimization failed', 'V_alpha': V_alpha}
    beta_peak = peak_result.x
    a2_peak = a2_of_beta(beta_peak)
    boundary_margin = 0.03
    if beta_peak <= beta_min + boundary_margin or beta_peak >= beta_max - boundary_margin:
        return {'class': 0, 'reason': 'no Mpemba effect: no interior a2 peak', 'beta_peak': beta_peak, 'a2_peak': a2_peak, 'lambda_2': new_roots[0], 'lambda_3': new_roots[1], 'V_alpha': V_alpha}
    db = 0.01
    beta_left = max(beta_min, beta_peak - db)
    beta_right = min(beta_max, beta_peak + db)
    a2_left = a2_of_beta(beta_left)
    a2_right = a2_of_beta(beta_right)
    if not (a2_peak > a2_left and a2_peak > a2_right):
        return {'class': 0, 'reason': 'no Mpemba effect: a2 extremum is not a local maximum', 'beta_peak': beta_peak, 'a2_peak': a2_peak, 'lambda_2': new_roots[0], 'lambda_3': new_roots[1], 'V_alpha': V_alpha}
    a3_peak = mode_amplitude(1, beta_peak)
    beta_candidates = np.linspace(beta_min, beta_max, N_beta_search)
    best_F = np.inf
    best_beta_other = np.nan
    best_Delta2 = np.nan
    best_Delta3 = np.nan
    crossing_exists = False
    for beta_other in beta_candidates:
        if abs(beta_other - beta_peak) < 0.01:
            continue
        a2_other = mode_amplitude(0, beta_other)
        a3_other = mode_amplitude(1, beta_other)
        Delta2 = a2_other ** 2 - a2_peak ** 2
        Delta3 = a3_other ** 2 - a3_peak ** 2
        if abs(Delta2) < 1e-12:
            continue
        F_value = Delta3 / Delta2
        if not np.isfinite(F_value):
            continue
        if F_value < best_F:
            best_F = F_value
            best_beta_other = beta_other
            best_Delta2 = Delta2
            best_Delta3 = Delta3
        if F_value < -1.0:
            crossing_exists = True
    if crossing_exists:
        if beta_peak < beta:
            phase_class = 2
            reason = 'normal Mpemba effect'
        elif beta_peak > beta:
            phase_class = 1
            reason = 'inverse Mpemba effect'
        else:
            phase_class = 0
            reason = 'no Mpemba effect: beta_peak = beta'
    else:
        phase_class = 0
        reason = 'no Mpemba effect: crossing condition not satisfied'
    return {'class': phase_class, 'reason': reason, 'beta_peak': beta_peak, 'beta_other': best_beta_other, 'a2_peak': a2_peak, 'a3_peak': a3_peak, 'best_F': best_F, 'Delta2': best_Delta2, 'Delta3': best_Delta3, 'lambda_2': new_roots[0], 'lambda_3': new_roots[1], 'V_alpha': V_alpha, 'r_m': r_m, 'r_p': r_p}
kmid_abs_list = np.linspace(0.05, 2.0, 40)
alpha_list = np.linspace(1.05, 3.0, 40)
phase = np.full((len(alpha_list), len(kmid_abs_list)), -1, dtype=int)
beta_peak_map = np.full(phase.shape, np.nan)
best_F_map = np.full(phase.shape, np.nan)
V_alpha_map = np.full(phase.shape, np.nan)
lambda2_map = np.full(phase.shape, np.nan)
lambda3_map = np.full(phase.shape, np.nan)
best_beta_other_map = np.full(phase.shape, np.nan)
for ia, alpha_value in enumerate(alpha_list):
    for ik, kmid_abs in enumerate(kmid_abs_list):
        try:
            result = solve_parameter_set(kmid_abs, alpha_value, verbose=False)
            phase[ia, ik] = result['class']
            beta_peak_map[ia, ik] = result.get('beta_peak', np.nan)
            best_F_map[ia, ik] = result.get('best_F', np.nan)
            V_alpha_map[ia, ik] = result.get('V_alpha', np.nan)
            lambda2_map[ia, ik] = result.get('lambda_2', np.nan)
            lambda3_map[ia, ik] = result.get('lambda_3', np.nan)
            best_beta_other_map[ia, ik] = result.get('beta_other', np.nan)
        except Exception as error:
            phase[ia, ik] = -1
rows = []
for ia, alpha_value in enumerate(alpha_list):
    for ik, kmid_abs in enumerate(kmid_abs_list):
        rows.append([kmid_abs, alpha_value, phase[ia, ik], beta_peak_map[ia, ik], best_beta_other_map[ia, ik], best_F_map[ia, ik], lambda2_map[ia, ik], lambda3_map[ia, ik], V_alpha_map[ia, ik]])
rows = np.asarray(rows, dtype=float)
np.savetxt('Fig10.dat', rows, header='-k_mid alpha class beta_peak beta_other_best F_min lambda_2 lambda_3 V_alpha', fmt='%.16e')
