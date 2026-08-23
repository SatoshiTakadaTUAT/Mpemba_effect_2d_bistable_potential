import numpy as np
import scipy.special as sc
from scipy import integrate
from scipy.optimize import brentq, minimize_scalar
T = 1.0
beta = 1.0 / T
Rmax = 100.0
beta_min = 0.1
beta_max = 3.0
N_beta_search = 250
lam_search_max = 15.0
N_lam_search = 5000
number_of_modes = 2

def potential_at_alpha(kmid_abs, kout, alpha):
    k_in = 1.0
    k_mid = -kmid_abs
    k_out = kout
    zeta = 1.0
    if kmid_abs <= 0.0 or kout <= 0.0:
        return np.nan
    if alpha <= zeta:
        return np.nan
    b_in = 0.0
    b_mid = -k_mid * zeta ** 2
    b_out = -k_out * alpha ** 2
    r_m = zeta * np.sqrt(-k_mid / (k_in - k_mid))
    r_p_squared = (k_out * alpha ** 2 - k_mid * zeta ** 2) / (k_out - k_mid)
    if r_p_squared <= 0.0:
        return np.nan
    r_p = np.sqrt(r_p_squared)
    if not (0.0 < r_m < zeta and zeta < r_p < alpha):
        return np.nan
    C_in = 0.0
    C_mid = (k_in - k_mid) * r_m ** 2 / 2.0 - b_mid * np.log(r_m) + C_in
    C_out = (k_mid - k_out) * r_p ** 2 / 2.0 + (b_mid - b_out) * np.log(r_p) + C_mid
    V_alpha = 0.5 * k_out * alpha ** 2 + b_out * np.log(alpha) + C_out
    return V_alpha

def find_alpha_case2(kmid_abs, kout, alpha_min=1.001, alpha_max=8.0, Nscan=500):
    alpha_scan = np.linspace(alpha_min, alpha_max, Nscan)
    values = np.array([potential_at_alpha(kmid_abs, kout, a) for a in alpha_scan])
    intervals = []
    for i in range(len(alpha_scan) - 1):
        f1 = values[i]
        f2 = values[i + 1]
        if not (np.isfinite(f1) and np.isfinite(f2)):
            continue
        if f1 * f2 < 0.0:
            intervals.append((alpha_scan[i], alpha_scan[i + 1]))
    if len(intervals) == 0:
        return np.nan
    left, right = intervals[0]
    try:
        alpha_root = brentq(lambda a: potential_at_alpha(kmid_abs, kout, a), left, right, xtol=1e-12, rtol=1e-12)
    except Exception:
        return np.nan
    return alpha_root

def solve_case2_parameter_set(kmid_abs, kout, verbose=False):
    alpha = find_alpha_case2(kmid_abs, kout)
    if not np.isfinite(alpha):
        return {'class': -1, 'reason': 'no alpha satisfying V(alpha)=0'}
    a_in = 1.0
    a_mid = -float(kmid_abs)
    a_out = float(kout)
    zeta = 1.0
    b_in = 0.0
    b_mid = -a_mid * zeta ** 2
    b_out = -a_out * alpha ** 2
    r_m = zeta * np.sqrt(-a_mid / (a_in - a_mid))
    r_p = np.sqrt((a_out * alpha ** 2 - a_mid * zeta ** 2) / (a_out - a_mid))
    if not (0.0 < r_m < zeta and zeta < r_p < alpha):
        return {'class': -1, 'reason': 'invalid matching radii', 'alpha': alpha}
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
            return np.linalg.det(MatrixM(lam))
        except Exception:
            return np.nan
    lam_vals = np.linspace(1e-05, lam_search_max, N_lam_search)
    det_vals = np.array([detM(lam) for lam in lam_vals])
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
        return {'class': -1, 'reason': 'fewer than two physical roots', 'alpha': alpha, 'V_alpha': V_alpha}
    new_roots = np.asarray(physical_roots[:2])

    def integrate_piecewise(func, upper=Rmax):
        total = 0.0
        for left, right in [(0.0, r_m), (r_m, r_p), (r_p, upper)]:
            total += integrate.quad(func, left, right, limit=400, epsabs=1e-09, epsrel=1e-09)[0]
        return total

    def coefficients_from_svd(lam):
        matrix = MatrixM(lam)
        _, _, vh = np.linalg.svd(matrix)
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
        norm = integrate_piecewise(lambda r: r * R(lam, r, A_in_values[j], A_mid_values[j], B_mid_values[j], B_out_values[j]) ** 2)
        if not np.isfinite(norm) or norm <= 0.0:
            return {'class': -1, 'reason': 'normalization failed', 'alpha': alpha}
        scale = np.sqrt(2.0 * np.pi * norm)
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
        return {'class': -1, 'reason': 'peak search failed', 'alpha': alpha}
    beta_peak = peak_result.x
    a2_peak = a2_of_beta(beta_peak)
    if beta_peak <= beta_min + 0.03 or beta_peak >= beta_max - 0.03:
        return {'class': 0, 'reason': 'no interior peak', 'alpha': alpha, 'beta_peak': beta_peak, 'lambda_2': new_roots[0], 'lambda_3': new_roots[1], 'V_alpha': V_alpha}
    db = 0.01
    a2_left = a2_of_beta(beta_peak - db)
    a2_right = a2_of_beta(beta_peak + db)
    if not (a2_peak > a2_left and a2_peak > a2_right):
        return {'class': 0, 'reason': 'no local maximum', 'alpha': alpha, 'beta_peak': beta_peak, 'lambda_2': new_roots[0], 'lambda_3': new_roots[1], 'V_alpha': V_alpha}
    a3_peak = mode_amplitude(1, beta_peak)
    beta_candidates = np.linspace(beta_min, beta_max, N_beta_search)
    best_F = np.inf
    best_beta_other = np.nan
    crossing_exists = False
    for beta_other in beta_candidates:
        if abs(beta_other - beta_peak) < 0.01:
            continue
        if beta_peak < beta and beta_other >= beta:
            continue
        if beta_peak > beta and beta_other <= beta:
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
        if F_value < -1.0:
            crossing_exists = True
    if crossing_exists:
        if beta_peak < beta:
            phase_class = 3
            reason = 'Mpemba criterion satisfied'
        else:
            phase_class = 2
            reason = 'inverse Mpemba criterion satisfied'
    else:
        phase_class = 1
        reason = 'peak exists but no crossing'
    return {'class': phase_class, 'reason': reason, 'alpha': alpha, 'V_alpha': V_alpha, 'beta_peak': beta_peak, 'beta_other': best_beta_other, 'best_F': best_F, 'lambda_2': new_roots[0], 'lambda_3': new_roots[1]}
kmid_abs_list = np.linspace(0.05, 2.0, 40)
kout_list = np.linspace(0.05, 2.0, 40)
phase = np.full((len(kout_list), len(kmid_abs_list)), -1, dtype=int)
alpha_map = np.full(phase.shape, np.nan)
beta_peak_map = np.full(phase.shape, np.nan)
best_F_map = np.full(phase.shape, np.nan)
for io, kout in enumerate(kout_list):
    for im, kmid_abs in enumerate(kmid_abs_list):
        try:
            result = solve_case2_parameter_set(kmid_abs, kout)
            phase[io, im] = result['class']
            alpha_map[io, im] = result.get('alpha', np.nan)
            beta_peak_map[io, im] = result.get('beta_peak', np.nan)
            best_F_map[io, im] = result.get('best_F', np.nan)
        except Exception as error:
            phase[io, im] = -1
rows = []
for io, kout in enumerate(kout_list):
    for im, kmid_abs in enumerate(kmid_abs_list):
        rows.append([kmid_abs, kout, phase[io, im], alpha_map[io, im], beta_peak_map[io, im], best_F_map[io, im]])
np.savetxt('Fig13.dat', np.asarray(rows), header='-k_mid k_out class alpha beta_peak F_min', fmt='%.16e')
