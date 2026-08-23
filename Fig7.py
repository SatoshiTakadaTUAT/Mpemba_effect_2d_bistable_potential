import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, minimize_scalar

T = 1.0
beta = 1.0 / T

k_in, k_mid, k_out = 1.0, -0.5, 0.8
xi, alpha = 1.0, 3.0

b_in = 0.0
b_mid = -k_mid * xi**2
b_out = -k_out * alpha**2

r_m = xi * np.sqrt(-k_mid / (k_in - k_mid))
r_p = np.sqrt((k_out * alpha**2 - k_mid * xi**2) / (k_out - k_mid))

C_in = 0.0
C_mid = (k_in - k_mid) * r_m**2 / 2.0 - b_mid * np.log(r_m) + C_in
C_out = ((k_mid - k_out) * r_p**2 / 2.0 + (b_mid - b_out) * np.log(r_p) + C_mid)

NUMBER_OF_MODES = 10

R_SHOOT = 15.0
EPS = 1.0e-6

beta_ini_sharp = 1.646

def V(r):
    r = np.asarray(r)
    ans = np.empty_like(r, dtype=float)

    mask1 = r < r_m
    mask2 = (r >= r_m) & (r < r_p)
    mask3 = r >= r_p

    ans[mask1] = 0.5 * k_in * r[mask1]**2 + C_in

    rr = r[mask2]
    ans[mask2] = 0.5 * k_mid * rr**2 + b_mid * np.log(rr) + C_mid

    rr = r[mask3]
    ans[mask3] = 0.5 * k_out * rr**2 + b_out * np.log(rr) + C_out

    return ans

def V_scalar(r):
    if r < r_m:
        return 0.5 * k_in * r**2 + C_in
    if r < r_p:
        return 0.5 * k_mid * r**2 + b_mid * np.log(r) + C_mid
    return 0.5 * k_out * r**2 + b_out * np.log(r) + C_out

def VS_scalar(r):
    if r < r_m:
        k, b = k_in, b_in
    elif r < r_p:
        k, b = k_mid, b_mid
    else:
        k, b = k_out, b_out

    return (-k + k * b / (2.0 * T) + k**2 * r**2 / (4.0 * T) + b**2 / (4.0 * T * r**2))

gamma_out = abs(k_out) / (2.0 * T)
nu_out = abs(b_out) / (2.0 * T)

def mu_out(lam):
    return ((1.0 + nu_out) / 2.0 - beta * (lam + k_out - k_out * b_out / (2.0 * T)) / (4.0 * gamma_out))

def ode_rhs(r, y, lam):
    phi, dphi = y
    return [dphi, -dphi / r - (lam - VS_scalar(r)) / T * phi,]

def integrate_left(lam, dense_output=False):
    c = -(lam + k_in) / (4.0 * T)

    y0 = [1.0 + c * EPS**2, 2.0 * c * EPS,]

    return solve_ivp(lambda r, y: ode_rhs(r, y, lam), (EPS, r_p), y0, method="DOP853", rtol=1.0e-10, atol=1.0e-12, max_step=0.03, dense_output=dense_output,)

def integrate_right(lam, dense_output=False):
    power = nu_out - 2.0 * mu_out(lam)

    phi_R = 1.0
    dphi_R = (power / R_SHOOT - gamma_out * R_SHOOT) * phi_R

    return solve_ivp(lambda r, y: ode_rhs(r, y, lam), (R_SHOOT, r_p), [phi_R, dphi_R], method="DOP853", rtol=1.0e-10, atol=1.0e-12, max_step=0.03, dense_output=dense_output,)

def shooting_residual(lam):
    left = integrate_left(lam, dense_output=False).y[:, -1]
    right = integrate_right(lam, dense_output=False).y[:, -1]

    return (left[0] * right[1] - left[1] * right[0])

ROOT_BRACKETS = [(0.80, 1.10), (1.80, 2.20), (3.20, 3.70), (4.50, 5.20), (6.00, 6.60), (7.40, 8.20), (9.00, 9.70), (10.50, 11.30), (12.00, 12.80), (13.50, 14.40),]

def find_eigenvalues(number_of_modes):
    roots = []

    for left, right in ROOT_BRACKETS[:number_of_modes]:
        f_left = shooting_residual(left)
        f_right = shooting_residual(right)

        if f_left * f_right >= 0.0:
            raise RuntimeError(f"No sign change in bracket ({left}, {right}).")

        root = brentq(shooting_residual, left, right, xtol=1.0e-11, rtol=1.0e-11, maxiter=200,)

        roots.append(root)

    return np.array(roots)

eigenvalues = find_eigenvalues(NUMBER_OF_MODES)

print("\nEigenvalues obtained by the two-sided shooting method")
print("======================================================")
for j, lam in enumerate(eigenvalues):
    print(f"m={j+2:2d}  " f"lambda_{j+2} = {lam:.12f}")

print("\nThe previously missed eigenvalue is")
print(f"lambda_3 = {eigenvalues[1]:.12f}")

r_grid = np.linspace(EPS, R_SHOOT, 4000)

def eigenfunction_on_grid(lam):
    left_sol = integrate_left(lam, dense_output=True)
    right_sol = integrate_right(lam, dense_output=True)

    left_at_rp = left_sol.sol(r_p)
    right_at_rp = right_sol.sol(r_p)

    if abs(right_at_rp[0]) > abs(right_at_rp[1]):
        scale_right = left_at_rp[0] / right_at_rp[0]
    else:
        scale_right = left_at_rp[1] / right_at_rp[1]

    phi = np.empty_like(r_grid)

    mask_left = r_grid <= r_p
    mask_right = ~mask_left

    phi[mask_left] = left_sol.sol(r_grid[mask_left])[0]

    phi[mask_right] = (scale_right * right_sol.sol(r_grid[mask_right])[0])

    norm = 2.0 * np.pi * np.trapezoid(r_grid * phi**2, r_grid,)

    phi /= np.sqrt(norm)

    if phi[0] < 0.0:
        phi *= -1.0

    return phi

phi_modes = np.array([eigenfunction_on_grid(lam) for lam in eigenvalues])

V_grid = V(r_grid)

def Z(beta_value):
    return 2.0 * np.pi * np.trapezoid(r_grid * np.exp(-beta_value * V_grid), r_grid,)

def raw_projection(j, beta_ini):
    integrand = (r_grid * phi_modes[j] * np.exp((beta / 2.0 - beta_ini) * V_grid))

    return (2.0 * np.pi * np.trapezoid(integrand, r_grid) / Z(beta_ini))

def mode_amplitude(j, beta_ini):
    return (raw_projection(j, beta_ini) - raw_projection(j, beta))

def a2_of_beta(beta_ini):
    return mode_amplitude(0, beta_ini)

peak_result = minimize_scalar(lambda x: -a2_of_beta(x), bounds=(0.1, 3.0), method="bounded", options={"xatol": 1.0e-8},)

beta_ini_star = peak_result.x

print("\nPeak of a2(beta_ini)")
print("======================================================")
print(f"beta_ini_star = {beta_ini_star:.12f}")
print(f"a2_peak       = {a2_of_beta(beta_ini_star):+.12e}")
print(f"beta_ini_sharp = {beta_ini_sharp:.12f}")

a_star = np.array([mode_amplitude(j, beta_ini_star) for j in range(NUMBER_OF_MODES)])

a_sharp = np.array([mode_amplitude(j, beta_ini_sharp) for j in range(NUMBER_OF_MODES)])

print("\nMode amplitudes")
print("======================================================")
for j in range(NUMBER_OF_MODES):
    print(f"m={j+2:2d}, " f"lambda={eigenvalues[j]:.9f}, " f"a_sharp={a_sharp[j]:+.8e}, " f"a_star={a_star[j]:+.8e}")

Cstar = Z(beta)

def delta_D_quadratic_over_Cstar(t, number_of_modes):
    ans = np.zeros_like(t, dtype=float)
    for j in range(number_of_modes):
        Delta_m = a_sharp[j]**2 - a_star[j]**2
        ans += 0.5 * Delta_m * np.exp(-2.0 * eigenvalues[j] * t)
    return ans

phi2 = phi_modes[0]
phi3 = phi_modes[1]
lam2 = eigenvalues[0]
lam3 = eigenvalues[1]

def D_overlap(phi_l, phi_m, phi_n):
    integrand = (r_grid * np.exp(0.5 * beta * V_grid) * phi_l * phi_m * phi_n)
    return 2.0 * np.pi * Cstar**2 * np.trapezoid(integrand, r_grid)

def E_overlap(phi_k, phi_l, phi_m, phi_n):
    integrand = (r_grid * np.exp(beta * V_grid) * phi_k * phi_l * phi_m * phi_n)
    return 2.0 * np.pi * Cstar**3 * np.trapezoid(integrand, r_grid)

D222 = D_overlap(phi2, phi2, phi2)
D223 = D_overlap(phi2, phi2, phi3)
D233 = D_overlap(phi2, phi3, phi3)
D333 = D_overlap(phi3, phi3, phi3)

E2222 = E_overlap(phi2, phi2, phi2, phi2)
E2223 = E_overlap(phi2, phi2, phi2, phi3)
E2233 = E_overlap(phi2, phi2, phi3, phi3)
E2333 = E_overlap(phi2, phi3, phi3, phi3)
E3333 = E_overlap(phi3, phi3, phi3, phi3)

print('\nNonlinear overlap coefficients')
print('======================================================')
for name, val in [
    ('Cstar', Cstar), ('D222', D222), ('D223', D223),
    ('D233', D233), ('D333', D333), ('E2222', E2222),
    ('E2223', E2223), ('E2233', E2233),
    ('E2333', E2333), ('E3333', E3333)
]:
    print(f'{name:7s} = {val:+.12e}')

def DKL_nonlinear_two_mode(t, amplitudes):
    a2, a3 = amplitudes[0], amplitudes[1]

    D2 = 0.5 * Cstar * (a2**2 * np.exp(-2.0 * lam2 * t) + a3**2 * np.exp(-2.0 * lam3 * t))

    D3 = -(1.0/6.0) * (D222 * a2**3 * np.exp(-3.0 * lam2 * t) + 3.0 * D223 * a2**2 * a3 * np.exp(-(2.0*lam2 + lam3)*t) + 3.0 * D233 * a2 * a3**2 * np.exp(-(lam2 + 2.0*lam3)*t) + D333 * a3**3 * np.exp(-3.0 * lam3 * t))

    D4 = (1.0/12.0) * (E2222 * a2**4 * np.exp(-4.0 * lam2 * t) + 4.0 * E2223 * a2**3 * a3 * np.exp(-(3.0*lam2 + lam3)*t) + 6.0 * E2233 * a2**2 * a3**2 * np.exp(-2.0*(lam2 + lam3)*t) + 4.0 * E2333 * a2 * a3**3 * np.exp(-(lam2 + 3.0*lam3)*t) + E3333 * a3**4 * np.exp(-4.0 * lam3 * t))

    return D2 + D3 + D4

t = np.linspace(0.0, 5.0, 1200)

Delta_two = delta_D_quadratic_over_Cstar(t, 2)
Delta_four = delta_D_quadratic_over_Cstar(t, 4)
Delta_ten = delta_D_quadratic_over_Cstar(t, 10)
Delta_nonlinear = (DKL_nonlinear_two_mode(t, a_sharp) - DKL_nonlinear_two_mode(t, a_star)) / Cstar

Peq_grid = np.exp(-beta * V_grid) / Z(beta)

def reconstructed_probability(amplitudes, time, number_of_modes=10):
    correction = np.zeros_like(r_grid)
    for j in range(number_of_modes):
        correction += amplitudes[j] * phi_modes[j] * np.exp(-eigenvalues[j]*time)
    return Peq_grid + np.exp(-0.5*beta*V_grid) * correction

def DKL_direct(amplitudes, time, number_of_modes=10):
    P = reconstructed_probability(amplitudes, time, number_of_modes)
    if np.any(P <= 0.0):
        return np.nan
    integrand = r_grid * P * np.log(P / Peq_grid)
    return 2.0*np.pi*np.trapezoid(integrand, r_grid)

Delta_direct_ten = np.array([(DKL_direct(a_sharp, tt, 10) - DKL_direct(a_star, tt, 10))/Cstar for tt in t])

def first_crossing(t_values, y_values):
    finite = np.isfinite(y_values)
    for i in range(len(t_values)-1):
        if not (finite[i] and finite[i+1]):
            continue
        if y_values[i] == 0.0:
            return t_values[i]
        if y_values[i]*y_values[i+1] < 0.0:
            return t_values[i] - y_values[i]*(t_values[i+1]-t_values[i])/(y_values[i+1]-y_values[i])
    return np.nan

print('\nCrossing times')
print('======================================================')
print('two-mode quadratic :', first_crossing(t, Delta_two))
print('four-mode quadratic:', first_crossing(t, Delta_four))
print('ten-mode quadratic :', first_crossing(t, Delta_ten))
print('two-mode O(dP^4)   :', first_crossing(t, Delta_nonlinear))
print('ten-mode direct KL :', first_crossing(t, Delta_direct_ten))

plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.size'] = 14
fig, ax = plt.subplots(figsize=(8.0, 6.0))
ax.axhline(0.0, color='0.4', linestyle=':', linewidth=1.2)
ax.plot(t, Delta_two, linewidth=2.5, linestyle='-', label='Two-mode approx.')
ax.plot(t, Delta_ten, linewidth=2.0, linestyle='-.', label='Ten-mode approx.')
ax.plot(t, Delta_nonlinear, linewidth=2.2, linestyle=':', label=r'Higher-order approx.')
ax.set_xlabel(r'$t$', fontsize=24)
ax.set_ylabel(r'$\Delta D_{\rm KL}/C^*$', fontsize=24)
ax.set_xlim(0.0, 5.0)
ax.tick_params(direction='in', labelsize=18)
ax.legend(fontsize=13, frameon=True)
fig.tight_layout()
fig.savefig('Fig7.eps', bbox_inches='tight')

plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.size'] = 14
fig, ax = plt.subplots(figsize=(4.0, 3.0))
ax.axhline(0.0, color='0.4', linestyle=':', linewidth=1.2)
ax.plot(t, Delta_two, linewidth=2.5, linestyle='-')
ax.plot(t, Delta_ten, linewidth=2.0, linestyle='-.')
ax.plot(t, Delta_nonlinear, linewidth=2.2, linestyle=':')
ax.set_xlabel(r'$t$', fontsize=24)
ax.set_ylabel(r'$\Delta D_{\rm KL}/C^*$', fontsize=24)
ax.set_xlim(2.5, 3.5)
ax.set_ylim(-1e-9, 1e-9)
ax.tick_params(direction='in', labelsize=18)
fig.tight_layout()
fig.savefig('Fig7_zoom.eps', bbox_inches='tight')
plt.show()
