import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, minimize_scalar

T = 1.0
beta = 1.0 / T

k_in  = 1.0
k_mid = -0.5
k_out = 0.8

xi    = 1.0
alpha = 3.0

b_in  = 0.0
b_mid = -k_mid * xi**2
b_out = -k_out * alpha**2

r_m = xi * np.sqrt(-k_mid / (k_in - k_mid))
r_p = np.sqrt((k_out * alpha**2 - k_mid * xi**2) / (k_out - k_mid))

C_in = 0.0
C_mid = ((k_in - k_mid) * r_m**2 / 2.0 - b_mid * np.log(r_m) + C_in)
C_out = ((k_mid - k_out) * r_p**2 / 2.0 + (b_mid - b_out) * np.log(r_p) + C_mid)

EPS = 1.0e-6
R_SHOOT = 15.0
N_R = 6000

beta_min = 0.10
beta_max = 3.00
N_beta = 600

ROOT_BRACKETS = [(0.80, 1.10), (1.80, 2.20),]

def Pot(r):
    r = np.asarray(r)
    ans = np.empty_like(r, dtype=float)

    mask_in = r < r_m
    mask_mid = (r >= r_m) & (r < r_p)
    mask_out = r >= r_p

    ans[mask_in] = (0.5 * k_in * r[mask_in]**2 + C_in)

    rr = r[mask_mid]
    ans[mask_mid] = (0.5 * k_mid * rr**2 + b_mid * np.log(rr) + C_mid)

    rr = r[mask_out]
    ans[mask_out] = (0.5 * k_out * rr**2 + b_out * np.log(rr) + C_out)

    return ans

def Pot_scalar(r):
    if r < r_m:
        return 0.5 * k_in * r**2 + C_in

    if r < r_p:
        return (0.5 * k_mid * r**2 + b_mid * np.log(r) + C_mid)

    return (0.5 * k_out * r**2 + b_out * np.log(r) + C_out)

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

def radial_rhs(r, y, lam):

    phi, dphi = y

    return [dphi, -dphi / r - (lam - VS_scalar(r)) / T * phi,]

def integrate_left(lam, dense_output=False):

    c = -(lam + k_in) / (4.0 * T)

    y0 = [1.0 + c * EPS**2, 2.0 * c * EPS,]

    return solve_ivp(lambda r, y: radial_rhs(r, y, lam), (EPS, r_p), y0, method="DOP853", rtol=1.0e-11, atol=1.0e-13, max_step=0.03, dense_output=dense_output,)

def integrate_right(lam, dense_output=False):

    power = nu_out - 2.0 * mu_out(lam)

    phi_R = 1.0
    dphi_R = (power / R_SHOOT - gamma_out * R_SHOOT) * phi_R

    return solve_ivp(lambda r, y: radial_rhs(r, y, lam), (R_SHOOT, r_p), [phi_R, dphi_R], method="DOP853", rtol=1.0e-11, atol=1.0e-13, max_step=0.03, dense_output=dense_output,)

def shooting_residual(lam):

    left = integrate_left(lam, dense_output=False,).y[:, -1]

    right = integrate_right(lam, dense_output=False,).y[:, -1]

    return (left[0] * right[1] - left[1] * right[0])

eigenvalues = []

for left, right in ROOT_BRACKETS:

    root = brentq(shooting_residual, left, right, xtol=1.0e-12, rtol=1.0e-12, maxiter=200,)

    eigenvalues.append(root)

eigenvalues = np.asarray(eigenvalues)

lambda_2 = eigenvalues[0]
lambda_3 = eigenvalues[1]

print("Shooting eigenvalues")
print("==============================")
print(f"lambda_2 = {lambda_2:.15f}")
print(f"lambda_3 = {lambda_3:.15f}")

r_grid = np.linspace(EPS, R_SHOOT, N_R,)

V_grid = Pot(r_grid)

def eigenfunction_on_grid(lam):

    left_sol = integrate_left(lam, dense_output=True,)

    right_sol = integrate_right(lam, dense_output=True,)

    left_rp = left_sol.sol(r_p)
    right_rp = right_sol.sol(r_p)

    if abs(right_rp[0]) > abs(right_rp[1]):

        scale_right = (left_rp[0] / right_rp[0])

    else:

        scale_right = (left_rp[1] / right_rp[1])

    phi = np.empty_like(r_grid)

    mask_left = (r_grid <= r_p)

    phi[mask_left] = (left_sol.sol(r_grid[mask_left])[0])

    phi[~mask_left] = (scale_right * right_sol.sol(r_grid[~mask_left])[0])

    norm = (2.0 * np.pi * np.trapz(r_grid * phi**2, r_grid,))

    phi /= np.sqrt(norm)

    if phi[0] < 0.0:
        phi *= -1.0

    return phi

phi2 = eigenfunction_on_grid(lambda_2)

phi3 = eigenfunction_on_grid(lambda_3)

def Z(beta_value):

    return (2.0 * np.pi * np.trapz(r_grid * np.exp(-beta_value * V_grid), r_grid,))

def raw_projection(
    phi,
    beta_ini,
):

    integrand = (r_grid * phi * np.exp((beta / 2.0 - beta_ini) * V_grid))

    return (2.0 * np.pi * np.trapz(integrand, r_grid,) / Z(beta_ini))

proj2_eq = raw_projection(phi2, beta,)

proj3_eq = raw_projection(phi3, beta,)

def a2(beta_ini):

    return (raw_projection(phi2, beta_ini,) - proj2_eq)

def a3(beta_ini):

    return (raw_projection(phi3, beta_ini,) - proj3_eq)

peak_result = minimize_scalar(lambda b: -a2(b), bounds=(beta_min, beta_max,), method="bounded", options={"xatol": 1.0e-9,},)

beta_star = peak_result.x

a2_star = a2(beta_star)
a3_star = a3(beta_star)

beta_sharp = np.linspace(beta_min, beta_max, N_beta,)

a2_sharp = np.array([a2(b) for b in beta_sharp])

a3_sharp = np.array([a3(b) for b in beta_sharp])

Delta2 = (a2_sharp**2 - a2_star**2)

Delta3 = (a3_sharp**2 - a3_star**2)

F = np.full_like(beta_sharp, np.nan,)

mask = (np.abs(Delta2) > 1.0e-12)

F[mask] = (Delta3[mask] / Delta2[mask])

output = np.column_stack((beta_sharp, a2_sharp, a3_sharp, Delta2, Delta3, F,))

crossing_mask = (np.isfinite(F) & (F < -1.0))

plt.rcParams["mathtext.fontset"] = "cm"

plt.rcParams["font.size"] = 14

fig, ax = plt.subplots(figsize=(7.0, 5.0))

ax.plot(beta_sharp, F, linewidth=2.5,)

ax.fill_between(beta_sharp, F, -1.0, where=(np.isfinite(F) & (F < -1.0)), interpolate=True, alpha=0.25,)

ax.axhline(-1.0, linestyle="--", linewidth=1.5, label=r"$\mathcal{F}=-1$",)

ax.axvline(beta_star, color="black", linestyle=":", linewidth=2.5, label=(rf"$\beta_{{\rm ini}}^*" rf"={beta_star:.3f}$"),)

ax.set_xlim(1, 3,)
ax.set_ylim(-1e3, 2e2,)

ax.set_xlabel(r"$\beta_{\rm ini}^{\sharp}$", fontsize=22,)

ax.set_ylabel(r"$\mathcal{F}$", fontsize=22,)

ax.tick_params(direction="in", labelsize=16,)

ax.legend(fontsize=13, frameon=True,)

fig.tight_layout()

fig.savefig("Fig8.eps", bbox_inches="tight",)
