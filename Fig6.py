import numpy as np
import scipy.special as sc
from scipy import integrate
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ============================================================
# Parameters
# ============================================================
T = 1.0
beta = 1.0 / T

a_in, a_mid, a_out = 1.0, -0.5, 0.8
zeta, alpha = 1.0, 3.0

# A moderate cutoff is sufficient because the outer solution decays rapidly.
# Using Rmax=100 tends to cause numerical overflow/underflow in high-order
# hypergeometric functions.
Rmax = 10.0

b_in = 0.0
b_mid = -a_mid * zeta**2
b_out = -a_out * alpha**2

r_m = zeta * np.sqrt(-a_mid / (a_in - a_mid))
r_p = np.sqrt((a_out * alpha**2 - a_mid * zeta**2) / (a_out - a_mid))

C_in = 0.0
C_mid = (a_in - a_mid) * r_m**2 / 2.0 - b_mid * np.log(r_m) + C_in
C_out = (a_mid - a_out) * r_p**2 / 2.0 + (b_mid - b_out) * np.log(r_p) + C_mid


def Pot(r):
    if r < r_m:
        return 0.5 * a_in * r**2 + C_in
    if r < r_p:
        return 0.5 * a_mid * r**2 + b_mid * np.log(r) + C_mid
    return 0.5 * a_out * r**2 + b_out * np.log(r) + C_out


# ============================================================
# Effective potential and Kummer parameters
# ============================================================
def VS_region(r, a, b):
    return (
        -a
        + a * b / (2.0 * T)
        + a**2 * r**2 / (4.0 * T)
        + b**2 / (4.0 * T * r**2)
    )


def VS_mid(r):
    return VS_region(r, a_mid, b_mid)


nu_in = abs(b_in) / (2.0 * T)
nu_mid = abs(b_mid) / (2.0 * T)
nu_out = abs(b_out) / (2.0 * T)

gamma_in = abs(a_in) / (2.0 * T)
gamma_mid = abs(a_mid) / (2.0 * T)
gamma_out = abs(a_out) / (2.0 * T)


def mu_value(lam, a, b, nu, gamma):
    return (
        (1.0 + nu) / 2.0
        - beta * (lam + a - a * b / (2.0 * T)) / (4.0 * gamma)
    )


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


# ============================================================
# Inner regular solution and its derivative
# ============================================================
def Phi1_inM(lam, r):
    z = gamma_in * r**2
    return r**nu_in * np.exp(-z / 2.0) * M(mu_in(lam), 1.0 + nu_in, z)


def Phi3_inM(lam, r):
    z = gamma_in * r**2
    return r**nu_in * np.exp(-z / 2.0) * M(1.0 + mu_in(lam), 2.0 + nu_in, z)


def Phi2_inM(lam, r):
    return (
        (nu_in - gamma_in * r**2) * Phi1_inM(lam, r)
        + 2.0 * mu_in(lam) / (1.0 + nu_in) * gamma_in * r**2 * Phi3_inM(lam, r)
    ) / r


# ============================================================
# Outer decaying solution and its derivative
# ============================================================
def Phi1_outU(lam, r):
    z = gamma_out * r**2
    return r**nu_out * np.exp(-z / 2.0) * U(mu_out(lam), 1.0 + nu_out, z)


def Phi3_outU(lam, r):
    z = gamma_out * r**2
    return r**nu_out * np.exp(-z / 2.0) * U(1.0 + mu_out(lam), 2.0 + nu_out, z)


def Phi2_outU(lam, r):
    return (
        (nu_out - gamma_out * r**2) * Phi1_outU(lam, r)
        - 2.0 * mu_out(lam) * gamma_out * r**2 * Phi3_outU(lam, r)
    ) / r



# ============================================================
# Middle-region analytic basis
#
# These functions are used only to extract the coefficients
# A_mid and B_mid after an eigenvalue has been obtained.
# The actual eigenfunction itself is still propagated through
# the middle region by solve_ivp, as below.
# ============================================================
def Phi1_midM(lam, r):
    z = gamma_mid * r**2
    return (
        r**nu_mid
        * np.exp(-z / 2.0)
        * M(mu_mid(lam), 1.0 + nu_mid, z)
    )


def Phi3_midM(lam, r):
    z = gamma_mid * r**2
    return (
        r**nu_mid
        * np.exp(-z / 2.0)
        * M(1.0 + mu_mid(lam), 2.0 + nu_mid, z)
    )


def Phi2_midM(lam, r):
    return (
        (nu_mid - gamma_mid * r**2) * Phi1_midM(lam, r)
        + 2.0 * mu_mid(lam) / (1.0 + nu_mid)
        * gamma_mid * r**2 * Phi3_midM(lam, r)
    ) / r


def Phi1_midU(lam, r):
    z = gamma_mid * r**2
    return (
        r**nu_mid
        * np.exp(-z / 2.0)
        * U(mu_mid(lam), 1.0 + nu_mid, z)
    )


def Phi3_midU(lam, r):
    z = gamma_mid * r**2
    return (
        r**nu_mid
        * np.exp(-z / 2.0)
        * U(1.0 + mu_mid(lam), 2.0 + nu_mid, z)
    )


def Phi2_midU(lam, r):
    return (
        (nu_mid - gamma_mid * r**2) * Phi1_midU(lam, r)
        - 2.0 * mu_mid(lam)
        * gamma_mid * r**2 * Phi3_midU(lam, r)
    ) / r


# ============================================================
# Middle region: direct ODE integration
# This avoids the M/U basis degeneracy near lambda = 2.
# ============================================================
def radial_ode_mid(r, y, lam):
    phi, dphi = y
    ddphi = -dphi / r + (beta * VS_mid(r) - beta * lam) * phi
    return [dphi, ddphi]


def build_mode(lam):
    """Return an unnormalized eigenfunction callable for a supplied eigenvalue.

    The inner solution is the regular Kummer-M solution.  It is propagated
    through the middle region by solve_ivp.  The outer amplitude is fixed by
    continuity of phi at r_p.  For a true eigenvalue the derivative also
    matches automatically.
    """
    phi_rm = Phi1_inM(lam, r_m)
    dphi_rm = Phi2_inM(lam, r_m)

    sol_mid = solve_ivp(
        lambda r, y: radial_ode_mid(r, y, lam),
        (r_m, r_p),
        [phi_rm, dphi_rm],
        method="DOP853",
        rtol=1.0e-11,
        atol=1.0e-13,
        dense_output=True,
    )
    if not sol_mid.success:
        raise RuntimeError(f"Middle-region integration failed for lambda={lam}")

    phi_rp = sol_mid.sol(r_p)[0]
    outer_basis_rp = Phi1_outU(lam, r_p)
    if not np.isfinite(outer_basis_rp) or outer_basis_rp == 0.0:
        raise RuntimeError(f"Invalid outer basis at lambda={lam}")
    B_out = phi_rp / outer_basis_rp

    # Matching diagnostic
    dphi_mid_rp = sol_mid.sol(r_p)[1]
    dphi_out_rp = B_out * Phi2_outU(lam, r_p)
    mismatch = dphi_mid_rp - dphi_out_rp

    def phi_raw(r):
        if r < r_m:
            return Phi1_inM(lam, r)
        if r < r_p:
            return float(sol_mid.sol(r)[0])
        return B_out * Phi1_outU(lam, r)

    return phi_raw, mismatch


# ============================================================
# Eigenvalues
# IMPORTANT: lambda = 1.999212561634259 is now included as the
# second nonstationary mode.  Subsequent mode indices are shifted.
# ============================================================
new_roots = np.array([
    0.981202658826,       # m=2
    1.999212561634259,    # m=3
    3.441568144643,       # m=4
    4.840317596314,       # m=5
    6.283510556951,       # m=6
    7.807205554006,       # m=7
    9.357564991303,       # m=8
    10.901533027304,      # m=9
    12.439665785837,      # m=10
    13.986470905770,      # m=11
], dtype=float)

number_of_modes = len(new_roots)  # 10 modes: m=2,...,11

print("Selected nonstationary eigenvalues")
for j, lam in enumerate(new_roots):
    print(f"m={j+2:2d}, lambda={lam:.15f}")


# ============================================================
# Build and normalize all modes
# ============================================================
def integrate_piecewise(func, upper=Rmax, epsabs=1.0e-9, epsrel=1.0e-9):
    total = 0.0
    for left, right in [(0.0, r_m), (r_m, r_p), (r_p, upper)]:
        total += integrate.quad(
            func,
            left,
            right,
            limit=500,
            epsabs=epsabs,
            epsrel=epsrel,
        )[0]
    return total


raw_modes = []
mode_mismatches = []
mode_norms = []

for j, lam in enumerate(new_roots):
    phi_raw, mismatch = build_mode(lam)
    mode_mismatches.append(mismatch)

    norm_sq = 2.0 * np.pi * integrate_piecewise(lambda r: r * phi_raw(r)**2)
    if not np.isfinite(norm_sq) or norm_sq <= 0.0:
        raise RuntimeError(f"Invalid norm for m={j+2}, lambda={lam}: {norm_sq}")

    scale = np.sqrt(norm_sq)
    mode_norms.append(scale)

    def make_normalized(raw_func, scale_value):
        return lambda r: raw_func(r) / scale_value

    raw_modes.append(make_normalized(phi_raw, scale))

    print(
        f"m={j+2:2d}, lambda={lam:.12f}, "
        f"matching mismatch={mismatch:+.3e}, norm scale={scale:.6e}"
    )


# ============================================================
# Extract normalized analytic coefficients
# A_in, A_mid, B_mid, B_out for m=2,...,11
#
# IMPORTANT:
# Near lambda ~= 2, the middle M/U basis is nearly degenerate.
# Therefore A_mid and B_mid for that mode can be very large
# and sensitive to numerical precision.  The eigenfunction itself,
# which is propagated by solve_ivp above, remains well behaved.
# ============================================================
A_in_values = np.zeros(number_of_modes)
A_mid_values = np.zeros(number_of_modes)
B_mid_values = np.zeros(number_of_modes)
B_out_values = np.zeros(number_of_modes)
mid_condition_numbers = np.zeros(number_of_modes)

for j, lam in enumerate(new_roots):
    scale = mode_norms[j]

    # Inner amplitude:
    # the raw inner solution was chosen with A_in = 1.
    A_in_values[j] = 1.0 / scale

    # Normalized value and derivative at r_m.
    phi_rm = Phi1_inM(lam, r_m) / scale
    dphi_rm = Phi2_inM(lam, r_m) / scale

    # Determine A_mid and B_mid from continuity at r_m.
    Mmid = np.array([
        [Phi1_midM(lam, r_m), Phi1_midU(lam, r_m)],
        [Phi2_midM(lam, r_m), Phi2_midU(lam, r_m)],
    ], dtype=float)

    rhs_mid = np.array([phi_rm, dphi_rm], dtype=float)

    mid_condition_numbers[j] = np.linalg.cond(Mmid)

    # lstsq is more robust than solve close to the M/U degeneracy.
    coeff_mid, _, _, _ = np.linalg.lstsq(
        Mmid,
        rhs_mid,
        rcond=None,
    )

    A_mid_values[j] = coeff_mid[0]
    B_mid_values[j] = coeff_mid[1]

    # Determine normalized B_out directly from the normalized
    # eigenfunction at r_p.
    phi_rp_normalized = raw_modes[j](r_p)
    B_out_values[j] = (
        phi_rp_normalized / Phi1_outU(lam, r_p)
    )


print("\n")
print("=" * 118)
print("Eigenvalues and normalized analytic coefficients")
print("=" * 118)
print(
    f"{'m':>3s} "
    f"{'lambda':>18s} "
    f"{'A_in':>18s} "
    f"{'A_mid':>18s} "
    f"{'B_mid':>18s} "
    f"{'B_out':>18s} "
    f"{'cond(mid)':>14s}"
)

for j, lam in enumerate(new_roots):
    print(
        f"{j+2:3d} "
        f"{lam:18.12e} "
        f"{A_in_values[j]:18.12e} "
        f"{A_mid_values[j]:18.12e} "
        f"{B_mid_values[j]:18.12e} "
        f"{B_out_values[j]:18.12e} "
        f"{mid_condition_numbers[j]:14.4e}"
    )


# Save the same table to a text file.
with open(
    "eigenvalues_and_coefficients_m2_m11.txt",
    "w",
    encoding="utf-8",
) as f:
    f.write(
        "# m lambda A_in A_mid B_mid B_out cond_mid\n"
    )

    for j, lam in enumerate(new_roots):
        f.write(
            f"{j+2:d}\t"
            f"{lam:.16e}\t"
            f"{A_in_values[j]:.16e}\t"
            f"{A_mid_values[j]:.16e}\t"
            f"{B_mid_values[j]:.16e}\t"
            f"{B_out_values[j]:.16e}\t"
            f"{mid_condition_numbers[j]:.16e}\n"
        )

print(
    "\nSaved: eigenvalues_and_coefficients_m2_m11.txt"
)


# ============================================================
# Partition function and equilibrium distribution
# ============================================================
def Z(beta_value):
    return 2.0 * np.pi * integrate_piecewise(
        lambda r: r * np.exp(-beta_value * Pot(r))
    )


def Peq(r, beta_value):
    return np.exp(-beta_value * Pot(r)) / Z(beta_value)


# ============================================================
# Expansion coefficients
# ============================================================
def raw_mode_projection(j, beta_prime):
    phi = raw_modes[j]
    integral = integrate_piecewise(
        lambda r: r * phi(r) * np.exp((beta / 2.0 - beta_prime) * Pot(r))
    )
    return 2.0 * np.pi * integral / Z(beta_prime)


def a_hat(j, beta_ini):
    # Explicit subtraction of the final equilibrium projection.
    return raw_mode_projection(j, beta_ini) - raw_mode_projection(j, beta)


# ============================================================
# Orthogonality diagnostics
# ============================================================
def overlap(i, j):
    return 2.0 * np.pi * integrate_piecewise(
        lambda r: r * raw_modes[i](r) * raw_modes[j](r)
    )


print("\nOverlap matrix for the first five nonstationary modes")
for i in range(min(5, number_of_modes)):
    print(" ".join(f"{overlap(i,j):+.4e}" for j in range(min(5, number_of_modes))))


# ============================================================
# Reconstruct P_ini for Ntr = 2, 6, 10
# ============================================================
beta_ini_compare = 0.82
Ntr_list = [2, 6, 10]
r_plot = np.linspace(1.0e-6, 5.0, 1200)


def P_ini(r, beta_ini):
    return np.exp(-beta_ini * Pot(r)) / Z(beta_ini)


# Compute amplitudes only once.
a_hat_values = np.array([a_hat(j, beta_ini_compare) for j in range(number_of_modes)])

print("\nExpansion coefficients at beta_ini =", beta_ini_compare)
for j, (lam, ah) in enumerate(zip(new_roots, a_hat_values)):
    print(f"m={j+2:2d}, lambda={lam:.12f}, a_hat={ah:+.12e}")


def P_truncated(r, Ntr):
    value = Peq(r, beta)
    for j in range(Ntr - 1):  # m=2,...,Ntr
        value += (
            np.exp(-beta * Pot(r) / 2.0)
            * a_hat_values[j]
            * raw_modes[j](r)
        )
    return value


P_ini_list = np.array([P_ini(r, beta_ini_compare) for r in r_plot])
P_truncated_lists = {
    Ntr: np.array([P_truncated(r, Ntr) for r in r_plot])
    for Ntr in Ntr_list
}


# ============================================================
# Save reconstruction data
# ============================================================
output_data = np.column_stack([
    r_plot,
    P_ini_list,
    P_truncated_lists[2],
    P_truncated_lists[6],
    P_truncated_lists[10],
])

np.savetxt(
    "Pini_with_lambda1999_Ntr2610.csv",
    output_data,
    delimiter=",",
    header="r,P_ini,Ntr2,Ntr6,Ntr10",
    comments="",
    fmt="%.16e",
)


# ============================================================
# Plot
# ============================================================
plt.figure(figsize=(8, 5))
plt.rcParams["mathtext.fontset"] = "cm"

plt.plot(r_plot, P_ini_list, linewidth=3, label=r"$P_{\rm ini}(r)$")

styles = ["--", "-.", ":"]
for Ntr, style in zip(Ntr_list, styles):
    plt.plot(
        r_plot,
        P_truncated_lists[Ntr],
        linewidth=2,
        linestyle=style,
        label=rf"$N_{{\rm tr}}={Ntr+1}$",
    )

plt.axvline(r_m, linestyle=":", linewidth=1.3)
plt.axvline(r_p, linestyle=":", linewidth=1.3)
plt.xlim(0.0, 5.0)
plt.xlabel(r"$r$", fontsize=24, math_fontfamily="cm")
plt.ylabel(r"$P(r,0)$", fontsize=24, math_fontfamily="cm")
plt.tick_params(labelsize=18)
plt.grid(alpha=0.4)
plt.legend(fontsize=16)
plt.tight_layout()
plt.savefig("Pini_with_lambda1999_Ntr2610.png", dpi=300, bbox_inches="tight")
plt.savefig("Pini_with_lambda1999_Ntr2610.eps", bbox_inches="tight")


# ============================================================
# Reconstruction errors
# ============================================================
print("\nReconstruction errors")
print("Ntr        max error          L2 error          normalization")

for Ntr in Ntr_list:
    P_N = P_truncated_lists[Ntr]
    error = P_N - P_ini_list
    max_error = np.max(np.abs(error))
    l2_error = np.sqrt(
        2.0 * np.pi * np.trapezoid(r_plot * error**2, r_plot)
    )
    normalization = 2.0 * np.pi * np.trapezoid(r_plot * P_N, r_plot)
    print(
        f"{Ntr:3d}   {max_error:.8e}   {l2_error:.8e}   {normalization:.12f}"
    )


# ============================================================
# Stable Parseval check
# ============================================================
def g_exact(r):
    # Algebraically equivalent to exp(beta V/2)*(P_ini-P_eq),
    # but avoids inf*0 overflow at large r.
    return (
        np.exp((beta / 2.0 - beta_ini_compare) * Pot(r)) / Z(beta_ini_compare)
        - np.exp(-beta * Pot(r) / 2.0) / Z(beta)
    )


g_norm_sq = 2.0 * np.pi * integrate_piecewise(lambda r: r * g_exact(r)**2)

print("\nParseval check")
print("||g||^2 =", g_norm_sq)

partial_sum = 0.0
for j, ah in enumerate(a_hat_values):
    partial_sum += ah**2
    residual = g_norm_sq - partial_sum
    print(
        f"Ntr={j+2:2d}, sum(a_m^2)={partial_sum:.12e}, residual={residual:.12e}"
    )

#plt.show()

# ============================================================
# Time evolution of P(r,t) - P_eq(r,beta)
# Ntr = 10, t = 0, 1, 2, 3
# ============================================================

# IMPORTANT:
# Use the same beta_ini as that used to calculate a_hat_values
beta_ini_evolution = beta_ini_compare

Ntr_evolution = 10
time_list = [0.0, 1.0, 2.0, 3.0]

r_plot_time = np.linspace(1.0e-6, 5.0, 1200)


# ------------------------------------------------------------
# Exact initial deviation
# ------------------------------------------------------------
def delta_P_ini(r, beta_ini):
    return (
        np.exp(-beta_ini * Pot(r)) / Z(beta_ini)
        - np.exp(-beta * Pot(r)) / Z(beta)
    )


# ------------------------------------------------------------
# Truncated spectral expansion
#
# P(r,t) - P_eq(r,beta)
# =
# exp[-beta V(r)/2]
# sum_{m=2}^{Ntr}
# a_hat_m varphi_m(r) exp[-lambda_m t]
# ------------------------------------------------------------
def delta_P_Ntr(r, t, Ntr=10):

    n_used = Ntr - 1

    if n_used > len(new_roots):
        raise ValueError(
            f"Ntr={Ntr} requires {n_used} modes, "
            f"but only {len(new_roots)} are available."
        )

    if n_used > len(a_hat_values):
        raise ValueError(
            f"Ntr={Ntr} requires {n_used} coefficients, "
            f"but only {len(a_hat_values)} a_hat values are available."
        )

    total = 0.0

    for j in range(n_used):

        lam = new_roots[j]

        # Expansion coefficient already calculated at beta_ini_compare
        ah = a_hat_values[j]

        # Normalized eigenfunction varphi_m(r)
        # In this version the eigenfunctions are stored as callables
        # in raw_modes; there are no A_in/A_mid/B_mid/B_out arrays.
        varphi = raw_modes[j](r)

        total += (
            ah
            * varphi
            * np.exp(-lam * t)
        )

    return (
        np.exp(-beta * Pot(r) / 2.0)
        * total
    )


# ============================================================
# Calculate exact initial deviation
# ============================================================

delta_P_ini_list = np.array([
    delta_P_ini(r, beta_ini_evolution)
    for r in r_plot_time
])


# ============================================================
# Calculate truncated time evolution
# ============================================================

delta_P_time = {}

for t in time_list:

    delta_P_time[t] = np.array([
        delta_P_Ntr(
            r,
            t,
            Ntr=Ntr_evolution,
        )
        for r in r_plot_time
    ])


# ============================================================
# Save data
# ============================================================

output_data = np.column_stack([
    r_plot_time,
    delta_P_ini_list,
    delta_P_time[0.0],
    delta_P_time[1.0],
    delta_P_time[2.0],
    delta_P_time[3.0],
])

np.savetxt(
    "deltaP_Ntr10_time_evolution.csv",
    output_data,
    delimiter=",",
    header="r,Pini_minus_Peq,t0,t1,t2,t3",
    comments="",
    fmt="%.16e",
)

print("Saved: deltaP_Ntr10_time_evolution.csv")


# ============================================================
# Plot
# ============================================================

plt.figure(figsize=(8, 5))

plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["legend.fontsize"] = 15


# Exact P_ini - P_eq
plt.plot(
    r_plot_time,
    delta_P_ini_list,
    linewidth=3,
    linestyle="-",
    label=r"$P_{\rm ini}(r)-P_{\rm eq}(r,\beta)$",
)


# Spectral expansion
line_styles = [
    "--",
    "-.",
    ":",
    (0, (5, 2)),
]

for t, style in zip(time_list, line_styles):

    plt.plot(
        r_plot_time,
        delta_P_time[t],
        linewidth=2.2,
        linestyle=style,
        label=rf"$t={t:g}$",
    )


# Matching radii
plt.axvline(
    r_m,
    linestyle=":",
    linewidth=1.3,
)

plt.axvline(
    r_p,
    linestyle=":",
    linewidth=1.3,
)

# zero line
plt.axhline(
    0.0,
    linestyle=":",
    linewidth=1.3,
)


plt.xlim(0, 5)

plt.xlabel(
    r"$r$",
    fontsize=24,
    math_fontfamily="cm",
)

plt.ylabel(
    r"$P(r,t)-P_{\rm eq}(r,\beta)$",
    fontsize=22,
    math_fontfamily="cm",
)

plt.tick_params(labelsize=18)

plt.grid(
    linestyle="solid",
    alpha=0.35,
)

plt.legend(
    loc="best",
    fontsize=15,
)

plt.tight_layout()

plt.savefig(
    "deltaP_Ntr10_time_evolution.eps",
    bbox_inches="tight",
)

plt.savefig(
    "deltaP_Ntr10_time_evolution.png",
    dpi=300,
    bbox_inches="tight",
)

#plt.show()

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# beta_ini range
# ============================================================
beta_ini_list = np.linspace(0.2, 3.0, 300)

# ============================================================
# Calculate a2, a3, a4, a5
#
# j=0 -> m=2
# j=1 -> m=3
# j=2 -> m=4
# j=3 -> m=5
#
# a_hat(j, beta_ini) must already be defined.
# ============================================================
a2 = np.array([a_hat(0, b) for b in beta_ini_list])
a3 = np.array([a_hat(1, b) for b in beta_ini_list])
a4 = np.array([a_hat(2, b) for b in beta_ini_list])
a5 = np.array([a_hat(3, b) for b in beta_ini_list])


# ============================================================
# Common plot settings
# ============================================================
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.size"] = 14


# ============================================================
# Fig. (a): a2
# ============================================================
fig, ax = plt.subplots(figsize=(6, 4.5))

ax.plot(
    beta_ini_list,
    a2,
    linewidth=2.5,
    color="red",
    label=r"$a_2$",
)

ax.axhline(
    0.0,
    color="black",
    linestyle=":",
    linewidth=1.2,
)

ax.axvline(
    beta,
    color="black",
    linestyle="--",
    linewidth=1.2,
)

ax.set_xlim(0.2, 3.0)

ax.set_xlabel(
    r"$\beta_{\rm ini}$",
    fontsize=22,
)

ax.set_ylabel(
    r"$a_2$",
    fontsize=22,
)

ax.tick_params(
    direction="in",
    labelsize=16,
)

ax.legend(
    fontsize=15,
    frameon=True,
)

fig.tight_layout()

fig.savefig(
    "a2_case1_new.eps",
    bbox_inches="tight",
)

fig.savefig(
    "a2_case1_new.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# Fig. (b): a3, a4, a5
# ============================================================
fig, ax = plt.subplots(figsize=(6, 4.5))

ax.plot(
    beta_ini_list,
    a3,
    linewidth=2.2,
    linestyle="--",
    color="blue",
    label=r"$a_3$",
)

ax.plot(
    beta_ini_list,
    a4,
    linewidth=2.2,
    linestyle=":",
    color="black",
    label=r"$a_4$",
)

ax.plot(
    beta_ini_list,
    a5,
    linewidth=2.2,
    linestyle="-.",
    color="magenta",
    label=r"$a_5$",
)

ax.axhline(
    0.0,
    color="black",
    linestyle=":",
    linewidth=1.2,
)

ax.axvline(
    beta,
    color="black",
    linestyle="--",
    linewidth=1.2,
)

ax.set_xlim(0.2, 3.0)

ax.set_xlabel(
    r"$\beta_{\rm ini}$",
    fontsize=22,
)

ax.set_ylabel(
    r"$a_m$",
    fontsize=22,
)

ax.tick_params(
    direction="in",
    labelsize=16,
)

ax.legend(
    fontsize=15,
    frameon=True,
)

fig.tight_layout()

fig.savefig(
    "a345_case1_new.eps",
    bbox_inches="tight",
)

fig.savefig(
    "a345_case1_new.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()