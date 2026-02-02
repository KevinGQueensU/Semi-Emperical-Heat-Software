from Beam import Beam
from Medium import Atom, Medium
from BoundaryConditions import BoundaryConditions
import Simulation as sim
from fipy.tools import numerix as nx

I_0 = 3.12e12 # [s^-1] 0.5 muAmps
E_0 = 4.4e6  # [MeV]
r = 1e-4 # m
sig_ga_y0 = r
sig_ga_z0 = r
Z = 1
beam = Beam(E_0, I_0, Z, sig_ga_y0=sig_ga_y0, sig_ga_z0=sig_ga_z0)

Lx_Ol = 100e-6
Ly_Ol = 20e-3
Lz_Ol = 15e-3

Lx_Pb = 200e-6
Ly_Pb = 100e-3
Lz_Pb = 15e-3

Ly = Ly_Pb
Lz = Lz_Pb
dx = 1e-6
dy = 2e-4


def C_Pb(T):
    M_Pb = 0.2072  # kg/mol
    # Prevent T from going to extreme values that break polynomials
    T_safe = nx.clip(T, 200, 2100)

    cond1 = (T_safe < 600.6)
    cond2 = (T_safe >= 600.6) & (T_safe < 2019.022)
    cond3 = (T_safe >= 2019.022)

    val1 = 25.01450 + 5.441836 * T_safe + 4.061367 * T_safe ** 2 + (-1.236217) * T_safe ** 3 + (-0.010657) / (
                T_safe ** 2)
    val2 = 38.00449 - 14.62249 * T_safe + 7.255475 * T_safe ** 2 - 1.033370 * T_safe ** 3 - 0.3309775 / (T_safe ** 2)

    t_limit = 2019.022
    val3 = 38.00449 - 14.62249 * t_limit + 7.255475 * t_limit ** 2 - 1.033370 * t_limit ** 3 - 0.3309775 / (
                t_limit ** 2)

    Cp_molar = cond1 * val1 + cond2 * val2 + cond3 * val3
    # Return J/kg*K and ensure it's always positive
    return nx.maximum(Cp_molar / M_Pb, 10.0)

# https://link.springer.com/article/10.1007/BF00514474
# https://inis.iaea.org/records/n4gzz-mpp69
def k_Pb(T):
    # Safety clip to avoid negative conductivity if T spikes
    T_safe = nx.clip(T, 100, 2500)

    cond1 = (T_safe < 600.6)
    cond2 = (T_safe >= 600.6) & (T_safe < 1300)
    cond3 = (T_safe >= 1300)

    # Solid Lead: k decreases with T
    val1 = 39.7 - 0.0138 * T_safe
    # Liquid Lead: k increases with T
    val2 = 9.15 + 0.0114 * T_safe
    # Limit: Stay constant at values above 1300K
    val3 = 9.15 + 0.0114 * 1300

    # Combine them
    k_combined = (cond1 * val1) + (cond2 * val2) + (cond3 * val3)

    # Final safety floor (k should never be 0 or negative)
    return nx.maximum(k_combined* 1e2, 1.0)

# Define lead
Z_Pb = 82
A_Pb = 207.2 #g/mol
Pb = Atom('Pb', Z_Pb, A_Pb, 1)
rho_Pb = 11.34e3 # kg/m^3
x0_Pb = 0
lead = Medium(rho_Pb, C_Pb, k_Pb,
              Pb, Lx_Pb, Ly_Pb, Lz_Pb,
              "Pb//Hydrogen in Lead.txt", x0=x0_Pb, name = 'Lead')


def C_olivine(T):
    M = 0.14069  # kg/mol
    T_safe = nx.clip(T, 200, 3000)

    a, b, c, d, e = 87.36, 8.717e-2, -3.699e6, 8.436e2, -2.237e-5

    Cp = a + b * T_safe + c / (T_safe ** 2) + d / (T_safe ** 0.5) + e * (T_safe ** 2)
    return nx.maximum(Cp / M, 10.0)

def k_olivine(T, k0=1.7, *, Tmin=160, Tmax=6000.0, kmin=1e-4, kmax=50.0):
    # ensure float and keep T in a numerically safe range
    Tf = nx.clip(1.0 * T, Tmin, Tmax)
    k = k0 * nx.exp(-(Tf - 298.0) / 300.0)

    return nx.clip(k, kmin, kmax)

rho_olivine = 3.3e3  # kg/m^3

# Parameters for the medium and values
Z_Mg = 12
A_Mg = 24.305  # g/mol
Mg = Atom('Mg', Z_Mg, A_Mg, 0.2222, )

Z_Fe = 26
A_Fe = 55.845  # g/mol
Fe = Atom('Fe', Z_Fe, A_Fe, 0.2222)

Z_Si = 14
A_Si = 28.086  # g/mol
Si = Atom('Si', Z_Si, A_Si, 0.1111)

Z_O = 8
A_O = 15.999  # g/mol
O = Atom('O', Z_O, A_O, 0.4444)

olivine = Medium(rho_olivine, C_olivine, k_olivine,
                [Mg, Fe, Si, O],
                Lx_Ol, Ly_Ol, Lz_Ol,
                "Olivine//Hydrogen in Olivine.txt",
                x0 = Lx_Pb, name = 'Olivine')

# Actual Sim
mediums = [lead, olivine]

BC = BoundaryConditions(['BBR', 'BBR', 'BBR', 'BBR'],
                        273, 298, eps=0.9)
ts, Ts = sim.heateq_solid_2d(beam, mediums, BC, Ly,10000, dt =  1e-6, alpha = 0, beta = 0,
                             dx=dx, dy=dy, view=True, dt_ramp=2, dT_target = 500, x_units='µm')