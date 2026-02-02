import numpy as np

from Beam import Beam
from Medium import Atom, Medium
from BoundaryConditions import BoundaryConditions
import Simulation as sim
from fipy.tools import numerix as nx

I_0 = 3.12e12 # [s^-1] 0.5 muAmps
E_0 = 5e6  # [eV]
r = 1e-6 # m
sig_ga_y0 = r
sig_ga_z0 = r
Z = 1
beam = Beam(E_0, I_0, Z, sig_ga_y0=sig_ga_y0, sig_ga_z0=sig_ga_z0, dim = 2)

Lx_Ol = 50e-6
Ly_Ol = 10e-4
Lz_Ol = Ly_Ol

Ly = Ly_Ol
Lz = Lz_Ol
dx = 2e-6
dy = 0.5e-4
dz = 0.5e-4

def C_olivine(T):
    M = 0.14069  # kg/mol
    T_safe = nx.clip(T, 200, 3000)

    a, b, c, d, e = 87.36, 8.717e-2, -3.699e6, 8.436e2, -2.237e-5

    Cp = a + b * T_safe + c / (T_safe ** 2) + d / (T_safe ** 0.5) + e * (T_safe ** 2)
    return nx.maximum(Cp/M, 10.0)

def k_olivine(T, Tmin=250.0, Tmax=3000.0):
    T = np.asarray(T, dtype=float)
    T_safe = np.clip(T, Tmin, Tmax)

    xi = 89.0
    eta = 0.20
    return 418.4 / (xi + eta * (T_safe - 300.0))

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
                x0 = 0, name = 'OL')

# Actual Sim
mediums = [olivine]

BC = BoundaryConditions(['Fixed', 'BBR', 'None', 'BBR'], 273, 298, eps = 0.9)
ts, Ts = sim.heateq_solid_2d(beam, mediums, BC, Ly,10000, dt =  1e-12, alpha = 0.1, beta = 0.1,
                             dx=dx, dy=dy, view=True, dt_ramp=2, dT_target = 500, x_units='µm')