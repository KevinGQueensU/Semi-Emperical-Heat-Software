from Beam import Beam
from Medium import Atom, Medium
from BoundaryConditions import BoundaryConditions
import Simulation as sim

I_0 = 6.24 * 1e9  # [s^-1]
E_0 = 400e6  # [MeV]
r = 0.5e-5  # m
sig_ga_y0 = r
sig_ga_z0 = r
Z = 1
beam = Beam(E_0, I_0, Z, sig_ga_y0=sig_ga_y0, sig_ga_z0=sig_ga_z0)

# Define Nickel
Z_Ni = 28
A_Ni = 58.693  # g/mol
Ni = Atom('Ni', Z_Ni, A_Ni, 1)
rho_Ni = 8.908e4  # kg/m^3
C_Ni = 445  # J/(kg*K)
k_Ni = 91  # W/m*K
Lx_Ni = 50e-6
Ly_Ni = 150e-6
Lz_Ni = 150e-6
x0_Ni = 10e-6
nickel = Medium(rho_Ni, Ni, Lx_Ni, Ly_Ni, Lz_Ni,
                "Ni//Stopping_Power//H.txt", x0=x0_Ni, name = 'Nickel')
# Define lead
Z_Pb = 82
A_Pb = 207.2 #g/mol
Pb = Atom('Pb', Z_Pb, A_Pb, 1)
rho_Pb = 11.34e3 # kg/m^3
C_Pb = 128 # J/(kg*K)
k_Pb = 35 #W/m*K
Lx_Pb = 10e-6
Ly_Pb = 150e-6
Lz_Pb = 150e-6
x0_Pb = 0
lead = Medium(rho_Pb, Pb, Lx_Ni/2, Ly_Pb, Lz_Pb,
              "Pb//Hydrogen in Lead.txt", x0=x0_Pb, name = 'Lead')
lead2 = Medium(rho_Pb, Pb, Lx_Ni*2, Ly_Pb, Lz_Pb,
              "Pb//Hydrogen in Lead.txt", x0=Lx_Pb+Lx_Ni, name = 'Lead')
# Actual Sim
mediums = [lead, nickel, lead2]
rhos = [rho_Pb, rho_Ni, rho_Pb]
Cs = [C_Pb*100, C_Ni, C_Pb]
ks = [k_Pb, k_Ni, k_Pb]

Lx = Lx_Ni/2 + Lx_Ni + Lx_Ni*2
Ly = Ly_Pb
Lz = Lz_Pb
dx = 1e-6
dy = 2e-6
BC = BoundaryConditions(['Fixed', 'Fixed', 'Fixed', 'BBR'],
                        273, 298)
ts, Ts = sim.heateq_solid_2d(beam, mediums, BC, Lx, Ly, rhos, Cs, ks, 10,
                             dx=dx, dy=dy, view=True, dt_ramp=1.1, dT_target = 100)