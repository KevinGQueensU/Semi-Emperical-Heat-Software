import scipy as sp
from Beam import Beam
from Medium import Atom, Medium
from BoundaryConditions import BoundaryConditions
import Simulation as sim
import numpy as np
import matplotlib
matplotlib.use("TkAgg")   # or "TkAgg" if you don't have Qt installed
import matplotlib.pyplot as plt
plt.ion()


rho = 7.9255 # g/cm^3

Z_C = 6
A_C = 12.011 # g/mol
C = Atom('C', Z_C, A_C, 0.12 / 100)

Z_Fe = 26
A_Fe = 55.847
Fe = Atom('Fe', Z_Fe, A_Fe, 88.97 / 100)

Z_Cr = 24
A_Cr = 51.996
Cr = Atom('Cr', Z_Cr, A_Cr, 8.91 / 100)

Z_W = 74
A_W = 183.85
W = Atom('W', Z_W, A_W, 1.08 / 100)

Z_N = 7
A_N = 14.007
N = Atom('N', Z_N, A_N, 0.02 / 100)

Z_Mn = 25
A_Mn = 54.938
Mn = Atom('Mn', Z_Mn, A_Mn, 0.48 / 100)

Z_Ta = 73
A_Ta = 180.95
Ta = Atom('Ta', Z_Ta, A_Ta, 0.14 / 100)

Z_P = 15
A_P = 30.974
P = Atom('P', Z_P, A_P, 0.005 / 100)

Z_S = 16
A_S = 32.066
S = Atom('S', Z_S, A_S, 0.004 / 100)

Z_V = 23
A_V = 50.942
V = Atom('V', Z_V, A_V, 0.2 / 100)

Z_Al = 13
A_Al = 26.982
Al = Atom('Al', Z_Al, A_Al, 0.009 / 100)

Z_Si = 14
A_Si = 28.086
Si = Atom('Si', Z_Si, A_Si, 0.04 / 100)

Z_Ni = 28
A_Ni = 58.69
Ni = Atom('Ni', Z_Ni, A_Ni, 0.02 / 100)

Z_Ti = 22
A_Ti = 47.9
Ti = Atom('Ti', Z_Ti, A_Ti, 0.006 / 100)


I_0 = 1.40e16  # [s^-1]
E_0 = 50e6  # [MeV]
r = 28e-3  # m
sig_ga_y0 = r/np.sqrt(2)
sig_ga_z0 = r/np.sqrt(2)
Z = 1

beam = Beam(E_0, I_0, Z, sig_ga_y0=sig_ga_y0, sig_ga_z0=sig_ga_z0)

Lx = 1e-3
dx = 0.2e-3
Ly = 67e-3
dy = 0.5e-3
Lz = 67e-3
dz = 0.5e-3

medium = Medium(rho, [C, Fe, Cr, W, N, Mn, Ta, P, S, V, Al, Si, Ni, Ti], Lx, Ly, Lz, "EuroFer97//eurofer97.txt", beam)

if False:
    xk = np.linspace(0, Lx, 10000) # cell edges
    cell_width = xk[1] - xk[0]
    cj = 0.5*(xk[:-1] + xk[1:]) # cell centers
    alpha = beta = 0.0
    y = z = 0

    E_beam = np.empty(xk.size + 1)  #
    dIdx = np.empty(xk.size + 1)
    E_inst = np.empty(xk.size + 1)
    I_beam = np.empty(xk.size + 1)
    dEb_dx = np.zeros_like(xk)  # eV/(m·s)
    dEdx = np.zeros_like(xk)
    dEdx_beam = np.zeros_like(xk)
    phi_free = np.array([beam.PD(x, y, z, alpha, beta) for x in xk]) # Free energy flux eV/(m^2·s)

    E_beam[0] = E_0*I_0
    E_inst[0] = E_0 # eV
    I_beam[0] = I_0    # 1/s

    for k, j in enumerate(cj):
        # Get energy gradient
        dEdx[k] = medium.get_dEdx(E_inst[k])
        dEdx_beam[k] = dEdx[k] * I_beam[k] + E_inst[k] * dIdx[k]
        I_beam[k + 1] = I_beam[k]
        E_beam[k + 1] = max(E_beam[k] + dEdx_beam[k] * cell_width, 0)
        E_inst[k + 1] = E_beam[k+1]/I_beam[k+1]

    print("initial finished")
    dx = cell_width
    N = len(cj)
    dEb_dx_test = np.zeros_like(dEb_dx)
    for k in range(N):
        dEb_dx[k] -= I_beam[k] * dEdx[k]

    eV_to_J = 1.602176634e-19
    dEb_dx_W = dEb_dx * eV_to_J  # W/m
    dEb_dx_kW_mm = dEb_dx_W / 1e6  # kW/mm
    plt.figure(figsize=(6, 4))
    plt.plot(cj * 1e3, dEb_dx_kW_mm[1:], label='Semi Empirical Model', color='black')
    plt.xlabel('Distance (mm)')
    plt.ylabel('Energy Deposition Gradient (kW/mm)')
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.title("dI/dx")
    plt.xlabel('Distance [mm]')
    plt.ylabel(r'$\frac{dI}{dx}$  $[s\cdot mm]^{-1}$')
    plt.plot(xk * 1e3, dIdx[1:] * 1e-3, 'ko', markersize=3)
    plt.show()

    plt.title("Instantaneous Beam Energy")
    plt.xlabel('xs [mm]')
    plt.ylabel('E [MeV]')
    plt.plot(xk * 1e3, E_inst[1:] * 1e-6)
    plt.show()

    plt.title("Beam Energy")
    plt.xlabel('xs [mm]')
    plt.ylabel(r'$E_{beam}$ [MeV/s]')
    plt.plot(xk * 1e3, E_beam[1:] * 1e-6)
    plt.show()

    plt.title("Beam Intensity")
    plt.xlabel('xs [mm]')
    plt.ylabel(r'I [/s]')
    plt.plot(xk * 1e3, I_beam[1:] * 1e-6)
    plt.show()

from fipy.tools import numerix as nx

def Cp(T):
    Cp_fit = 2.696*T - 0.00496*T**2 + 3.335e-6*T**3
    return nx.maximum(Cp_fit, 0.0)


def k(T):
    k_out= T*(0.190706 - 4.3053e-4*T + 3.817e-7*T**2 - 1.158e-10*T**3)

    return nx.maximum(k_out, 28.0)

if True:

    # Parameters for eurofer97
    rho *= 1e3  # kg/m^3
    BC = BoundaryConditions(['Fixed', 'Fixed', 'None', 'None', 'BBR', 'BBR'],
                            273, 298, 0.9)

    Ts, ts = sim.heateq_solid_3d(beam, medium, BC,
                                 Lx, Ly, Lz,
                                 rho, Cp, k,
                                 5,
                                 T0=298,
                                 dx=dx, dy=dy, dz=dz, view=True,
                                 dT_target=100,
                                 dt=1e-5, dt_ramp=1.5)
    #%%

    plt.close()
    plt.plot(ts, Ts, color='b', label='Center Temperature')
    plt.xlabel('Time [s]', size='x-large')
    plt.ylabel('Temperature [K]', size='x-large')
    plt.legend()
    plt.grid()
    plt.show(block=True)