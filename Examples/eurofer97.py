from Beam import Beam
from Medium import Atom, Medium
from BoundaryConditions import BoundaryConditions
import Simulation as sim
import numpy as np
from fipy.tools import numerix as nx

# Define material
rho = 7.750 * 1e3 # kg/m^3

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

def Cp(T):
    return nx.maximum(2.696*T - 4.96e-3*T**2 + 3.335e-6*T**3, 1.0)

def k(T):
    return nx.maximum(T*(0.190706 - 4.3053e-4*T + 3.817e-7*T**2 - 1.158e-10*T**3), 1.0)

Lx = 1e-3
dx = 0.1e-3
Ly = 67e-3
dy = 0.5e-3
Lz = 67e-3
dz = 0.5e-3

medium = Medium(rho, Cp, k,[C, Fe, Cr, W, N, Mn, Ta, P, S, V, Al, Si, Ni, Ti],
                "..//SRIMTables//H in eurofer97.txt",
                Lx=Lx, Ly=Ly, Lz=Lz)

# Define beam
I_0 = 1.40e16  # [s^-1]
E_0 = 50e6  # [MeV]
r = 28e-3  # m
Z = 1

beam = Beam(E_0, I_0, Z, 3, r=r)


if True:
    # Parameters for eurofer97
    BC = BoundaryConditions(types=['Fixed', 'Fixed', 'BBR', 'BBR', 'BBR', 'BBR'],
                            T0 = 273, T_amb= 298, eps = 0.9)

    ts, Ts = sim.heateq_solid_3d(beam, medium, BC,
                                 Ly, Lz,5,
                                 T0=298,
                                 dx=dx, dy=dy, dz=dz, view=True,
                                 dT_target=200,
                                 dt=1e-2, dt_ramp=1.5, debug=True,
                                 min_sweeps= 1, max_sweeps = 5)


    #%%
    import matplotlib.pyplot as plt
    # FROM THE ANSYS STUDY
    ts_ref = [-0.009578544061302681, 0.19636015325670497, 0.3544061302681992, 0.5890804597701149, 0.9961685823754789,
         1.431992337164751, 1.997126436781609, 2.926245210727969, 3.5392720306513406, 3.994252873563218]
    Ts_ref = [304.74631751227497, 605.8919803600654, 802.291325695581, 998.6906710310965, 1224.5499181669395,
         1398.0360065466448, 1574.795417348609, 1800.6546644844516, 1915.2209492635025, 1993.7806873977086]
    plt.figure(figsize=(8, 6))
    plt.plot(ts, Ts, color='b', label='Simulated Center Temperature')
    plt.plot(ts_ref, Ts_ref, color='r', ls='--', label ='Reference Center Temperature')
    plt.xlabel('Time [s]', size='xx-large')
    plt.ylabel('Temperature [K]', size='xx-large')
    plt.tick_params(labelsize=14)
    plt.axhline(900, label=r'Max Temperature of $C$ and $\rho$ Fits (900K) ', color='k', ls = '--')
    plt.legend(fontsize =14, loc='lower right')
    plt.grid()
    plt.show(block=True)

    #%%
    #### TESTING STUFF
    if False:
        xk = np.linspace(0, Lx, 10000)  # cell edges
        cell_width = xk[1] - xk[0]
        cj = 0.5 * (xk[:-1] + xk[1:])  # cell centers
        alpha = beta = 0.0
        y = z = 0

        E_beam = np.empty(xk.size + 1)  #
        dIdx = np.empty(xk.size + 1)
        E_inst = np.empty(xk.size + 1)
        I_beam = np.empty(xk.size + 1)
        dEb_dx = np.zeros_like(xk)  # eV/(m·s)
        dEdx = np.zeros_like(xk)
        dEdx_beam = np.zeros_like(xk)
        phi_free = np.array([beam.PD(x, y, z, alpha, beta) for x in xk])  # Free energy flux eV/(m^2·s)

        E_beam[0] = E_0 * I_0
        E_inst[0] = E_0  # eV
        I_beam[0] = I_0  # 1/s

        for k, j in enumerate(cj):
            # Get energy gradient
            dEdx[k] = medium.get_dEdx(E_inst[k])
            dEdx_beam[k] = dEdx[k] * I_beam[k] + E_inst[k] * dIdx[k]
            I_beam[k + 1] = I_beam[k]
            E_beam[k + 1] = max(E_beam[k] + dEdx_beam[k] * cell_width, 0)
            E_inst[k + 1] = E_beam[k + 1] / I_beam[k + 1]

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