import scipy as sp
from Beam import Beam
from Medium import Atom, Medium
import Simulation as sim
import numpy as np
import matplotlib.pyplot as plt

I_0 = 6.24 * 1e15  # [s^-1]
E_0 = 200e6  # [MeV]
r = 1e-3  # m
sig_ga_y0 = r
sig_ga_z0 = r
Z = 1
x0 = 0
beam = Beam(E_0, I_0, Z, sig_ga_y0=sig_ga_y0, sig_ga_z0=sig_ga_z0)

Z_Ni = 28
A_Ni = 58.693  # g/mol
Ni = Atom('Ni', Z_Ni, A_Ni, 1)

rho = 8.908  # g/cm^3
Lx = 60e-3
Ly = 10e-3
Lz = 10e-3
medium = Medium(rho, Ni, Lx, Ly, Lz,"Ni//Stopping_Power//H.txt", beam)

##NICKEL
if True:
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
        dEdx_beam[k] = dEdx[k] * I_beam[k]
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
    eV_to_J = 1.602176634e-19
    dEb_dx_W = dEb_dx * eV_to_J  # W/m
    dEb_dx_kW_mm = dEb_dx_W / 1e6  # kW/mm
#%%
    fig, ax1 = plt.subplots()
    plt.xlim(-10, 200)
    Se_elec = medium.Se_elec * medium.N_tot * 1e-9
    Se_nucl = medium.Se_nucl * medium.N_tot * 1e-9
    Es = medium.Es * 1e-6
    plt.plot(Es[:-1], Se_elec, 'b--', label = 'Electronic Stopping')
    plt.plot(Es[:-1], Se_nucl, 'r--', label = 'Nuclear Stopping')
    plt.plot(0, 0, 'black', label='Full Stopping Power')
    plt.legend(loc='center')
    ax1.set_xlabel('Energy [MeV]', size='x-large')
    ax1.set_ylabel(r'Stopping Power $[\frac{MeV}{mm}]$', size='x-large')
    ax2 = ax1.twiny()
    ax2.set_xlim(-0.05, 0.4)
    ax2.set_xlabel('Depth [mm]', size='x-large')
    ax2.plot(xk * 1e3, -dEdx * 1e-9, 'black')
    fig.tight_layout()
    plt.show()
    #%%
    plt.figure(figsize=(6, 4))
    plt.plot(cj * 1e3, dEb_dx_kW_mm[1:], label='Semi Empirical Model', color='black')
    plt.xlabel('Distance (mm)')
    plt.ylabel('Energy Deposition Gradient (kW/mm)')
    plt.legend()
    plt.tight_layout()
    plt.xlim(0, Lx*1e3)
    plt.yticks(np.arange(min(dEb_dx_kW_mm), max(dEb_dx_kW_mm) + 1, 6))
    plt.show()
    print(np.trapz(dEb_dx[:-1], cj)/I_0)
#%%
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

if False:
    Lx = 60e-3
    dx = 1e-3
    Ly = 40e-3
    dy = 0.1e-3
    # Parameters for the beam and values
    I_0 = 6.24 * 1e15 # [s^-1]
    E_0 = 200e6 # [MeV]
    r = 1e-3 # m
    sig_ga_y0 = r
    sig_ga_z0 = r
    Z = 1
    x0 = 0
    beam = Beam(E_0, I_0, Z, sig_ga_y0 = sig_ga_y0, sig_ga_z0 = sig_ga_z0)

    # Parameters for the medium and values
    Z_Ni = 28
    A_Ni = 58.693  # g/mol
    Ni = Atom('Ni', Z_Ni, A_Ni, 1, "Ni//Cross_Sections//Ni_px.txt")
    rho = 8.908  # g/cm^3
    n = 2.56e30 # 1/m^3
    H = 140e-3
    W = 10e-3  # 10 mm square slab
    A_xsec = W * H
    P_perim = 2 * (W + H)
    medium = Medium(n, rho, Ni, Lx, A_xsec, P_perim, "Ni//Stopping_Power//H.txt", beam,
                    x0 = x0)

    # Parameters for
    rho *= 1e3 # kg/m^3
    C = 445 # J/(kg*K)
    k = 91 # W/m*K

    ts, Ts = sim.heateq_solid_2d(beam, medium, Lx, Ly, rho, C, k, 10,
                        dx = dx, dy = dy, view = True, dt_fact = 0.01)