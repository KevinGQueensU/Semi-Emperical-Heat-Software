from Beam import Beam
from Medium import Atom, Medium
import numpy as np
import matplotlib.pyplot as plt

# Define the beam
I_0 = 6.24 * 1e15  # [s^-1]
E_0 = 200e6  # [MeV]
r = 1e-3  # m
sig_ga_y0 = r
sig_ga_z0 = r
Z = 1
x0 = 0
beam = Beam(E_0, I_0, Z, 3, sig_ga_y0=sig_ga_y0, sig_ga_z0=sig_ga_z0)

# Define the atom
Z_Ni = 28
A_Ni = 58.693  # g/mol
Ni = Atom('Ni', Z_Ni, A_Ni, 1)

# Define the medium
rho_Ni = 8.908*1e3  # kg/m^3
C_Ni = 444          # J/(kg*m^3)
k_Ni = 90.9         # W/(m*k)
Lx = 60e-3
Ly = 10e-3
Lz = 10e-3

medium = Medium(rho_Ni, C_Ni, k_Ni, Ni, "..//SRIMTables//H in Ni.txt",
                Lx, Ly, Lz)

##NICKEL TEST
if True:
    #### MANUALLY ITERATE TO GET ENERGY AND INTENSITY AND HEATING ####
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

    # Calculate dEb_dx (energy deposition gradient)
    dx = cell_width
    N = len(cj)
    dEb_dx_test = np.zeros_like(dEb_dx)
    for k in range(N):
        dEb_dx[k] -= I_beam[k] * dEdx[k]
    eV_to_J = 1.602176634e-19
    dEb_dx_W = dEb_dx * eV_to_J  # W/m
    dEb_dx_kW_mm = dEb_dx_W / 1e6  # kW/mm
#%%
    # Plot the stopping power components
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
    # Plot the energy deposition gradient
    plt.figure(figsize=(6, 4))
    plt.plot(cj * 1e3, dEb_dx_kW_mm[1:], label='Semi Empirical Model', color='black')
    plt.xlabel('Distance (mm)')
    plt.ylabel('Energy Deposition Gradient (kW/mm)')
    plt.legend()
    plt.tight_layout()
    plt.xlim(0, Lx*1e3)
    plt.show()
#%%
    # Plot the change in beam energy and intensity
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
