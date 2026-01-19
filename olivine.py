import scipy as sp
from Beam import Beam
from Medium import Atom, Medium
import Simulation as sim
import numpy as np
import matplotlib.pyplot as plt
from fipy.tools import numerix as nx
##OLIVINE
from fipy.tools import numerix as nx

def C_f(T):
    # constants as floats
    a = 87.36
    b = 8.717e-2
    c = -3.699e6
    d = 8.436e2
    e = -2.237e-5
    M = 0.14069  # kg/mol (forsterite)

    # ensure floating FiPy variable inside the expression
    Tf = T + 0.0

    # avoid negative integer exponents: use reciprocals with float powers
    Cp = a + b * Tf + c * (1.0 / (Tf**2.0)) + d * (1.0 / (Tf**0.5)) + e * (Tf**2.0)

    # guard against tiny negatives from roundoff
    return nx.maximum(Cp / M, 0.0)

def k_f(T, k0=1.7, *, Tmin=160, Tmax=6000.0, kmin=1e-4, kmax=50.0):
    # ensure float and keep T in a numerically safe range
    Tf = nx.clip(1.0 * T, Tmin, Tmax)

    # your original law, evaluated on the safe Tf
    k = k0 * nx.exp(-(Tf - 298.0) / 300.0)

    # keep operator strictly elliptic & avoid overflow
    return nx.clip(k, kmin, kmax)


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

I_0 = 6.24e11 * 5 # [s^-1]
E_0 = 8e6  # [MeV]
r = 1e-4  # m
sig_ga_y0 = r
sig_ga_z0 = r
Z = 1
x0 = 0
beam = Beam(E_0, I_0, Z, sig_ga_y0=sig_ga_y0, sig_ga_z0=sig_ga_z0)

rho = 3.3  # g/cm^3
Lx = 0.5e-3
dx = 0.5e-4
Ly = 1.0e-3
dy = 0.01e-3
Lz = Ly
medium = Medium(rho, [Mg, Fe, Si, O], Lx, Ly, Lz,"Olivine//Hydrogen in Olivine.txt", beam)

#%%
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
    dEb_dx_kW_mm = dEb_dx_W/1e6  # kW/mm
#%%

    plt.figure(figsize=(6, 4))
    plt.title("Energy Deposition Gradient")
    plt.plot(cj * 1e3, dEb_dx_kW_mm[1:])
    plt.xlabel('x [mm]')
    plt.ylabel(r'$\frac{dE_b}{dx}$ [kW/mm]')
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.title("Instantaneous Beam Energy")
    plt.xlabel('x [mm]')
    plt.ylabel('E [MeV]')
    plt.plot(xk * 1e3, E_inst[1:] * 1e-6)
    plt.show()


    plt.title("Beam Energy")
    plt.xlabel('x [mm]')
    plt.ylabel(r'$E_{beam}$ [MeV/s]')
    plt.plot(xk * 1e3, E_beam[1:] * 1e-6)
    plt.show()
#%%
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

fig, ax1 = plt.subplots()
plt.xlim(-10, 2000)

Se_elec = medium.Se_elec * 1e-6
Se_nucl = medium.Se_nucl * 1e-6
Es = medium.Es * 1e-3  # MeV

# Main plot (energy axis)
ax1.plot(Es[:-1], Se_elec, 'b--', label='Electronic Stopping')
ax1.plot(Es[:-1], Se_nucl, 'r--', label='Nuclear Stopping')
ax1.plot(0, 0, 'black', label='Full Stopping Power')
ax1.set_xlabel('Energy [keV]', size='x-large')
ax1.set_ylabel(r'Stopping Power $[\frac{\mathrm{keV}}{\mathrm{mm}}]$', size='x-large')
ax1.legend(loc='lower right')

# Second x-axis (depth)
ax2 = ax1.twiny()
ax2.set_xlim(-0.05, 0.4)
ax2.set_xlabel('Depth [mm]', size='x-large')
ax2.plot(xk * 1e3, -dEdx * 1e-6/medium.N_tot, 'black')

# ----------------- ZOOMED INSET ON NUCLEAR -----------------
# Choose energy window you want to zoom on (e.g. low energy region)
E1, E2 = 0.0, 1.0  # MeV, adjust as needed

mask = (Es[:-1] >= E1) & (Es[:-1] <= E2)

# Create inset axis inside ax1
axins = inset_axes(ax1, width="35%", height="35%", loc="center")

# Plot only the nuclear (and optionally electronic) in the inset
axins.plot(Es[:-1][mask], Se_nucl[mask], 'r--', label='Nuclear')
axins.set_xlim(-0.03, 0.05)
axins.set_ylim(Se_nucl[mask].min(), Se_nucl[mask].max()*1.1)

# Clean up ticks if you want a minimalist zoom box
axins.set_xticks([])
axins.set_yticks([])

# Draw a rectangle + connectors showing which region is zoomed
mark_inset(ax1, axins, loc1=2, loc2=3, fc="none", ec="0.5")
# -----------------------------------------------------------

fig.tight_layout()
plt.show()
#%%
# SHOWING BLACKBODY RADIATION
if False:
    # Parameters for
    rho_kg = rho * 1e3  # kg/m^3

    sim.heateq_solid_2d(beam, medium, Lx, Ly, rho_kg, C_f, k_f, 300, SE = 0,
                                 T0 = 500, T0_faces = [None, None, None, None],
                                 rad_bnd = [True, True, True, True], T_amb = 200,
                                 dx=dx, dy=dy, dt = 1e-6, dt_ramp = 2, dt_max = 1, view=True, dT_target = 50)
# 3D test
if False:
    sim.heateq_solid_3d(beam, medium, Lx, Ly, Lz, rho, C, k, 1000,
                                 T0 = 298, T0_faces = 298, rad_bnd = False,
                                 dx=dx, dy=dy, dz=dz, dt=0.000001, view=True, view_freq = 0.0001)
# 2D Constant Temp
if False:
    # Parameters for
    rho_kg = rho * 1e3  # kg/m^3

    sim.heateq_solid_2d(beam, medium, Lx, Ly, rho_kg, C_f, k_f, 600,
                        T0 = 298, T_amb = 298, T0_faces = [273, 273, 298, 298],
                        rad_bnd = [False, False, True, True], eps = 0.9,
                        dx = dx, dy = dy, view = True,
                        dt = 1e-6, dt_ramp = 1.5, dt_max = 1, x_units = 'mm', y_units = 'mm')

# 2D BB Radiation
if False:
    Lx = 10e-6
    dx = 0.05e-6
    Ly = 10e-6
    dy = 0.1e-6
    # Parameters for the beam and values
    I_0 = 6.24 * 1e11  # [s^-1]
    E_0 = 1e6  # [MeV]
    r = 0.5e-6  # m
    sig_ga_y0 = r
    sig_ga_z0 = r
    Z = 1
    x0 = 0
    beam = Beam(E_0, I_0, Z, sig_ga_y0=sig_ga_y0, sig_ga_z0=sig_ga_z0)

    # Parameters for the medium and values
    Z_Mg = 12
    A_Mg = 24.305  # g/mol
    Mg = Atom('Mg', Z_Mg, A_Mg, 0.2222, )

    Z_Fe = 9
    A_Fe = 18.998  # g/mol
    Fe = Atom('Mg', Z_Fe, A_Fe, 0.2222)

    Z_Si = 14
    A_Si = 28.086  # g/mol
    Si = Atom('Mg', Z_Si, A_Si, 0.1111)

    Z_O = 8
    A_O = 15.999  # g/mol
    O = Atom('Mg', Z_O, A_O, 0.4444)

    rho = 3.3  # g/cm^3
    n = 9.8e23 * 1e6  # 1/m^3
    W = H = 10e-6  # 10 mm square slab
    A_xsec = W * H
    P_perim = 2 * (W + H)
    medium = Medium(n, rho, [Mg, Fe, Si, O], Lx, A_xsec, P_perim, "Ni//Stopping_Power//H.txt", beam,
                    x0=x0)

    # Parameters for
    rho *= 1e3  # kg/m^3
    C = 850  # J/(kg*K)
    k = 1.7  # W/m*K

    sim.heateq_solid_2d(beam, medium, Lx, Ly, rho, C, k, 1000,
                                 T0 = 1, T0_faces = 1, rad_bnd = True,
                                 dx=dx, dy=dy, dt=0.000001, view=True, view_freq = 0.0001, res = 1e-9)