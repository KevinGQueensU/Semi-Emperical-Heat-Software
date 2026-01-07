import scipy as sp
from Beam import Beam
from Medium import Atom, Medium
import simulation as sim
import numpy as np
import matplotlib.pyplot as plt
##NICKEL
if False:
    I_0 = 6.24 * 1e15 # [s^-1]
    E_0 = 200e6 # [MeV]
    r = 1e-3 # m
    sig_ga_y0 = r
    sig_ga_z0 = r
    Z = 1
    x0 = 0
    beam = Beam(E_0, I_0, Z, sig_ga_y0 = sig_ga_y0, sig_ga_z0 = sig_ga_z0)

    Z_Ni = 28
    A_Ni = 58.693  # g/mol
    Ni = Atom('Ni', Z_Ni, A_Ni, 1, "Ni//Cross_Sections//Ni_px.txt")

    rho = 8.908  # g/cm^3
    n = 2.56e30 # 1/m^3
    W = H = 10e-3  # 10 mm square slab
    A_xsec = W * H
    P_perim = 2 * (W + H)
    L = 60e-3
    medium = Medium(n, rho, Ni, L, A_xsec, P_perim, "Ni//Stopping_Power//H.txt", beam,
                    x0 = x0)

    xk = np.linspace(0, L, 1000) # cell edges
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
    dIdx[0] = medium.get_dIdx(E_inst[0], I_beam[0])

    for k, j in enumerate(cj):
        # Get energy gradient
        dEdx[k] = medium.get_dEdx(E_inst[k])
        dEdx_beam[k] = dEdx[k] * I_beam[k] + E_inst[k] * dIdx[k]
        I_beam[k + 1] = I_beam[k] + dIdx[k] * cell_width
        E_beam[k + 1] = max(E_beam[k] + dEdx_beam[k] * cell_width, 0)
        E_inst[k + 1] = E_beam[k+1]/I_beam[k+1]
        dIdx[k+1] = medium.get_dIdx(E_inst[k+1], I_beam[k+1])  # (1/s)/m

    print("initial finished")
    dx = cell_width
    N = len(cj)
    dEb_dx_test = np.zeros_like(dEb_dx)
    for k in range(N):
        dEb_dx[k] -= I_beam[k] * dEdx[k]
    print(np.trapezoid(dEb_dx, xk))
    medium.set_LBD(E_inst[0])

    for j in range(N):
        # forward receivers k >= j
        for k in range(j, N):
            temp = medium._Ed_fwd(cj[j], cell_width, xk[k], E_inst[j], I_beam[j])
            if(np.isnan(temp) or temp < 0):
                temp = 0
            dEb_dx[k] += temp
            dEb_dx_test[k] += temp # eV/m
        # backward receivers k <= j-1
        for k in range(j - 1, -1, -1):
            temp =  medium._Ed_bwd(cj[j], cell_width, xk[k], E_inst[j], I_beam[j])
            if(np.isnan(temp) or temp < 0):
                temp = 0
            dEb_dx[k] += temp
            dEb_dx_test[k] += temp # eV/s

    # for j in range(N):
    #     medium.set_LBD(E_inst[j])
    #     # forward receivers k >= j
    #     for k in range(j+1, N):
    #         temp = medium._Ed_fwd_test(cj[j], xk[k], dx, E_inst[j], I_beam[j])
    #         if(np.isnan(temp) or temp < 0):
    #             temp = 0
    #         dEb_dx[k] += temp
    #         dEb_dx_test[k] += temp # eV/sMFORWG
    #     # backward receivers k <= j-1
    #     for k in range(j - 1, -1, -1):
    #         temp =  medium._Ed_bwd_test(cj[j], xk[k], dx, E_inst[j], I_beam[j])
    #         if(np.isnan(temp) or temp < 0):
    #             temp = 0
    #
    #         dEb_dx[k] += temp
    #         dEb_dx_test[k] += temp # eV/s

    eV_to_J = 1.602176634e-19
    dEb_dx_W = dEb_dx * eV_to_J  # W/m
    dEb_dx_kW_mm = dEb_dx_W / 1e6  # kW/mm
    eV_to_J = 1.602176634e-19
    dEb_dx_W = dEb_dx_test * eV_to_J  # W/m
    dEb_dx_kW_mm_test = dEb_dx_W / 1e6  # kW/mm
#%%

    plt.figure(figsize=(6, 4))
    plt.plot(cj * 1e3, dEb_dx_kW_mm[1:], label='Semi Empirical Model', color='black')
    plt.xlabel('Distance (mm)')
    plt.ylabel('Energy Deposition Gradient (kW/mm)')
    plt.legend()
    plt.tight_layout()
    plt.xlim(0, L*1e3)
    plt.show()
    plt.figure(figsize=(6, 4))
    plt.plot(cj * 1e3, dEb_dx_kW_mm_test[1:], label='Semi Empirical Model', color='black')
    plt.xlabel('Distance (mm)')
    plt.ylabel('Energy Deposition Gradient (kW/mm)')
    plt.legend()
    plt.xlim(0, L*1e3)
    plt.tight_layout()
    plt.show()
    #%%
    E = np.linspace(0, 100e6, 1000)
    Se = medium.get_Se_ev_m(E)
    plt.figure(figsize=(6, 4))
    plt.plot(E,  Se, label='Semi Empirical Model', color='black')
    plt.xlabel('Distance (mm)')
    plt.ylabel('Energy Deposition Gradient (kW/mm)')
    plt.legend()
    plt.tight_layout()
    plt.show()
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

##OLIVINE
# Stopping Powers
if True:
    I_0 = 6.24 * 1e11 # [s^-1]
    E_0 = 1e6 # [MeV]
    r = 1e-3 # m
    sig_ga_y0 = r
    sig_ga_z0 = r
    Z = 1
    x0 = 0
    beam = Beam(E_0, I_0, Z, sig_ga_y0 = sig_ga_y0, sig_ga_z0 = sig_ga_z0)

    Z_Mg = 12
    A_Mg = 24.305 # g/mol
    Mg = Atom('Mg', Z_Mg, A_Mg, 0.2222, )

    Z_Fe = 9
    A_Fe = 18.998 # g/mol
    Fe = Atom('Mg', Z_Fe, A_Fe, 0.2222)

    Z_Si = 14
    A_Si = 28.086 # g/mol
    Si = Atom('Mg', Z_Si, A_Si, 0.1111)

    Z_O = 8
    A_O = 15.999  # g/mol
    O = Atom('Mg', Z_O, A_O, 0.4444)

    rho = 3.3  # g/cm^3
    n = 9.8e23*1e6 # 1/m^3
    W = H = 10e-3  # 10 mm square slab
    A_xsec = W * H
    P_perim = 2 * (W + H)
    L = 7e-6
    medium = Medium(rho, [Mg, Fe, Si, O], L, A_xsec, P_perim, "Ni//Stopping_Power//H.txt", beam,
                    x0 = x0)

    xk = np.linspace(0, L, 10000) # cell edges
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

    # for j in range(N):
    #     medium.set_LBD(E_inst[j])
    #     # forward receivers k >= j
    #     for k in range(j+1, N):
    #         temp = medium._Ed_fwd_test(cj[j], xk[k], dx, E_inst[j], I_beam[j])
    #         if(np.isnan(temp) or temp < 0):
    #             temp = 0
    #         dEb_dx[k] += temp
    #         dEb_dx_test[k] += temp # eV/sMFORWG
    #     # backward receivers k <= j-1
    #     for k in range(j - 1, -1, -1):
    #         temp =  medium._Ed_bwd_test(cj[j], xk[k], dx, E_inst[j], I_beam[j])
    #         if(np.isnan(temp) or temp < 0):
    #             temp = 0
    #
    #         dEb_dx[k] += temp
    #         dEb_dx_test[k] += temp # eV/s

    eV_to_J = 1.602176634e-19
    dEb_dx_W = dEb_dx * eV_to_J  # W/m
    dEb_dx_kW_mm = dEb_dx_W / 1e6  # kW/mm
#%%

    plt.figure(figsize=(6, 4))
    plt.plot(cj * 1e3, dEb_dx_kW_mm[1:], label='Semi Empirical Model', color='black')
    plt.xlabel('Distance (mm)')
    plt.ylabel('Energy Deposition Gradient (kW/mm)')
    plt.legend()
    plt.tight_layout()
    plt.show()

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
# 3D test
if False:
    Lx = 10e-6
    dx = 0.05e-6
    Ly = 10e-6
    dy = 0.1e-6
    Lz = 10e-6
    dz = 0.1e-6
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
    medium = Medium(rho, [Mg, Fe, Si, O], Lx, A_xsec, P_perim, "Ni//Stopping_Power//H.txt", beam,
                    x0=x0)

    # Parameters for
    rho *= 1e3  # kg/m^3
    C = 850  # J/(kg*K)
    k = 1.7  # W/m*K

    sim.heateq_solid_3d(beam, medium, Lx, Ly, Lz, rho, C, k, 1000,
                                 T0 = 298, T0_faces = 298, rad_bnd = False,
                                 dx=dx, dy=dy, dz=dz, dt=0.000001, view=True, view_freq = 0.0001)
def C_f(T):
    a = 87.36
    b = 8.717e-2
    c = -3.699e6
    d = 8.436e2
    e = -2.237e-5
    M = 0.14069 # molar mass of forsterite kg/mol
    Cp = a + b * T + c * T ** -2 + d * T ** -0.5 + e * T ** 2
    return Cp/M
def k_f(T, k0 = 1.7):
    return k0*np.exp(-(T - 298)/300)

# 2D Constant Temp
if True:
    Lx = 10e-6
    dx = 0.1e-5
    Ly = 10e-3
    dy = 0.1e-4
    # Parameters for the beam and values
    I_0 = 6.24 * 1e11 # [s^-1]
    E_0 = 1e6 # [MeV]
    r = 1e-6 # m
    sig_ga_y0 = r
    sig_ga_z0 = r
    Z = 1
    x0 = 0
    beam = Beam(E_0, I_0, Z, sig_ga_y0 = sig_ga_y0, sig_ga_z0 = sig_ga_z0)

    # Parameters for the medium and values
    Z_Mg = 12
    A_Mg = 24.305 # g/mol
    Mg = Atom('Mg', Z_Mg, A_Mg, 0.2222)

    Z_Fe = 9
    A_Fe = 18.998 # g/mol
    Fe = Atom('Mg', Z_Fe, A_Fe, 0.2222)

    Z_Si = 14
    A_Si = 28.086 # g/mol
    Si = Atom('Mg', Z_Si, A_Si, 0.1111)

    Z_O = 8
    A_O = 15.999  # g/mol
    O = Atom('Mg', Z_O, A_O, 0.4444)

    rho = 3.3  # g/cm^3
    n = 9.8e23*1e6 # 1/m^3
    W = H = 10e-3  # 10 mm square slab
    A_xsec = W * H
    P_perim = 2 * (W + H)
    medium = Medium(rho, [Mg, Fe, Si, O], Lx, A_xsec, P_perim, "Ni//Stopping_Power//H.txt", beam,
                    x0 = x0)


    # Parameters for
    rho *= 1e3 # kg/m^3
    k0 = 1.7 # W/m*K
    sim.heateq_solid_2d(beam, medium, Lx, Ly, rho, C_f, k_f, 1000,
                        T0 = 298, T_amb = 298, T0_faces = [213, 213, 298, 298], rad_bnd = [False, False, True, False], dx = dx, dy = dy, view = True,
                        dT_target= 50,
                        dt = 1e-13, dt_ramp = 2)

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

# SHOWING BLACKBODY RADIATION
if False:
    Lx = 10e-3
    dx = 0.1e-3
    Ly = 10e-3
    dy = 0.1e-3
    # Parameters for the beam and values
    I_0 = 6.24 * 1e9  # [s^-1]
    E_0 = 100e6  # [MeV]
    r = 1e-3  # m
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
    W = H = 10e-3  # 10 mm square slab
    A_xsec = W * H
    P_perim = 2 * (W + H)
    medium = Medium(n, rho, [Mg, Fe, Si, O], Lx, A_xsec, P_perim, "Ni//Stopping_Power//H.txt", beam,
                    x0=x0)

    # Parameters for
    rho *= 1e3  # kg/m^3
    C = 850  # J/(kg*K)
    k = 1.7  # W/m*K

    ts, Ts = sim.heateq_solid_2d(beam, medium, Lx, Ly, rho, C, k, 10, SE = 0,
                                 T0 = 298, T0_faces=5000, rad_bnd=True,
                                 dx=dx, dy=dy, dt = 0.005, view=True)