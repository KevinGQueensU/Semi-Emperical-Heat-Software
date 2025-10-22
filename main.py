import scipy as sp
from Beam import Beam
from Medium import atom, Medium
import numpy as np
import matplotlib.pyplot as plt

if True:
    I_0 = 6.24 * 1e15 # [s^-1]
    E_0 = 500e6 # [MeV]
    r = 1e-3 # m
    sig_ga_y0 = r
    sig_ga_z0 = r
    Z = 1
    x0 = 0
    beam = Beam(E_0, I_0, Z, sig_ga_y0 = sig_ga_y0, sig_ga_z0 = sig_ga_z0)

    Z_Ni = 28
    A_Ni = 58.693  # g/mol
    Ni = atom('Ni', Z_Ni, A_Ni, 1, "Ni//Cross_Sections//Ni_px.txt")

    rho = 8.908  # g/cm^3
    n = 2.56e30 # 1/m^3
    W = H = 10e-3  # 10 mm square slab
    A_xsec = W * H
    P_perim = 2 * (W + H)
    L = 350e-3
    medium = Medium(n, rho, Ni, L, A_xsec, P_perim, "Ni//Stopping_Power//H.txt", beam,
                    x0 = x0)

    xk = np.linspace(0, L, 1000) # cell edges
    cell_width = xk[1] - xk[0]
    cj = 0.5*(xk[:-1] + xk[1:]) # cell centers
    alpha = beta = 0.0
    y = z = 0

    E_beam = np.empty(xk.size + 1)
    E_beam[0] = E_0*I_0  # eV/s
    dIdx = np.empty(xk.size + 1)
    E_inst = np.empty(xk.size + 1)
    E_inst[0] = E_0 # eV
    I_beam = np.empty(xk.size + 1)
    I_beam[0] = I_0  # 1/s
    dIdx[0] = medium.get_dIdx(E_inst[0], I_beam[0])
    dEb_dx = np.zeros_like(xk)  # eV/(m·s)
    dEdx = np.zeros_like(xk)
    dEdx_beam = np.zeros_like(xk)
    phi_free = np.array([beam.PD(x, y, z, alpha, beta) for x in xk]) # Free energy flux eV/(m^2·s)

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
            temp = medium._Ed_fwd(cj[j], cell_width, xk[k], E_inst[j], I_beam[j]) * 1e2
            if(np.isnan(temp) or temp < 0):
                temp = 0
            dEb_dx[k] += temp
            dEb_dx_test[k] += temp # eV/sMFORWG
        # backward receivers k <= j-1
        for k in range(j - 1, -1, -1):
            temp =  medium._Ed_bwd(cj[j], cell_width, xk[k], E_inst[j], I_beam[j]) * 1e2
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