from Beam import Beam
from Medium import atom, Medium
import numpy as np
import matplotlib.pyplot as plt
if True:
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
    Ni = atom('Ni', Z_Ni, A_Ni, 1, "C://Users//k_gao//Desktop//Thesis//Semi-Emperical Heat Software//Ni//Cross_Sections//Ni_px.txt")

    rho = 8.908  # g/cm^3
    n = 2.56e30 # 1/m^3
    W = H = 10e-3  # 10 mm square slab
    A_xsec = W * H
    P_perim = 2 * (W + H)
    L = 60e-3
    medium = Medium(n, rho, Ni, L, A_xsec, P_perim, "C://Users//k_gao//Desktop//Thesis//Semi-Emperical Heat Software//Ni//Stopping_Power//H.txt",
                    x0 = x0)

    dx_m = 0.01e-3
    xs = np.arange(0.0, L, dx_m)
    alpha = beta = 0.0
    y = z = 0
    phi_free = np.array([beam.PD(x, y, z, alpha, beta) for x in xs])

    E_beam = np.empty(xs.size + 1);
    E_beam[0] = E_0 * I_0  # eV/s
    dIdx = np.empty(xs.size + 1);
    I_beam = np.empty(xs.size + 1);
    I_beam[0] = I_0  # 1/s
    E_inst = np.empty(xs.size + 1);
    E_inst[0] = E_0 # eV
    dIdx[0] = medium.get_dIdx(E_inst[0], I_beam[0])
    dEb_dx = np.empty_like(xs)  # eV/(m·s)
    phi_free = np.array([beam.PD(x, y, z, alpha, beta) for x in xs]) # Free energy flux eV/(m^2·s)

    for k, x in enumerate(xs):
        # Get Se
        dEdx = medium.get_Se_ev_m(E_inst[k])

        # Get energy gradient
        dEb_dx[k] = medium.get_Egrad(x, dx_m, E_inst[k], I_beam[k])

        # Eq. 38
        E_beam[k + 1] = max(E_beam[k] - dEb_dx[k] * dx_m, 0.0)

        # Intensity attenuation: dI/dx
        dIdx[k+1] = medium.get_dIdx(E_inst[k], I_beam[k])  # (1/s)/m

        # Decrease beam intensity
        I_beam[k + 1] = I_beam[k] + dIdx[k] * dx_m

        # Decrease particle energy
        E_inst[k + 1] = max(E_inst[k] - dEdx * dx_m, 0.0)

    eV_to_J = 1.602176634e-19
    dEb_dx_W = dEb_dx * eV_to_J  # W/m
    dEb_dx_kW_mm = dEb_dx_W / 1e6  # kW/mm

    plt.figure(figsize=(6, 4))
    plt.plot(xs * 1e3, dEb_dx_kW_mm, label='Semi Empirical Model', color='black')
    plt.xlabel('Distance (mm)')
    plt.ylabel('Energy Deposition Gradient (kW/mm)')
    plt.legend()
    plt.tight_layout()
    plt.show()
    #%%
    Es_test = np.linspace(0, 500e6)
    dEdxs = medium.get_Se_ev_m(Es_test)
    plt.plot(Es_test*1e-6, dEdxs*1e-6)
    plt.title("Stopping Power: H -> Ni")
    plt.xlabel('Energy (MeV)')
    plt.ylabel('Stopping Power (MeV/m)')
    plt.show()

    plt.title("dI/dx")
    plt.xlabel('Distance [mm]')
    plt.ylabel(r'$\frac{dI}{dx}$  $[s\cdot mm]^{-1}$')
    plt.plot(xs*1e3, dIdx[1:]*1e-3)
    plt.show()

    plt.title("Instantaneous Beam Energy")
    plt.xlabel('xs [mm]')
    plt.ylabel('E [MeV]')
    plt.plot(xs*1e3, E_inst[1:]*1e-6)
    plt.show()

    plt.title("Beam Energy")
    plt.xlabel('xs [mm]')
    plt.ylabel(r'$E_{beam}$ [MeV/s]')
    plt.plot(xs*1e3, E_beam[1:]*1e-6)
    plt.show()
