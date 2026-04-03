#%%
from fipy import Gmsh3D
from fipy.tools import numerix as nx
from Beam import Beam
from Medium import Atom, Medium
from BoundaryConditions import BoundaryConditionsGmsh
import numpy as np
import scipy as sp
#%%

gmsh_mesh = Gmsh3D("GmshGrids//PISH.geo")
mesh = gmsh_mesh * 1e-2

# Force physical masks to be numpy bool arrays (no mesh reference inside)
mesh.physicalCells = {k: np.asarray(v, dtype=bool) for k, v in gmsh_mesh.physicalCells.items()}
mesh.physicalFaces = {k: np.asarray(v, dtype=bool) for k, v in gmsh_mesh.physicalFaces.items()}
mesh.exteriorFaces = np.asarray(gmsh_mesh.exteriorFaces, dtype=bool)

I_0 = (0.2e-6)/(1.602e-19)
E_0 = 5e6  # [MeV]
L = 15e-3
W = 15e-3
Z = 1
beam = Beam(E_0, I_0, Z, L = L, W = W, dim=3, type = 'Rectangular')


def C_Ta(T):
    M_Ta = 180.95e-3  # kg/mol

    T = np.asarray(T, dtype=float)
    T_safe = np.clip(T, 53.0, 3258.0)

    m1 = T_safe <= 298.0
    m2 = (T_safe > 298.0) & (T_safe <= 1300.0)
    m3 = T_safe > 1300.0

    Ts_low = np.array([53, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110,
                       115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175,
                       180, 185, 190, 195, 200, 205, 210, 215, 220, 225, 230, 235, 240,
                       245, 250, 255, 260, 265, 270, 275, 280, 285, 290, 295], dtype=float)

    Cp_low = np.array([2.81, 2.95, 3.26, 3.54, 3.80, 4.03, 4.22, 4.40, 4.55, 4.68, 4.78,
                       4.87, 4.97, 5.06, 5.14, 5.21, 5.27, 5.33, 5.39, 5.44, 5.48, 5.52, 5.55,
                       5.58, 5.62, 5.65, 5.68, 5.70, 5.73, 5.76, 5.78, 5.80, 5.82, 5.84, 5.86, 5.88,
                       5.89, 5.91, 5.92, 5.94, 5.95, 5.96, 5.98, 5.99, 6.01, 6.02, 6.03, 6.03, 6.04, 6.04],
                      dtype=float)

    Cp_low_J_per_molK = Cp_low * 4.184
    f_low = sp.interpolate.PchipInterpolator(Ts_low, Cp_low_J_per_molK)
    val1 = f_low(T_safe) / M_Ta  # -> J/(kg K)

    T_safe /= 1000
    A, B, C, D, E = 20.69482, 17.29992, -15.68987, 5.608694, 0.061581
    val2 = (A + B * T_safe + C * T_safe ** 2 + D * T_safe ** 3 + E / (T_safe ** 2)) / M_Ta


    A, B, C, D, E = -43.87133, 73.02084, -27.40796, 4.004682, 26.30414
    val3 = (A + B * T_safe + C * T_safe ** 2 + D * T_safe ** 3 + E / (T_safe ** 2)) / M_Ta

    Cp = m1*val1 + m2*val2 + m3*val3
    return nx.maximum(1.0, Cp)

# https://srd.nist.gov/jpcrdreprint/1.3253100.pdf
def k_Ta(T):
    T_safe = nx.clip(T, 53, 1800)
    cond1 = (T_safe < 298)
    cond2 = (T_safe >= 298)

    Ts_low = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                       18, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100,
                       123.2, 150, 173.2, 200, 223.2, 250, 273.2, 298.2])
    k_low = np.array([0, 0.115, 0.230, 0.344, 0.458, 0.569, 0.678, 0.784, 0.884, 0.979, 1.07, 1.15, 1.22,
                      1.28, 1.33, 1.37, 1.140, 1.43, 1.42, 1.30, 1.15, 0.99, 0.87, 0.78, 0.72, 0.651, 0.616,
                      0.603, 0.596, 0.592, 0.586, 0.580, 0.578, 0.575, 0.574, 0.574, 0.574, 0.575])
    k_low *= 1e2 # convert to W/m*K
    f_low = sp.interpolate.PchipInterpolator(Ts_low, k_low)
    val1 = f_low(T_safe)
    val2 = 57.5 + 0.00025 * (T - 273.15)

    return  cond1 * val1 + cond2 * val2

Z_Ta = 73
A_Ta = 181
Ta = Atom('Ta', Z_Ta, A_Ta, 1)
rho_Ta = 16690

tantalum = Medium(rho_Ta, C_Ta, k_Ta, Ta,
                  "SRIMTables//H in Ta.txt", name='Tantalum')
def C_Al(T):
    M_Al = 0.02698  # kg/mol
    # Prevent T from going to extreme values that break polynomials
    T_safe = nx.clip(T, 1, 2200)

    Ts_low = np.array([1, 2, 3, 4, 6, 8, 10, 15, 20, 25,
                       30, 35, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160,
                       180, 200, 220, 240, 260, 280, 300])
    Cp_low = np.array([0.000051, 0.000108, 0.000176, 0.000261,
                       0.00050, 0.00088, 0.0014, 0.0040, 0.0089, 0.0175,
                       0.0315, 0.0515, 0.0775, 0.142, 0.214, 0.287, 0.357, 0.422,
                       0.481, 0.580, 0.654, 0.713, 0.760, 0.797, 0.826, 0.849,
                       0.869, 0.886, 0.902]) * 1e3

    f_low = sp.interpolate.PchipInterpolator(Ts_low, Cp_low)
    cond1 = (T_safe < 300)
    cond2 = (T_safe >= 300) & (T_safe < 933)
    cond3 = (T_safe >= 933)

    val1 = f_low(T_safe)

    T_safe /= 1000

    A = 28.08920
    B = -5.414849
    C = 9.560423
    D = 3.427370
    E = -0.277375
    val2 = A + B * T_safe + C * T_safe ** 2 + D * T_safe ** 3 + E / (T_safe ** 2)
    val2 /= M_Al

    A = 31.75104
    B = 3.935826e-8
    C = -1.786515e-8
    D = 2.694171e-9
    E = 5.480037e-9
    val3 = A + B * T_safe + C * T_safe ** 2 + D * T_safe ** 3 + E / (T_safe ** 2)
    val3 /= M_Al
    Cp = cond1 * val1 + cond2 * val2 + cond3 * val3
    # Return J/kg*K and ensure it's always positive
    return nx.maximum(Cp, 1.0)


# https://inis.iaea.org/records/n4gzz-mpp69
def k_Al(T):
    # Prevent T from going to extreme values that break polynomials
    T_safe = nx.clip(T, 0, 2200)

    Ts_solid = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                16, 18, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100,
                123.2, 150, 173.2, 200, 223.2, 250, 273.2, 298.2, 300, 323.2,
                350, 373.2, 400, 473.2, 500, 573.2, 600, 673.2, 700, 773.2,
                800, 873.2, 900, 933.52]
    Ks_solid = [0, 41.1, 81.8, 121, 157, 188, 213, 229, 237, 239,
                235, 226, 214, 201, 189, 176, 138, 117, 75.2, 49.5,
                33.8, 24.0, 17.7, 13.5, 8.50, 5.85, 4.32, 3.42, 3.02,
                2.62, 2.48, 2.41, 2.37, 2.35, 2.35, 2.35,
                2.36, 2.37, 2.37, 2.39, 2.40, 2.40, 2.40, 2.37, 2.36, 2.33,
                2.31, 2.26, 2.25, 2.19, 2.18, 2.12, 2.10, 2.08]
    Ts_liquid = [933.52, 973.2, 1000, 1073.2, 1100, 1173.2, 1200, 1273.2,
                 1300, 1372.2, 1400, 1473.2, 1500, 1573.2, 1600, 1673.2, 1700,
                 1773.2, 1800, 1873.2, 1900, 1973.2, 2000, 2073.2, 2173.2, 2200]
    Ks_liquid = [0.907, 0.921, 0.390, 0.955, 0.964, 0.986, 0.994, 1.01, 1.02, 1.04, 1.05, 1.07,
                 1.07, 1.08, 1.09, 1.10, 1.11, 1.11, 1.12, 1.13, 1.13, 1.14, 1.14, 1.14, 1.15, 1.15]

    f_sol = sp.interpolate.PchipInterpolator(Ts_solid, Ks_solid)
    f_liq = sp.interpolate.PchipInterpolator(Ts_liquid, Ks_liquid)
    cond1 = (T_safe < 933.52)
    cond2 = (T_safe >= 933.52)
    val1 = f_sol(T_safe)
    val2 = f_liq(T_safe)

    k_combined = (cond1 * val1 + cond2 * val2) * 1e2
    # Final safety floor (k should never be 0 or negative)
    return nx.maximum(k_combined, 1e-6)

# Define aluminum
Z_Al = 13
A_Al = 27  # g/mol
Al = Atom('Al', Z_Al, A_Al, 1)
rho_Al = 2700  # kg/m^3

aluminum = Medium(rho_Al, C_Al, k_Al,
                  Al,
                  "SRIMTables//H in Al.txt", name='Aluminum')


def C_olivine(T):
    """"
    Sources:
      Forsterite high-T: Gillet et al. (1991), J. Geophys. Res., 96(B7), 11831-11838
      Forsterite low-T:  Robie, Hemingway & Takei (1982), Am. Mineral., 67, 470-482
      Fayalite high-T:   Dachs et al. (2007), J. Chem. Thermodyn., 39, 906-933
      Mixing rule:       Linear in mole fractions, valid >300K
    """
    # Molar masses [kg/mol]
    M_Fo = 0.14069  # Mg2SiO4
    M_Fa = 0.20378  # Fe2SiO4

    # Fo fraction
    x_Fo = 0.2533 / (0.2533 + 0.1457)
    x_Fa = 1.0 - x_Fo

    # Effective molar mass for this composition [kg/mol]
    M = x_Fo * M_Fo + x_Fa * M_Fa  # = 0.14252 kg/mol

    T_safe = nx.clip(T, 200.0, 2000.0)


    # Low-T: Robie et al. (1982), tabulated 5-380 K, anchored at 298 K = 118.6 J/mol/K
    # Smooth polynomial fit to Robie (1982) data for 200-700K:
    Cp_Fo_lowT = (118.6
                  + 0.02489 * (T_safe - 298.0)
                  - 1.037e-5 * (T_safe - 298.0) ** 2)

    # High-T: Gillet et al. (1991), valid 700-1850 K
    Cp_Fo_highT = (-402.753
                   + 74.290 * nx.log(nx.maximum(T_safe, 1.0))
                   + 87.588e3 / T_safe
                   - 25.913e6 / T_safe ** 2
                   + 25.374e8 / T_safe ** 3)

    cond_lo = (T_safe < 700.0)
    Cp_Fo = cond_lo * Cp_Fo_lowT + (1.0 - cond_lo) * Cp_Fo_highT

    # Dachs et al. (2007)
    Cp_Fa = (-217.137
             + 63023.1 / T_safe
             - 2.15863e7 / T_safe ** 2
             + 2.23513e9 / T_safe ** 3
             + 51.7620 * nx.log(nx.maximum(T_safe, 1.0)))

    # Linear mixing
    Cp_mix = x_Fo * Cp_Fo + x_Fa * Cp_Fa

    # Convert to J/(kg·K)
    return nx.maximum(Cp_mix / M, 1.0)


def k_olivine(T, Tmin=160, Tmax=2000.0):
    Fo = 0.2533 / (0.2533 + 0.1457)

    k0 = 3.09 - 1.17 * Fo + 3.35 * Fo ** 2

    Tf = nx.clip(1.0 * T, Tmin, Tmax)
    k = k0 * (298.0 / Tf) ** 0.49

    return nx.clip(k, 1e-4, 50.0)

rho_olivine = 3.37e3  # kg/m^3

# Parameters for the medium and values
Z_Mg = 12
A_Mg = 24.305  # g/mol
Mg = Atom('Mg', Z_Mg, A_Mg, 0.2533)

Z_Fe = 26
A_Fe = 55.845  # g/mol
Fe = Atom('Fe', Z_Fe, A_Fe, 0.1457)

Z_Si = 14
A_Si = 28.086  # g/mol
Si = Atom('Si', Z_Si, A_Si, 0.1833)

Z_O = 8
A_O = 15.999  # g/mol
O = Atom('O', Z_O, A_O, 0.4177)

olivine = Medium(rho_olivine, C_olivine, k_olivine,
                 [Mg, Fe, Si, O],
                 "SRIMTables//H in Olivine.txt",
                 name='Olivine')

mediums = [tantalum, olivine, aluminum]
print(f"N_tot olivine: {olivine.N_tot:.3e}")  # should be ~9e28 atoms/m³
print(f"N_tot aluminum: {aluminum.N_tot:.3e}")  # should be ~6e28 atoms/m³
print(f"Se at 5MeV: {olivine.get_Se_ev_m(5e6):.3e} eV/m")  # should be ~1e9 eV/m
BC = BoundaryConditionsGmsh(mesh, T0=[298.0, 298.0, 298.0], T_amb=298.0,
                            eps=[0.21, 0.85, 0.04], h = [40000, 40000, 10000], R = [2000, 2000, 2000])
if __name__ == "__main__":
    import Simulation as sim
    ts, Ts = sim.heateq_solid_3d_gmsh(
        beam, mediums, mesh, BC,
        6000, T0=298.0,
        x_scale=1e2, x_scale_min=0,
        dt=0.1, dt_ramp=1.5, dt_max=500,
        dT_target=10, omega=0.9,
        abs_residual_target=0.01,
        min_sweeps=2, max_sweeps=6,
        debug=True
    )
    #%%
    import matplotlib.pyplot as plt
    # Remove stalled points (same T at consecutive steps)
    ts_clean = [ts[0]]
    Ts_clean = [Ts[0]]
    for i in range(1, len(ts)):
        if abs(Ts[i] - Ts[i - 1]) > 1e-3:
            ts_clean.append(ts[i])
            Ts_clean.append(Ts[i])

    plt.figure(figsize=(8, 6))
    print(max(np.array(Ts_clean)*1.05))
    plt.plot(ts_clean, np.array(Ts_clean)*1.04, 'ko', label='Maximum Temperature')
    plt.ylabel("Temperature [K]", fontsize='xx-large')
    plt.xlabel("Simulation Time [s]", fontsize='xx-large')
    plt.tick_params(labelsize='x-large')
    plt.show()
