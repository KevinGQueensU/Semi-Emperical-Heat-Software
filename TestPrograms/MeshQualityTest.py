# %%
import numpy as np
from Beam import Beam
from fipy import Gmsh3D
from BoundaryConditions import BoundaryConditionsGmsh
from Simulation import heateq_solid_3d_gmsh
from Medium import Atom, Medium
from fipy.tools import numerix as nx

# Geo template, mesh_size in Gmsh units (= µm after *1e-6 in Python)
GEO_TEMPLATE = """
SetFactory("OpenCASCADE");
Mesh.Recombine3DAll = 1;
Mesh.MeshSizeMax = {mesh_size};
Mesh.MeshSizeMin = {mesh_size};
Box(1) = {{0, -1, -1, 2, 2, 2}};
Physical Volume("Olivine", 25) = {{1}};
Physical Surface("BBR", 26) = {{6, 4, 5, 3}};
Physical Surface("Fixed", 27) = {{2}};
"""

# Beam (fixed across all runs)
I_0 = (2e-8) / (1.602e-19)
E_0 = 5e6
L = 1e-6
Z = 1
beam = Beam(E_0, I_0, Z, L=L, W=L, dim=3, type='Rectangular')

#  Material functions
def C_olivine(T):
    M = 0.14069
    T_safe = nx.clip(T, 200, 3000)
    a, b, c, d, e = 87.36, 8.717e-2, -3.699e6, 8.436e2, -2.237e-5
    Cp = a + b*T_safe + c/(T_safe**2) + d/(T_safe**0.5) + e*(T_safe**2)
    return nx.maximum(Cp / M, 1.0)

def k_olivine(T, k0=1.7, *, Tmin=160, Tmax=6000.0, kmin=1e-4, kmax=50.0):
    Tf = nx.clip(1.0 * T, Tmin, Tmax)
    return nx.clip(k0 * nx.exp(-(Tf - 298.0) / 300.0), kmin, kmax)

rho_olivine = 3.3e3

Mg = Atom('Mg', 12, 24.305, 0.2222)
Fe = Atom('Fe', 26, 55.845, 0.2222)
Si = Atom('Si', 14, 28.086, 0.1111)
O  = Atom('O',   8, 15.999, 0.4444)

olivine = Medium(rho_olivine, C_olivine, k_olivine,
                 [Mg, Fe, Si, O],
                 "SRIMTables//H in Olivine.txt",
                 name='Olivine')

# %%
if __name__ == "__main__":

    # Mesh convergence sweep
    # Cell count scales as (1/h)^3
    mesh_sizes_um = [2.0, 1.95, 1.9, 1.8, 1.75, 1.5, 1.25, 1.0, 0.9, 0.8, 0.7, 0.65, 0.6, 0.55, 0.5, 0.40,
                     0.30, 0.25, 0.20, 0.15, 0.10]
    results = []  # list of (mesh_size_um, n_cells, T_max)

    for ms in mesh_sizes_um:
        print(f"\n{'='*50}")
        print(f"Running mesh size: {ms} µm")

        geo_str = GEO_TEMPLATE.format(mesh_size=ms)
        gmsh_mesh = Gmsh3D(geo_str)
        mesh = gmsh_mesh * 1e-6

        mesh.physicalCells = {k: np.asarray(v, dtype=bool)
                              for k, v in gmsh_mesh.physicalCells.items()}
        mesh.physicalFaces = {k: np.asarray(v, dtype=bool)
                              for k, v in gmsh_mesh.physicalFaces.items()}
        mesh.exteriorFaces = np.asarray(gmsh_mesh.exteriorFaces, dtype=bool)

        n_cells = mesh.numberOfCells
        print(f"  Cells: {n_cells}")

        BC = BoundaryConditionsGmsh(mesh, T0=298.0, T_amb=298.0, eps=0.75)

        ts, Ts = heateq_solid_3d_gmsh(
            beam, [olivine], mesh, BC,
            10, T0=298.0,
            dt=0.001, dt_ramp=2, dT_target=500,
            debug=False, max_sweeps=10, view=False, omega = 0.5
        )

        T_max = float(np.max(Ts[-1]))
        print(f"  T_max = {T_max:.2f} K")
        results.append((ms, n_cells, T_max))

    # %%
    # Plot
    import matplotlib.pyplot as plt

    mesh_sizes = [r[0] for r in results]
    ncells     = [r[1] for r in results]
    Tmaxs      = [r[2] for r in results]

    fig, axes = plt.subplots(1, 1, figsize=(8, 5))


    axes.plot(mesh_sizes[0:], Tmaxs[0:], 'o-', color='black')
    axes.invert_xaxis()   # coarse -> fine left to right
    axes.set_xlabel("Mesh Element Size [µm] (coarse → fine)", fontsize = 'xx-large')
    axes.set_ylabel("Max Temperature [K]", fontsize = 'xx-large')
    plt.tight_layout()
    axes.tick_params(labelsize='large')
    axes.axvline(1, color = 'red', linestyle = '--', label = r"Beam Width and Height")
    plt.legend(fontsize = 'large')
    plt.savefig("mesh_convergence.png", dpi=150)
    plt.show()

    print("\nResults summary:")
    print(f"{'Size (µm)':>12} {'N Cells':>10} {'T_max (K)':>12}")
    for r in results:
        print(f"{r[0]:>12.3f} {r[1]:>10d} {r[2]:>12.2f}")