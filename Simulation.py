from Medium import Medium
from Beam import Beam
from BoundaryConditions import BoundaryConditions, BoundaryConditionsGmsh
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from fipy import Gmsh3D, CellVariable, Grid2D, Grid3D, TransientTerm, DiffusionTerm, Viewer, FaceVariable
from scipy.interpolate import PchipInterpolator
from scipy.integrate import cumulative_trapezoid
from typing import Any
import bisect

# Unit multipliers
unit_factor = {
    "m": 1.0,
    "cm": 1e2,
    "mm": 1e3,
    "um": 1e6, "µm": 1e6,
    "nm": 1e9,
    "km": 1e-3,
}

#### HELPER FUNCTIONS ####
# FUCNTION: Import a FiPy Viewer 2D object and scale the x and y axis to match a certain unit
def scale_viewer_axes_2D(viewer: Viewer,  # FiPy 2D Viewer
                         x_units: str = "mm", y_units: str ="mm",  # Default is m -> mm
                         x_label: str = "x", y_label: str="y",  # Default labels
                         decimals: int = 3,  # How many decimals to display on each axis
                         equal_aspect: bool = True  # Make x width = y height?
                         ) -> matplotlib.axes.Axes:

    # Get the axes from the viewer
    ax = getattr(viewer, "axes", None) or getattr(viewer, "_axes", None)
    if ax is None:
        raise ValueError("Couldn't find Matplotlib axes on the viewer. "
                         "Make sure you're using a Matplotlib-based viewer and call this after viewer.plot().")
    fig = ax.figure
    if fig is not None and fig.canvas is not None:
        fig.canvas.draw_idle()
    fig.set_size_inches((6, 6))

    unit_factor = {
        "m": 1.0,
        "cm": 1e2,
        "mm": 1e3,
        "um": 1e6, "µm": 1e6,
        "nm": 1e9,
        "km": 1e-3,
    }

    if x_units not in unit_factor or y_units not in unit_factor:
        raise ValueError("Unsupported units. Use one of: m, cm, mm, um/µm, nm, km")

    fx = unit_factor[x_units]
    fy = unit_factor[y_units]

    # Create formatters for the ticks that multiply the underlying data
    def _mk_formatter(f, sig):
        fmt = "{:." + str(sig) + "g}"
        return ticker.FuncFormatter(lambda val, pos: fmt.format(val * f))

    # Set the axes
    ax.xaxis.set_major_formatter(_mk_formatter(fx, decimals))
    ax.yaxis.set_major_formatter(_mk_formatter(fy, decimals))
    ax.set_xlabel(f"{x_label} [{x_units}]")
    ax.set_ylabel(f"{y_label} [{y_units}]")

    # Enforce aspect ratio(?)
    if equal_aspect:
        ax.set_aspect("equal", adjustable="box")
        forceAspect(ax)

    return ax


# FUNCTION: Given a set of axes scale the data to match a specific unit
def scale_axes(ax: matplotlib.pyplot.axes,
               plane: str = "xy",  # "xy", "xz", or "yz"
               units: str = ("mm", "mm"),  # (x_units, y_units)
               decimals: int = 3,  # OPTIONAL: tick label precision
               equal_aspect: bool =True,  # OPTIONAL: force equal aspect ratio
               preserve_limits: bool =True,  # OPTIONAL: Keep x and y limits
               labelsize: int = 13,  # OPTIONAL: Axis label font size
               ticklabelsize:int = 11
               ) -> matplotlib.pyplot.axes:    # OPTIONAL: Tick label font size

    # Validate given planes
    plane = plane.lower()
    if plane not in ("xy", "xz", "yz"):
        raise ValueError("plane must be one of: 'xy', 'xz', 'yz'")

    x_label_char, y_label_char = {
        "xy": ("x", "y"),
        "xz": ("x", "z"),
        "yz": ("y", "z"),
    }[plane]

    ux, uy = units
    if ux not in unit_factor or uy not in unit_factor:
        raise ValueError("Unsupported units. Use one of: m, cm, mm, um/µm, nm, km")

    fx = unit_factor[ux]
    fy = unit_factor[uy]

    # Keep current limits
    if preserve_limits:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

    # Scaled tick formatters
    def _mk_formatter(scale, sig):
        fmt = "{:." + str(sig) + "g}"
        return ticker.FuncFormatter(lambda val, pos: fmt.format(val * scale))

    ax.xaxis.set_major_formatter(_mk_formatter(fx, decimals))
    ax.yaxis.set_major_formatter(_mk_formatter(fy, decimals))

    # Labels + font sizes
    ax.set_xlabel(f"{x_label_char} [{ux}]", fontsize=labelsize)
    ax.set_ylabel(f"{y_label_char} [{uy}]", fontsize=labelsize)
    ax.tick_params(axis="both", labelsize=ticklabelsize)

    # Equal aspect in displayed units
    if equal_aspect:
        ax.set_aspect(fy / fx, adjustable="box")

    fig = ax.figure
    if fig is not None and fig.canvas is not None:
        fig.canvas.draw_idle()

    if preserve_limits:
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

    return ax


# FUNCTION: Force the plot to be 1:1 aspect ratio
def forceAspect(ax, aspect=1):
    im = ax.get_images()
    extent = im[0].get_extent()
    ax.set_aspect(abs((extent[1] - extent[0]) / (extent[3] - extent[2])) / aspect);


# FUNCTION: Return the region (index) where x lies given a bunch of regions defined by x0
def region_index(x0s: list[float] | float, # x0s that separate regions
                 x: float                  # Position where you want to find the region its in
                 ) -> float:
    # x0 must be sorted (strictly increasing recommended)
    x0s = np.asarray(x0s, dtype=float)
    m = len(x0s)
    if m == 0:
        raise ValueError("x0s must contain at least one region start.")

    # Region start positions: choose last start <= x
    i = bisect.bisect_right(x0s, x) - 1
    if i < 0:
        return 0
    if i >= m:
        return m - 1
    return i

# FUNCTION: Compute the energy deposition gradient at a specific point given a beam and target medium
def compute_dEb_dx( x: float,        # Position to compute SE at (beam is fired along x-axis)
                    x_ref: float,    # Initial x position where values are known
                    dx: float,       # Step size along x
                    beam: Beam,      # Particle beam shot along x-direction
                    medium: Medium | list[Medium],   # Target medium(s) that particle beam is being shot at
                    E = None,        # For stepping with specific energy
                    ) -> [float, float]:
    if x_ref > x:
        dx = -abs(dx)

    if not np.iterable(medium):
        medium = np.array([medium])
    x_med = np.asarray([med.x0 for med in medium])

    if(E is not None):
        E_inst = E
    else:
        E_inst = float(beam.E_0)   # eV per particle

    I_beam = float(beam.I_0)   # 1/s

    E_beam = E_inst * I_beam
    xi = float(x_ref)
    med_i = 0

    while True:
        if (dx > 0 and xi >= x) or (dx < 0 and xi <= x):
            break
        med_i = region_index(x_med, xi)

        # Step everything
        dEdx = medium[med_i].get_dEdx(E_inst)
        dIdx = medium[med_i].get_dIdx(E_inst, I_beam)
        dEdx_beam = dEdx * I_beam + E_inst * dIdx
        I_beam = I_beam + dIdx * dx
        E_beam = max(E_beam + dEdx_beam * dx, 0)
        E_inst = E_beam / I_beam
        if E_beam == 0:
            break

        xi += dx

    dEdx = medium[med_i].get_dEdx(E_inst)
    return -(I_beam * dEdx), E_inst


#### SIMULATION FUNCTIONS ####

# FUNCTION: Simulate heat generation for a particle beam irradiating a solid material(s) in 2D.
# Given boundary conditions and material(s) properties.
def heateq_solid_2d(
        beam: Beam,
        medium: Medium | list[Medium],              # Supports connected materials
        BC: BoundaryConditions,
        Ly: float,                                  # Height of sim box
        t: float,                                   # Total simulation time [s]
        T0: float = 298,                            # Initial simulation temperature [K]
        SE = None,                                  # Pre-computed source energy term
        x_shift: float = None, y_shift=None,        # How much to shift the origin by
        alpha: float = 0,                           # Beam divergence in y
        dx: float = 1e-4, dy=1e-4,                  # Cell widths and heights
        dt: float = 1e-3,                           # Time interval between steps
        view: bool = False,                         # Enable viewer?
        view_freq: int = 20,                        # Update viewer every N steps
        dT_target: float = None,                    # Scale dt to achieve a specific dT per step
        dt_ramp: float = None,                      # Scaling factor to ramp dt each step
        dt_max: float = 1,                          # Maximum value dt can ramp to
        x_units: str = 'mm', y_units='mm',          # Scale viewer axes to a specific unit
        min_sweeps: int = 3, max_sweeps=10,         # Min/max sweeps per timestep
        rel_residual_target: float = 1e-4,          # Relative residual to stop sweeping at
        debug: bool = False                         # Debugging outputs
        ) -> [float, float]:

    # If only single material given, make it a list for ease of use later
    if not np.iterable(medium):
        medium = [medium]
    Lx = 0
    for med in medium:
        Lx += med.Lx
    # Create the mesh
    nx = int(np.floor(Lx / dx))
    ny = int(np.floor(Ly / dy))
    mesh = Grid2D(dx=dx, dy=dy, nx=nx, ny=ny)
    if x_shift is None:
        x_shift = 0
    if y_shift is None:
        y_shift = -Ly / 2

    mesh += ((x_shift,), (y_shift,))  # shift mesh
    cx = mesh.cellCenters[0].value
    cy = mesh.cellCenters[1].value
    CX = cx.reshape((nx, ny), order='F')  # shape (nx, ny)
    CY = cy.reshape((nx, ny), order='F')

    # Mask regions of the mesh for each medium
    x = mesh.cellCenters[0]
    masks = []
    for i, med in enumerate(medium):
        xL = med.x0
        xR = medium[i + 1].x0 if i + 1 < len(medium) else (med.x0 + med.Lx)
        masks.append((x >= xL) & (x < xR))

    # Compute SE if not given
    if SE is None:
        cj = CX[:, 0]
        dEb_dx = np.zeros_like(cj)  # eV/(m·s)
        E_inst = np.zeros_like(cj)
        if(debug):
            dE_dx = np.zeros_like(cj)

        for l in range(nx - 1):
            dEb_dx[l], E_inst[l] = compute_dEb_dx(cj[l], cj[0], dx, beam, medium)
            if(debug):
                dE_dx[l] = medium[0].get_Se_ev_m(E_inst[l])
            if E_inst[l] <= 1.0 and debug:  # Cutoff at 1 eV to avoid float noise
                print(f"Beam stopped at x = {cj[l]:.2e} m (Cell {l}/{nx})")
                dEb_dx[l + 1:] = 0.0  # Explicitly zero out the rest
                E_inst[l + 1:] = 0.0
                break
        dEb_dx *= 1.602176634e-19  # ev to J
        phi_free = np.array(beam.PD(CX, CY, 0.0, alpha, 0))
        SE = dEb_dx[:, None] * phi_free * 1 / (float(beam.E_0 * beam.I_0))
        SE /= medium[0].Lz

        if(debug):
            plt.plot(cj*unit_factor[x_units], dE_dx*1e-3*1e-6, 'bo', markersize=2)
            plt.xlabel(rf'x [{x_units}]', size='x-large')
            plt.ylabel(r"$dE/dx$ $[keV/nm$)]", size='x-large')
            plt.show()

            plt.plot(cj*unit_factor[x_units], dEb_dx*1e-3, 'bo', markersize=2)
            plt.xlabel(rf'x [{x_units}]', size='x-large')
            plt.ylabel(r"$dE_{b}/dx$ $[keV/m · s^{-1}$)]", size='x-large')
            ax = plt.gca()
            for i, med in enumerate(medium[1:]):
                ax.axvline(med.x0*unit_factor[x_units], ls='--', color='k', lw=1)
            plt.show()

            plt.plot(cj*unit_factor[x_units], E_inst*1e-3, 'bo', markersize = 2)
            plt.xlabel(rf'x [{x_units}]', size='x-large')
            plt.ylabel(r"$E_{inst}$ [keV]", size='x-large')
            ax = plt.gca()
            for i, med in enumerate(medium[1:]):
                ax.axvline(med.x0 * unit_factor[x_units], ls='--', color='k', lw=1)
            plt.show()

            plt.figure(figsize=(10, 4))
            plt.plot(cj*unit_factor[x_units], SE[:, ny // 2], label=r"Source Term ($\frac{W}{m^3}$)")
            ax = plt.gca()
            for i, med in enumerate(medium[1:]):
                ax.axvline(med.x0 * unit_factor[x_units], ls='--', color='k', lw=1)
            plt.xlabel(rf'x [{x_units}]', size='x-large')
            plt.ylabel("Heat Source Intensity", size='x-large')
            plt.title("Beam Energy Deposition Profile (Centerline)")
            plt.legend()
            plt.show()
            P_per_m = float((SE.reshape(-1, order='F') * mesh.cellVolumes).sum())
            print("Injected power [W/m] =", P_per_m)
            print("Injected power [W]   =", P_per_m * Ly)

        SE = SE.reshape(-1, order='F')

    # Create FiPy variables
    T0 = float(T0)
    SE = CellVariable(mesh=mesh, value=SE, name=r"$S_{E}$")
    T = CellVariable(mesh=mesh, value=T0, name="Temperature [K]", hasOld=True)

    rhoC = CellVariable(mesh=mesh, name="rhoC")
    k_cell = CellVariable(mesh=mesh, name="k")

    def _manual_refresh_props():
        current_T = T.value
        new_rhoC = np.zeros(mesh.numberOfCells)
        new_k = np.zeros(mesh.numberOfCells)

        for i, med in enumerate(medium):
            cp_vals = med.get_C(current_T)
            k_vals = med.get_k(current_T)
            m = np.array(masks[i])
            new_rhoC[m] = med.rho * cp_vals[m]
            new_k[m] = k_vals[m]

        rhoC.setValue(new_rhoC)
        k_cell.setValue(new_k)

    k_face = k_cell.harmonicFaceValue
    eq = TransientTerm(coeff=rhoC) == DiffusionTerm(coeff=k_face) + SE

    if view:
        try:
            viewer = Viewer(vars=(T,), title="Temperature Distribution")
            ax = viewer.axes

            y_min = mesh.cellCenters[1].min()
            y_max = mesh.cellCenters[1].max()
            y_text = y_min + 0.03 * (y_max - y_min)

            for i, med in enumerate(medium):
                ax.axvline(med.x0, ls = '--', color = 'k', lw = 1)
                if med.name is None:
                    continue
                x_left = med.x0
                if i < len(medium) - 1:
                    x_right = medium[i + 1].x0
                else:
                    x_right = mesh.cellCenters[0].max()
                x_mid = 0.5 * (x_left + x_right)
                ax.text(x_mid, y_text, med.name,
                    ha='center', va='bottom', fontsize=11,
                    color='gray', alpha=0.8, transform=ax.transData)
        except Exception:
            viewer = None

    t_elapsed = 0.0
    step = 0
    T_maxes = []
    ts = []

    while t_elapsed < t:
        T.updateOld()
        T_old = T.value.copy()
        prev_res = float('inf')
        diverged = False

        while True:
            T.setValue(T_old)
            initial_res = None

            for sweep in range(1, max_sweeps + 1):
                _manual_refresh_props()
                bcs = BC.update(mesh, T)
                res = eq.sweep(var=T, dt=dt, boundaryConditions=bcs)

                T.setValue(T.value)

                if initial_res is None:
                    initial_res = res + 1e-30
                rel_res = res / initial_res

                if debug:
                    print(f"Sweep {sweep}: abs={res:.3e} rel={rel_res:.3e} dt={dt:.3e}")

                if sweep >= 2 and rel_res > 2.0:
                    dT_sweep = float(np.max(np.abs(T.value - T_old)))
                    T_min_val = float(T.value.min())
                    if dT_sweep > 100 or T_min_val < 200:
                        T.setValue(T_old)
                        dt *= 0.5
                        diverged = True
                        if debug:
                            print(f"T unstable (dT={dT_sweep:.1f}, Tmin={T_min_val:.1f}), halving dt -> {dt:.3e}")
                        break
                    elif dT_target is not None and dT_sweep > dT_target:
                        T.setValue(T_old)
                        dt *= 0.5
                        diverged = True
                        if debug:
                            print(f"Residual growing and dT={dT_sweep:.2f} > target={dT_target}, halving dt -> {dt:.3e}")
                        if dt < 1e-30:
                            raise RuntimeError(f"dt underflow at t={t_elapsed:.4e}")
                        break
                    else:
                        if debug:
                            print(f"Residual growing but T reasonable (dT={dT_sweep:.1f}), continuing")

                if sweep >= min_sweeps:
                    dT_sweep = float(np.max(np.abs(T.value - T_old)))
                    converged = rel_res < rel_residual_target
                    stagnated = abs(res - prev_res) / (prev_res + 1e-30) < 1e-3

                    if converged and dT_sweep > 1e-6:
                        break

                    if stagnated:
                        if debug:
                            print(f"Residual stagnated, accepting step (dT={dT_sweep:.2e})")
                        break

                prev_res = res

            if diverged:
                if dt < 1e-30:
                    raise RuntimeError(f"dt underflow at t={t_elapsed:.4e}")
                continue

            if dT_target is not None:
                dT_inf = float(np.max(np.abs(T.value - T_old)))
                if dT_inf <= dT_target:
                    break
                else:
                    dt *= 0.5
                    if debug:
                        print(f"dT={dT_inf:.2f} > target={dT_target}, halving dt -> {dt:.3e}")
                    if dt < 1e-30:
                        raise RuntimeError(f"dt underflow at t={t_elapsed:.4e}")
            else:
                break

        # Increment time step
        t_elapsed += dt
        step += 1

        # Ramp dt
        if dt_ramp is not None and dt < dt_max:
            dt *= dt_ramp
        if dt > dt_max:
            dt = dt_max
        if t_elapsed + dt > t:
            dt = t - t_elapsed

        # Viewer update
        if view and viewer is not None and (step % max(1, view_freq) == 0):
            scale_viewer_axes_2D(viewer,
                                 x_units=x_units, y_units=y_units,
                                 x_label="x", y_label="y",
                                 decimals=4)
            viewer.plot()

        vmin = float(T.value.min())
        vmax = float(T.value.max())
        T_maxes.append(vmax)
        ts.append(t_elapsed)

        if debug:
            plt.plot(ts, T_maxes, 'ko', label='Maximum Temperature')
            plt.ylabel("Temperature [K]", fontsize='xx-large')
            plt.xlabel("Simulation Time [s]", fontsize='xx-large')
            plt.tick_params(labelsize='x-large')
            plt.show()
            print(f"step={step}  t={t_elapsed:.4e}s  T[min,max]=[{vmin:.2f}, {vmax:.2f}]  ΔT={vmax - vmin:.2f}")

    return ts, T_maxes

# FUNCTION: Simulate heat generation for a particle beam irradiating a solid in 3D.
# Given boundary conditions and material properties.
def heateq_solid_3d(
        beam: Beam,
        medium: Medium | list[Medium],
        BC: BoundaryConditions,
        Ly: float, Lz: float,                                # YZ Dimensions of medium
        t: float,                                            # Total simulation time [s]
        T0: float = 298,                                     # Initial simulation temperature [K]
        SE=None,                                             # Pre-computed source energy term
        x_shift: float =None, y_shift=None, z_shift=None,    # How much to shift the origin by
        alpha:float =0, beta=0,                              # Beam divergence in y (alpha) and z (beta)
        dx: float = 1e-4, dy = 1e-4, dz = 1e-4,              # Cell widths and heights
        dt: float = 1e-3,                                    # Time interval between steps
        view: bool = False,                                  # Enable viewer?
        view_freq: int = 2,                                  # Update viewer every N steps
        dT_target: float = None,                             # Scale dt to achieve a specific dT per step
        dt_ramp: float = None,                               # Scaling factor to ramp dt each step
        dt_max: float = 1,                                   # Maximum value dt can ramp to
        x_units: str = 'm', y_units='m', z_units='m',        # Scale viewer axes to a specific unit
        x_scale_min: float = -np.inf, x_scale_max=np.inf,    # Range to scale viewer axes in x-dir (along beam)
        T_min: float = None, T_max = None,                   # Minimum and max values of viewer temperature
        min_sweeps: int = 3, max_sweeps=10,                  # Min/max sweeps per timestep
        rel_residual_target: float = 1e-4,                   # Relative residual to stop sweeping at
        omega: float = 0.5,                                  # Under-relaxation mixing factor
        debug: bool = False                                  # Debugging outputs
        ) -> [float, float]:

    # If only single material given, make it a list for ease of use later
    if not np.iterable(medium):
        medium = [medium]
    Lx = 0
    for med in medium:
        Lx += med.Lx

    # Making mesh
    if x_shift is None: x_shift = 0
    if y_shift is None: y_shift = -Ly / 2
    if z_shift is None: z_shift = -Lz / 2
    nx = int(np.floor(Lx / dx))
    ny = int(np.floor(Ly / dy))
    nz = int(np.floor(Lz / dz))
    mesh = Grid3D(nx=nx, dx=dx, ny=ny, dy=dy, nz=nz, dz=dz)
    mesh += ((x_shift,), (y_shift,), (z_shift,))

    # Getting cells in each direction
    cx = mesh.cellCenters[0].value
    cy = mesh.cellCenters[1].value
    cz = mesh.cellCenters[2].value
    CX = cx.reshape((nx, ny, nz), order='F')
    CY = cy.reshape((nx, ny, nz), order='F')
    CZ = cz.reshape((nx, ny, nz), order='F')

    # Mask regions of the mesh for each medium
    x = mesh.cellCenters[0]
    masks = []
    for i, med in enumerate(medium):
        xL = med.x0
        xR = medium[i + 1].x0 if i + 1 < len(medium) else (med.x0 + med.Lx)
        masks.append((x >= xL) & (x < xR))

    # Compute SE if not given
    if SE is None:
        cj = CX[:, 0, 0]
        dEb_dx = np.zeros_like(cj)  # eV/(m·s)
        E_inst = np.zeros_like(cj)
        for l in range(nx - 1):
            dEb_dx[l], E_inst[l] = compute_dEb_dx(cj[l], cj[0], dx, beam, medium)

        dEb_dx *= 1.602176634e-19  # ev to J

        phi_free = np.array(beam.PD(CX, CY, CZ, alpha, beta))
        SE = dEb_dx[:, None, None] * phi_free * 1 / (beam.E_0 * beam.I_0)

        if (debug):
            # Plot energy Gradient
            plt.title("1D Beam Energy Gradient Profile")
            plt.plot(cj * unit_factor[x_units], dEb_dx * 1e-3, 'bo', markersize=2)
            plt.xlabel(rf'x [{x_units}]', size='x-large')
            plt.ylabel(r"$dE_{b}/dx$ [$W·m^{-1}$]", size='x-large')
            ax = plt.gca()
            # Loop over regions, label each one by their name
            for i, med in enumerate(medium[1:]):
                ax.axvline(med.x0 * unit_factor[x_units], ls='--', color='k', lw=1)  # seperator
            plt.show()

            # Plot beam energy
            plt.title("Beam Instantaneous Energy")
            plt.plot(cj * unit_factor[x_units], E_inst * 1e-3, 'bo', markersize=2)
            plt.xlabel(rf'x [{x_units}]', size='x-large')
            plt.ylabel(r"$E_{inst}$ [keV]", size='x-large')
            ax = plt.gca()
            for i, med in enumerate(medium[1:]):
                differences = np.abs(cj - med.x0)
                ax.axvline(med.x0 * unit_factor[x_units], ls='--', color='k', lw=1,
                           label=r'$E_{inst} \approx$' + f"{E_inst[ differences.argmin()]*1e-3:.2f} keV")  # seperator
            plt.legend(fontsize='large')
            plt.show()

            # Plot deposited energy
            plt.plot(cj * unit_factor[x_units], SE[:, ny // 2, nz //2], 'bo', markersize=2)
            ax = plt.gca()
            for i, med in enumerate(medium[1:]):
                ax.axvline(med.x0 * unit_factor[x_units], ls='--', color='k', lw=1)  # seperator
            plt.xlabel(rf'x [{x_units}]', size='x-large')
            plt.ylabel(r"$S_{E}$ [$W\cdot m^{-3}$]", size='x-large')
            plt.title("Beam Volumetric Energy Deposition (Centerline z = y = 0)")
            plt.legend(fontsize='large')
            plt.show()
            P = float((SE.reshape(-1, order='F') * mesh.cellVolumes).sum())  # W/m
            print("Total power [W] =", P)

        SE = SE.reshape(-1, order='F')  # Need Fortran ordering

    # Create FiPy variables
    T0 = float(T0)
    SE = CellVariable(mesh=mesh, value=SE, name=r"$S_{E}$")
    T = CellVariable(mesh=mesh, value=T0, name="Temperature [K]", hasOld=True)
    rhoC = CellVariable(mesh=mesh, name="rhoC")
    k_cell = CellVariable(mesh=mesh, name="k")

    # FUNCTION: Update material properties based on current value
    def _manual_refresh_props():
        current_T = T.value # Current temp array

        new_rhoC = np.zeros(mesh.numberOfCells)
        new_k = np.zeros(mesh.numberOfCells)

        for i, med in enumerate(medium):
            # Evaluate material functions
            cp_vals = med.get_C(current_T)
            k_vals = med.get_k(current_T)

            # Use the masks to fill the arrays
            m = np.array(masks[i])
            new_rhoC[m] = med.rho * cp_vals[m]
            new_k[m] = k_vals[m]

        # Set values in FiPy variables
        rhoC.setValue(new_rhoC)
        k_cell.setValue(new_k)

    k_face = k_cell.harmonicFaceValue
    eq = TransientTerm(coeff=rhoC) == DiffusionTerm(coeff=k_face) + SE

    if (view):
        # Build a display only mesh with scaled coordinates
        mesh_view = Grid3D(nx=nx, dx=dx, ny=ny, dy=dy, nz=nz, dz=dz)
        mesh_view += ((x_shift,), (y_shift,), (z_shift,))

        vc = mesh_view.vertexCoords
        fx = unit_factor[x_units]
        fy = unit_factor[y_units]
        fz = unit_factor[z_units]

        # Scale x only within bounds, scale y and z everywhere
        mask_x = (vc[0] >= x_scale_min) & (vc[0] <= x_scale_max)
        vc[0, mask_x] *= fx
        vc[1] *= fy
        vc[2] *= fz

        T_view = CellVariable(mesh=mesh_view, value=T0, name="Temperature [K]")
        viewer = Viewer(vars=(T_view,), datamin=T_min, datamax=T_max)

    t_elapsed = 0.0
    step = 0
    T_maxes = []
    ts = []

    while t_elapsed < t:
        T.updateOld()
        T_old = T.value.copy()
        prev_res = float('inf')
        diverged = False
        while True:
            T.setValue(T_old)
            T_prev_sweep = T_old.copy()
            initial_res = None
            for sweep in range(1, max_sweeps + 1):
                # Updating values
                _manual_refresh_props()
                bcs = BC.update(mesh, T)
                res = eq.sweep(var=T, dt=dt, boundaryConditions=bcs)

                # Mixing solutions
                T.setValue(omega * T.value + (1 - omega) * T_prev_sweep)
                T_prev_sweep = T.value.copy()

                if initial_res is None:
                    initial_res = res + 1e-30

                rel_res = res / initial_res

                if debug:
                    print(f"Sweep {sweep}: abs={res:.3e} rel={rel_res:.3e} dt={dt:.3e}")

                if rel_res < rel_residual_target:
                    break

                if sweep >= 1 and rel_res > 2.0:
                    T.setValue(T_old)
                    dt *= 0.5
                    diverged = True
                    if debug:
                        print(f"Solution is diverging, halving dt -> {dt:.3e}")
                    break

                if sweep >= min_sweeps and abs(res - prev_res) / (prev_res + 1e-30) < 1e-3:
                    if debug:
                        print(f"Residual stagnated, incrementing time step")
                    break

                prev_res = res

            if diverged:
                if dt < 1e-30:
                    raise RuntimeError(f"dt underflow at t={t_elapsed:.4e}")
                continue  # retry with smaller dt

            # Check dT after sweep loop completes
            if (dT_target is not None):
                dT_inf = float(np.max(np.abs(T.value - T_old)))

                if dT_inf <= dT_target:
                    break
                else:
                    # Reject and try again with lower timestep
                    dt *= 0.5
                    if debug:
                        print(f"  dT={dT_inf:.2f} > target={dT_target}, halving dt -> {dt:.3e}")
                    if dt < 1e-30:
                        raise RuntimeError(f"dt underflow at t={t_elapsed:.4e}")
            else:
                break

        # Increment time step
        t_elapsed += dt
        step += 1

        # Ramp dt
        if (dt_ramp is not None and dt < dt_max):
            dt *= dt_ramp
        if (dt > dt_max):
            dt = dt_max
        if (t_elapsed + dt > t):
            dt = t - t_elapsed

        # Viewer update every view_freq steps
        if view and (step % view_freq == 0):
            T_view.setValue(T.value)
            viewer.plot()

        vmin = float(T.value.min())
        vmax = float(T.value.max())
        T_maxes.append(vmax)
        ts.append(t_elapsed)

        if debug:
            plt.plot(ts, T_maxes, 'ko', label='Maximum Temperature')
            plt.ylabel("Temperature [K]", fontsize='xx-large')
            plt.xlabel("Simulation Time [s]", fontsize='xx-large')
            plt.tick_params(labelsize='x-large')
            plt.show()
            print(f"step={step}  t={t_elapsed:.4e}s  T[min,max]=[{vmin:.2f}, {vmax:.2f}]  ΔT={vmax - vmin:.2f}")

    return ts, T_maxes


def heateq_solid_3d_gmsh(
            beam: Beam,
            medium: Medium | list[Medium],
            mesh: Gmsh3D,
            BC: BoundaryConditionsGmsh,
            t: float,                                                          # Total simulation time [s]
            T0: float = 298.0,                                                 # Initial simulation temperature [K]
            SE = None,                                                         # Pre-computed source energy term
            n_SE: float = 5000,                                                # Number of points for SE interpolation
            T_min: float = None, T_max = None,                                 # Temperature bounds for viewer
            x_scale: float = 1.0, x_scale_min = -np.inf, x_scale_max= np.inf,  # X viewer scaling
            alpha: float = 0.0, beta = 0.0,                                    # Beam divergence in y (alpha) and z (beta)
            dt: float = 1e-3,                                                  # Time interval between steps
            view: bool = True,                                                 # Enable viewer?
            view_freq: int = 1,                                                # Update viewer every N steps
            dT_target: float = None,                                           # Scale dt to achieve a specific dT per step
            dt_ramp: float = None,                                             # Scaling factor to ramp dt each step
            dt_max: float = 5.0,                                               # Maximum value dt can ramp to
            min_sweeps: int = 3, max_sweeps = 10,                              # Min/max sweeps per timestep
            rel_residual_target: float = 1e-4, abs_residual_target = 1e-3,     # Relative residual to stop sweeping at
            omega: float = 0.5,                                                # Under relaxation mixing factor
            debug: bool = False                                                # Debugging outputs
            ) -> tuple[list[Any], list[Any]]:


    if not np.iterable(medium):
        medium = [medium]

    # Cell center coords
    x = mesh.cellCenters[0].value
    y = mesh.cellCenters[1].value
    z = mesh.cellCenters[2].value
    nCells = mesh.numberOfCells

    # Store material (volume) names
    tag_map = mesh.physicalCells  # (nCells,)
    names = []
    for name in mesh.physicalCells:
        names.append(name)

    # Create masks for materials
    masks = []
    for med in medium:
        if med.name not in names:
            raise KeyError(f"Medium name '{med.name}' not in list of physical volumes = {names}")
        m = tag_map[med.name]
        masks.append(m)
        # Use face coords to get actual boundary
        face_mask = np.zeros(mesh.numberOfFaces, dtype=bool)
        face_mask[mesh.cellFaceIDs[:, np.where(m)[0]]] = True
        med.x0 = float(mesh.faceCenters[0][face_mask].min())

    if (debug):
        covered = np.zeros(nCells, dtype=int)
        for m in masks:
            covered += m.astype(int)
        if (covered == 0).any() or (covered > 1).any():
            raise RuntimeError(
                f"Material masks bad: uncovered={(covered == 0).sum()} overlap={(covered > 1).sum()} "
                f"tags={np.unique(tag_map)}"
            )

    # Compute SE if not given
    if SE is None:
        # 1D profile along +x — include material boundaries in grid
        x_boundaries = [med.x0 for med in medium if med.x0 >= 0]
        x_grid = np.sort(np.unique(np.concatenate([
            np.linspace(0.0, float(x.max()), n_SE),
            x_boundaries
        ])))

        dEb_dx = np.zeros_like(x_grid)
        E_inst = np.zeros_like(x_grid)
        E_inst[0] = beam.E_0

        for i in range(len(x_grid) - 1):
            dx_step = (x_grid[i + 1] - x_grid[i]) / 2
            dEb_dx[i + 1], E_inst[i + 1] = compute_dEb_dx(x_grid[i + 1], x_grid[i], dx_step, beam, medium, E=E_inst[i])

        dEb_dx[0] = dEb_dx[1]

        # Build cumulative integral of dEb_dx along x_grid
        Eb_cumul = np.zeros_like(x_grid)
        Eb_cumul[1:] = cumulative_trapezoid(dEb_dx, x_grid)

        # Build separate interpolators between each material boundary
        x_breaks = sorted(set([x_grid[0]] + x_boundaries + [x_grid[-1]]))
        Eb_interps = []
        for k in range(len(x_breaks) - 1):
            mask = (x_grid >= x_breaks[k]) & (x_grid <= x_breaks[k + 1])
            Eb_interps.append((x_breaks[k], x_breaks[k + 1],
                               PchipInterpolator(x_grid[mask], Eb_cumul[mask])))

        def Eb_cumul_eval(xs):
            out = np.zeros_like(xs)
            for x_lo, x_hi, interp in Eb_interps:
                m = (xs >= x_lo) & (xs <= x_hi)
                out[m] = interp(xs[m])
            return out

        # For each cell, find its cell edges in x direction
        x_unique = np.sort(np.unique(x))
        x_mid = 0.5 * (x_unique[:-1] + x_unique[1:])
        x_edges = np.r_[
            2 * x_unique[0] - x_mid[0],
            x_mid,
            2 * x_unique[-1] - x_mid[-1]
        ]

        # Map each cell to its "bin"
        bin_idx = np.searchsorted(x_mid, x)
        cell_x_left = x_edges[bin_idx]
        cell_x_right = x_edges[bin_idx + 1]

        # Clamp cell edges to x_grid range
        cell_x_left = np.clip(cell_x_left, x_grid[0], x_grid[-1])
        cell_x_right = np.clip(cell_x_right, x_grid[0], x_grid[-1])

        # Clamp (again) cell edges to not cross material boundaries
        x_breaks_arr = np.array(x_boundaries)
        for xb in x_breaks_arr:
            cross = (cell_x_left < xb) & (cell_x_right > xb)
            cell_x_right[cross & (x < xb)] = xb
            cell_x_left[cross & (x >= xb)] = xb
        dx_cell = cell_x_right - cell_x_left

        # Average the dEb_dx across the cell from its left and right edges
        dEb_dx_cells = (Eb_cumul_eval(cell_x_right) - Eb_cumul_eval(cell_x_left)) / (dx_cell + 1e-30)
        dEb_dx_cells = np.maximum(dEb_dx_cells, 0.0) # just in case
        dEb_dx_cells *= 1.602176634e-19  # eV -> J

        # Get heat generation
        phi_free = np.asarray(beam.PD(x, y, z, alpha, beta))
        SE_cells = dEb_dx_cells * phi_free / (beam.E_0 * beam.I_0)

        if debug:
            # Find where beam stops (first zero)
            non_zero = np.where(dEb_dx > 0)[0]
            if len(non_zero) > 0:
                i_stop = min(non_zero[-1] + 3, len(x_grid))
                x_cutoff = x_grid[i_stop - 1]
            else:
                i_stop = len(x_grid)
                x_cutoff = x_grid[-1]

            plt.figure()
            plt.title("1D Beam Energy Gradient Profile")
            plt.plot(x_grid[:i_stop], dEb_dx[:i_stop], "b.")
            colors = plt.cm.tab10.colors
            j_color = 0
            for med in medium:
                if med.x0 < 0:
                    continue
                plt.axvline(med.x0, ls='--', lw=1, color=colors[j_color], label=med.name)
                j_color += 1
            plt.xlabel("x [m]", size='x-large')
            plt.ylabel(r"$dE_{b}/dx$ $[eV/m · s^{-1}$)]", size='x-large')
            plt.legend(fontsize='large', loc='upper left')
            plt.show()

            plt.figure()
            plt.title(r"Normalized Volumetric Source Heating Term $S_{E}$")
            mask_beam = (x >= 0) & (x <= x_cutoff) & (np.abs(y) <= beam.L / 2) & (np.abs(z) <= beam.W / 2)
            plt.plot(x[mask_beam], SE_cells[mask_beam], "b.", markersize=2)
            j_color = 0
            for med in medium:
                if med.x0 < 0:
                    continue
                plt.axvline(med.x0, ls='--', lw=1, color=colors[j_color], label=med.name)
                j_color += 1
            plt.xlabel("x [m]", size='x-large')
            plt.ylabel(r"$S_{E}$ $[\frac{W}{m^3}]$", size='x-large')
            plt.legend(fontsize='large', loc='upper left')
            plt.show()

            P = float((SE_cells * mesh.cellVolumes).sum())
            print("Total power [W] =", P)

    else:
        SE_cells = np.asarray(SE, dtype=float)
        if SE_cells.shape != (nCells,):
            raise ValueError(f"Provided SE must be shape {(nCells,)}, got {SE_cells.shape}")

    # FiPy variables
    T = CellVariable(mesh=mesh, value=float(T0), name="Temperature [K]", hasOld=True)
    SE_var = CellVariable(mesh=mesh, value=SE_cells, name=r"$S_E$")
    rhoC = CellVariable(mesh=mesh, name="rhoC")
    k_cell = CellVariable(mesh=mesh, name="k")

    # FUNCTION: Refresh the material properties based on new temperature
    def _manual_refresh_props():
        # Initialize variables to hold values
        current_T = T.value
        new_rhoC = np.zeros(nCells)
        new_k = np.zeros(nCells)

        # Update values
        for med, m in zip(medium, masks):
            cp_vals = med.get_C(current_T)
            k_vals = med.get_k(current_T)
            new_rhoC[m] = med.rho * cp_vals[m]
            new_k[m] = k_vals[m]
        rhoC.setValue(new_rhoC)
        if new_rhoC.max() > 1e10:
            bad = np.where(new_rhoC > 1e10)[0]
            print(f"WARNING: rhoC blowup at cells {bad[:5]}, T={current_T[bad[:5]]}, rhoC={new_rhoC[bad[:5]]}")
        k_cell.setValue(new_k)

        # Apply contact resistance at internal interfaces
        k_face_vals = k_cell.harmonicFaceValue.value.copy()
        j_R = 0
        for face_name in mesh.physicalFaces:   # Get internal interface face masks
            face_name_check = face_name.replace(" ", "").lower()
            if 'internal' in face_name_check:
                # Find cell center distances across interface faces
                interface_mask = mesh.physicalFaces[face_name]
                dx = mesh._cellDistances

                # Effective k including contact resistance
                # 1/k_eff = 1/k_harmonic + R/dx --> k_eff = 1/(1/k_harmonic + R/dx)
                k_face_vals[interface_mask] = 1.0 / (
                        1.0 / (k_face_vals[interface_mask] + 1e-30)
                        + (BC.R[j_R]) / dx[interface_mask]
                )
                j_R += 1
        k_face.setValue(k_face_vals)

    # Enable the viewer object
    if(view):
        vertex_coords = mesh.vertexCoords
        mask_x = (vertex_coords[0] >= x_scale_min) & (vertex_coords[0] <= x_scale_max)
        vertex_coords[0, mask_x] *= x_scale
        viewer = Viewer(vars=(T,), datamin = T_min, datamax = T_max)

    # Initialize properties and define equation
    k_face = FaceVariable(mesh=mesh, value=0.0)
    _manual_refresh_props()

    # Initialize equations
    BC.init_implicit_bbr(mesh)

    # Time loop
    t_elapsed = 0.0
    step = 0
    T_maxes = []
    ts = []

    T_old_prev = T.value.copy()
    dt_prev = dt
    jiggle_idx = 0

    while t_elapsed < t:
        T.updateOld()
        T_old = T.value.copy()

        while True:
            T.setValue(T_old)
            T_prev_sweep = T_old.copy()
            initial_res = None
            prev_res = float('inf')
            diverged = False

            BC.init_implicit_bbr(mesh)
            eq = (TransientTerm(coeff=rhoC)
                  == DiffusionTerm(coeff=k_face)
                  + SE_var
                  - BC._bbr_h_cell * T
                  + BC._bbr_src_cell)

            for sweep in range(1, max_sweeps + 1):
                _manual_refresh_props()
                bcs = BC.update(mesh, T)
                res = eq.sweep(var=T, dt=dt, boundaryConditions=bcs)

                T.setValue(omega * T.value + (1 - omega) * T_prev_sweep)
                T_prev_sweep = T.value.copy()

                if initial_res is None:
                    initial_res = res + 1e-30
                rel_res = res / initial_res

                if debug:
                    print(f"Sweep {sweep}: abs = {res:.3e} rel = {rel_res:.3e} dt = {dt:.3e}")

                if sweep >= 2 and rel_res > 2.0:
                    dT_sweep = float(np.max(np.abs(T.value - T_old)))
                    T_min_val = float(T.value.min())
                    if dT_sweep > 100 or T_min_val < 200:
                        T.setValue(T_old)
                        dt *= 0.5
                        diverged = True
                        if debug:
                            print(f"T unstable (dT = {dT_sweep:.1f}, Tmin = {T_min_val:.1f}), halving dt -> {dt:.3e}")
                        break
                    elif dT_target is not None and dT_sweep > dT_target:
                        T.setValue(T_old)
                        dt *= 0.5
                        diverged = True
                        if debug:
                            print(
                                f"Residual growing and dT = {dT_sweep:.2f} > target = {dT_target}, halving dt -> {dt:.3e}")
                        if dt < 1e-30:
                            raise RuntimeError(f"dt underflow at t = {t_elapsed:.4e}")
                        break
                    else:
                        if debug:
                            print(f"Residual growing but T reasonable (dT = {dT_sweep:.1f}), continuing")

                if sweep >= min_sweeps:
                    dT_sweep = float(np.max(np.abs(T.value - T_old)))
                    converged = rel_res < rel_residual_target or res < abs_residual_target
                    stagnated = abs(res - prev_res) / (prev_res + 1e-30) < 1e-3

                    if converged and dT_sweep > 1e-6:
                        break

                    if (converged or stagnated) and dT_sweep < 1e-6:
                        jiggle = [1.73, 0.37, 2.41, 0.53, 3.17, 0.19]
                        jig = jiggle[jiggle_idx % len(jiggle)]
                        jiggle_idx += 1

                        T.setValue(T_old_prev)
                        T.updateOld()
                        T_old = T_old_prev.copy()
                        t_elapsed -= dt_prev
                        dt *= jig
                        dt = min(dt, dt_max)
                        diverged = True
                        if debug:
                            print(f"Stuck -> rolling back 1 step and randomizing dt -> {dt:.3e}")
                        break

                    if stagnated:
                        if debug:
                            print(f"Residual stagnated, accepting step (dT = {dT_sweep:.2e})")
                        break

                prev_res = res

            if diverged:
                if dt < 1e-30:
                    raise RuntimeError(f"dt underflow at t = {t_elapsed:.4e}")
                continue

            if dT_target is not None:
                dT_inf = float(np.max(np.abs(T.value - T_old)))
                if dT_inf <= dT_target:
                    break
                else:
                    dt *= 0.5
                    if debug:
                        print(f"dT = {dT_inf:.2f} > target = {dT_target}, halving dt -> {dt:.3e}")
                    if dt < 1e-30:
                        raise RuntimeError(f"dt underflow at t = {t_elapsed:.4e}")
            else:
                break

        # Save state for potential rollback next step
        T_old_prev = T_old.copy()
        dt_prev = dt
        jiggle_idx = 0  # reset on successful step

        # Increment time step
        t_elapsed += dt
        step += 1
        if dt_ramp is not None and dt < dt_max:
            dt *= dt_ramp
        if dt > dt_max:
            dt = dt_max
        if t_elapsed + dt > t:
            dt = t - t_elapsed

        # Increment time step
        t_elapsed += dt
        step += 1
        if dt_ramp is not None and dt < dt_max:
            dt *= dt_ramp
        if dt > dt_max:
            dt = dt_max
        if t_elapsed + dt > t:
            dt = t - t_elapsed

        vmin = float(T.value.min())
        vmax = float(T.value.max())

        if view and (step % max(1, view_freq) == 0):
            viewer.plot()

        if debug:
            plt.plot(ts, T_maxes, 'ko', label='Maximum Temperature')
            plt.ylabel("Temperature [K]", fontsize='xx-large')
            plt.xlabel("Simulation Time [s]", fontsize='xx-large')
            plt.tick_params(labelsize='x-large')
            plt.show()
            print(f"step = {step}  t = {t_elapsed:.4e}s  T[min,max] = [{vmin:.2f}, {vmax:.2f}]  ΔT = {vmax - vmin:.2f}")

        T_maxes.append(vmax)
        ts.append(t_elapsed)

    return ts, T_maxes