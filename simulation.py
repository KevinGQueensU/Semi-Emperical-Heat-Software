from Medium import Medium
from Beam import Beam
from BoundaryConditions import BoundaryConditions
import numpy as np
from collections.abc import Callable
import bisect
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (needed for 3D)
from fipy import CellVariable, Grid2D, Grid3D, TransientTerm, DiffusionTerm, Viewer
from fipy.tools import numerix as nx
_SIGMA_SB = 5.670374419e-8  # Stefan–Boltzmann constant [W/(m^2·K^4)]

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

    # Valide planes
    plane = plane.lower()
    if plane not in ("xy", "xz", "yz"):
        raise ValueError("plane must be one of: 'xy', 'xz', 'yz'")

    x_label_char, y_label_char = {
        "xy": ("x", "y"),
        "xz": ("x", "z"),
        "yz": ("y", "z"),
    }[plane]

    # Unit multipliers
    unit_factor = {
        "m": 1.0,
        "cm": 1e2,
        "mm": 1e3,
        "um": 1e6, "µm": 1e6,
        "nm": 1e9,
        "km": 1e-3,
    }
    ux, uy = units
    if ux not in unit_factor or uy not in unit_factor:
        raise ValueError("Unsupported units. Use one of: m, cm, mm, um/µm, nm, km")

    fx = unit_factor[ux]
    fy = unit_factor[uy]

    # Keep current limits
    if preserve_limits:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

    # Unit-scaled tick formatters
    def _mk_formatter(scale, sig):
        fmt = "{:." + str(sig) + "g}"
        return ticker.FuncFormatter(lambda val, pos: fmt.format(val * scale))

    ax.xaxis.set_major_formatter(_mk_formatter(fx, decimals))
    ax.yaxis.set_major_formatter(_mk_formatter(fy, decimals))

    # Labels + font sizes
    ax.set_xlabel(f"{x_label_char} [{ux}]", fontsize=labelsize)
    ax.set_ylabel(f"{y_label_char} [{uy}]", fontsize=labelsize)
    ax.tick_params(axis="both", labelsize=ticklabelsize)

    # Equal aspect in displayed units ----
    if equal_aspect:
        ax.set_aspect("equal", adjustable="box")

    # Redraw and restore limits
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
def region_index(x0s: np.ndarray[float] | list[float] | float, # x0s that separate regions
                 x: float                  # Position where you want to find the region its in
                 ) -> float:
    # x0 must be sorted (strictly increasing recommended)
    m = len(x0s)
    if m < 2:
        # With fewer than 2 breakpoints, only "(x0[-1], +∞)" exists
        return -1 if x <= x0s[-1] else 0

    if x < x0s[0]:
        return -1

    i = bisect.bisect_left(x0s, x)  # index of first breakpoint >= x

    if i == m - 1:
        # x >= last breakpoint
        return (m - 1) if x > x0s[-1] else (m - 2)
    else:
        # x in [x0[i], x0[i+1])
        return i

# FUNCTION: Compute the energy deposition gradient at a specific point given a beam and target medium
def compute_dEb_dx( x: float,        # Position to compute SE at (beam is fired along x-axis)
                    x_ref: float,    # Initial x position where values are known
                    dx: float,       # Step size along x
                    beam: Beam,      # Particle beam shot along x-direction
                    medium: Medium | list[Medium],   # Target medium(s) that particle beam is being shot at
                    ):
    if x_ref > x:
        dx = -abs(dx)

    if not np.iterable(medium):
        medium = np.array([medium])

    x_med = np.asarray([med.x0 for med in medium])

    E_inst = float(beam.E_0)   # eV per particle
    I_beam = float(beam.I_0)   # 1/s
    E_beam = E_inst * I_beam
    xi = float(x_ref)
    med_i = 0

    while True:
        if (dx > 0 and xi >= x) or (dx < 0 and xi <= x):
            break

        # Step everything
        dEdx = medium[med_i].get_dEdx(E_inst)
        dIdx = medium[med_i].get_dIdx(E_inst, I_beam)
        dEdx_beam = dEdx * I_beam + E_inst * dIdx
        I_beam = I_beam + dIdx * dx
        E_beam = max(E_beam + dEdx_beam * dx, 0)
        E_inst = E_beam / I_beam
        xi += dx

        if E_inst <= 0.0:
            break

    dEdx = medium[med_i].get_dEdx(E_inst)

    return -(I_beam * dEdx)     # Energy deposition per length [eV/(m*s)]


#### SIMULATION FUNCTIONS ####

# FUNCTION: Simulate heat generation for a particle beam irradiating a solid in 2D.
# Given boundary conditions and material properties.
def heateq_solid_2d(
        beam: Beam,
        medium: Medium,
        BC: BoundaryConditions,
        Lx: float, Ly: float,       # Dimensions of medium
        rho: float,                 # Target material bulk density [kg/m^3]
        C_f: float | Callable[[float], float], # Heat capacity [J/(kg·K)]: can be static value or function Cp(T)
        k_f: float | Callable[[float], float], # Heat conductivity [W/(m·K)]: k(T): can be static value or function k(T)
        t: float,               # Total simulation time [s]
        T0: float = 298,            # OPTIONAL: Initial simulation temperature [K]
        SE = None,                  # OPTIONAL: Can give a pre-computed source energy term, otherwise it will compute it for you
        x_shift=None, y_shift=None, # OPTIONAL: How much to shift the origin by
        alpha = 0, beta = 0,        # OPTIONAL: Beam divergence in y (alpha) and z (beta) directions
        dx: float = 1e-4, dy: float = 1e-4, # OPTIONAL: Cell widths and heights
        dt: float = 1e-3,           # OPTIONAL: Time interval between steps
        view: bool = False,         # OPTIONAL: Enable viewer?
        view_freq: int = 20,        # OPTIONAL: Update viewer every N steps
        dT_target: float = None,    # OPTIONAL: Scale dt so that a specific dT between steps can be achieved
        dt_ramp: float = None,      # OPTIONAL: Scaling factor to ramp dt by every step
        dt_max: float = 1,          # OPTIONAL: Set a maximum value that dt can ramp to
        x_units: str = 'mm', y_units: str = 'mm' # OPTIONAL: Scale viewer axes to a specific unit
        ):

    # Create the mesh
    nx_cells = int(nx.round(Lx / dx))
    ny_cells = int(nx.round(Ly / dy))
    mesh = Grid2D(dx=dx, dy=dy, nx=nx_cells, ny=ny_cells)
    if x_shift is None:
        x_shift = 0
    if y_shift is None:
        y_shift = -Ly / 2

    mesh += ((x_shift,), (y_shift,))  # shift mesh
    cx = mesh.cellCenters[0].value
    cy = mesh.cellCenters[1].value
    CX = cx.reshape((nx_cells, ny_cells), order='F')  # shape (nx, ny)
    CY = cy.reshape((nx_cells, ny_cells), order='F')

    # Compute SE if not given
    if SE is None:
        cj = CX[:, 0]
        dEb_dx = np.zeros_like(cj)  # eV/(m·s)

        for l in range(nx_cells - 1):
            dEb_dx[l] = compute_dEb_dx(cj[l], cj[0], dx, beam, medium)
        dEb_dx *= 1.602176634e-19  # ev to J
        plt.plot(cj, dEb_dx)
        plt.show()
        phi_free = np.array(beam.PD(CX, CY, 0, alpha, beta))

        SE = dEb_dx[:, None] * phi_free * 1 / (beam.E_0 * beam.I_0)
        SE = SE.reshape(-1, order='F')

    T0 = float(T0)
    SE = CellVariable(mesh=mesh, value=SE, name=r"$S_{E}$")
    T = CellVariable(mesh=mesh, value=T0, name="Temperature [K]", hasOld=True)

    rhoC = CellVariable(mesh=mesh, name="rho*C", value=rho * float(C_f(T0)))
    k_cell = CellVariable(mesh=mesh, name="k(T)", value=float(k_f(T0)))

    def _refresh_material_props():
        Cp_val = C_f(T)   # J/(kg·K)
        k_val  = k_f(T)   # W/(m·K)
        rhoC.setValue(rho * Cp_val)
        k_cell.setValue(k_val)

    _refresh_material_props()

    eq = TransientTerm(coeff=rhoC) == DiffusionTerm(coeff=k_cell) + SE


    viewer = None
    if view:
        try:
            viewer = Viewer(vars=(T,), title="Temperature Distribution")
        except Exception:
            viewer = None

    t_elapsed = 0.0
    step = 0

    while t_elapsed < t:
        T.updateOld()
        _refresh_material_props()
        T_old = T.value.copy()
        bcs = BC.update(mesh, T)

        if dT_target is not None:
            while True:
                eq.solve(var=T, dt=dt, boundaryConditions = bcs)
                dT_inf = np.max(np.abs(T.value - T_old))
                if dT_inf > dT_target:
                    print("Time step too large for required dT, reducing by 50%")
                    dt *= 0.5
                else:
                    break
        else:
            eq.solve(var=T, dt=dt, boundaryConditions= bcs)

        t_elapsed += dt
        if (dt_ramp is not None and dt < dt_max):
            dt *= dt_ramp
        if (dt > dt_max):
            dt = dt_max
        if (t_elapsed + dt > t):
            dt = t - t_elapsed

        # Troubleshooting stuff
        if (step % max(1, view_freq)) == 0:
            Tamb = float(BC.T_amb)
            err = float(np.max(np.abs(T.value - Tamb)))
            print(f"t={t_elapsed:.3f}s  Tmax={T.value.max():.6f}  Tmin={T.value.min():.6f}  max|T-Tamb|={err:.6e}")

        if viewer is not None:
            Tmin = float(T.value.min())
            Tmax = float(T.value.max())
            print(f"t={t_elapsed + dt:0.3f}s  T[min,max]=[{Tmin:.2f}, {Tmax:.2f}]")
            scale_viewer_axes_2D( viewer,
                                  x_units=x_units, y_units=y_units,
                                  x_label="x", y_label="y",
                                  decimals=4)
            viewer.plot()
        step += 1

# FUNCTION: Simulate heat generation for a particle beam irradiating a solid in 3D.
# Given boundary conditions and material properties.
def heateq_solid_3d(beam: Beam,
                    medium: Medium,
                    BC: BoundaryConditions,
                    Lx: float, Ly: float, Lz: float,        # Dimensions of medium
                    rho: float,             # Target material bulk density [kg/m^3]
                    C_f: float | Callable[[float], float],  # Heat capacity [J/(kg·K)]: can be static value or function Cp(T)
                    k_f: float | Callable[[float], float],  # Heat conductivity [W/(m·K)]: k(T): can be static value or function k(T)
                    t: float,               # Total simulation time [s]
                    T0: float = 298,        # OPTIONAL: Initial simulation temperature [K]
                    SE=None,                # OPTIONAL: Can give a pre-computed source energy term, otherwise it will compute it for you
                    x_shift=None, y_shift=None, z_shift = None,  # OPTIONAL: How much to shift the origin by
                    alpha=0, beta=0,        # OPTIONAL: Beam divergence in y (alpha) and z (beta) directions
                    dx: float = 1e-4, dy: float = 1e-4, dz: float = 1e-4,           # OPTIONAL: Cell widths and heights
                    dt: float = 1e-3,       # OPTIONAL: Time interval between steps
                    view: bool = False,     # OPTIONAL: Enable viewer?
                    view_freq: int = 2,     # OPTIONAL: Update viewer every N steps
                    dT_target: float = None,# OPTIONAL: Scale dt so that a specific dT between steps can be achieved
                    dt_ramp: float = None,  # OPTIONAL: Scaling factor to ramp dt by every step
                    dt_max: float = 1,      # OPTIONAL: Set a maximum value that dt can ramp to
                    x_units: str = 'mm', y_units: str = 'mm', z_units: str = 'mm'  # OPTIONAL: Scale viewer axes to a specific unit
                    ):
    # Making mesh
    if x_shift is None: x_shift = 0
    if y_shift is None: y_shift = -Ly / 2
    if z_shift is None: z_shift = -Lz / 2
    import numpy as np
    nx = int(np.ceil(Lx / dx))
    ny = int(np.ceil(Ly / dy))
    nz = int(np.ceil(Lz / dz))
    mesh = Grid3D(nx=nx, dx=dx, ny=ny, dy=dy, nz=nz, dz=dz)
    mesh += ((x_shift,), (y_shift,), (z_shift,))

    # Getting cells in each direction
    cx = mesh.cellCenters[0].value
    cy = mesh.cellCenters[1].value
    cz = mesh.cellCenters[2].value
    CX = cx.reshape((nx, ny, nz), order='F')
    CY = cy.reshape((nx, ny, nz), order='F')
    CZ = cz.reshape((nx, ny, nz), order='F')

    # Compute SE if not given
    if SE is None:
        cj = CX[:, 0, 0]
        dEb_dx = np.zeros_like(cj)  # eV/(m·s)

        for l in range(nx - 1):
            dEb_dx[l] = compute_dEb_dx(cj[l], cj[0], dx, beam, medium)

        dEb_dx *= 1.602176634e-19  # ev to J

        phi_free = np.array(beam.PD(CX, CY, CZ, alpha, beta))
        SE = dEb_dx[:, None, None] * phi_free * 1 / (beam.E_0 * beam.I_0)
        SE = SE.reshape(-1, order='F')

    SE = CellVariable(mesh=mesh, value=SE, name=r"$S_{E}$")
    T = CellVariable(mesh=mesh, value=float(T0), hasOld=True, name="T [K]")

    # Material properties
    if callable(k_f):
        K = CellVariable(mesh=mesh, value=k_f(T))
    else:
        K = CellVariable(mesh=mesh, value=k_f, rank=0)
    if callable(C_f):
        rho_C = CellVariable(mesh=mesh, value=rho * C_f(T))
    else:
        rho_C = CellVariable(mesh=mesh, value=rho * C_f, rank=0)

    if view:
        # Three orthogonal mid-slices: xy, xz, yz
        ix, iy, iz = 0, ny // 2, nz // 2
        x0, x1 = x_shift, x_shift + Lx
        y0, y1 = y_shift, y_shift + Ly
        z0, z1 = z_shift, z_shift + Lz

        # Use manual spacing
        fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.5), constrained_layout= True)

        A = T.value.reshape((nx, ny, nz), order='F')

        # Plot three slices
        im_xy = axes[0].imshow(
            A[:, :, iz].T, origin='lower',
            extent=[x0, x1, y0, y1], aspect='equal', cmap='turbo'
        )
        axes[0].set_title('XY plane at Lz/2', fontsize=14)

        im_xz = axes[1].imshow(
            A[:, iy, :].T, origin='lower',
            extent=[x0, x1, z0, z1], aspect='equal', cmap='turbo'
        )
        axes[1].set_title('XZ plane at Ly/2', fontsize=14)

        im_yz = axes[2].imshow(
            A[ix, :, :].T, origin='lower',
            extent=[y0, y1, z0, z1], aspect='equal', cmap='turbo'
        )
        axes[2].set_title('YZ plane at x = 0', fontsize=14)

        # Scale/label each panel correctly
        scale_axes(axes[0], plane="xy", units=(x_units, y_units), decimals=3,
                   equal_aspect=True, preserve_limits=True, labelsize=14, ticklabelsize=12)
        scale_axes(axes[1], plane="xz", units=(x_units, z_units), decimals=3,
                   equal_aspect=True, preserve_limits=True, labelsize=14, ticklabelsize=12)
        scale_axes(axes[2], plane="yz", units=(y_units, z_units), decimals=3,
                   equal_aspect=True, preserve_limits=True, labelsize=14, ticklabelsize=12)

        # Colorbar w/ larger label & ticks
        cbar = plt.colorbar(im_yz, ax=axes.ravel().tolist(), shrink=0.9)
        cbar.set_label('T [K]', fontsize=14)
        cbar.ax.tick_params(labelsize=12)

        plt.show(block=False)

    eq = TransientTerm(coeff=rho_C, var=T) == DiffusionTerm(coeff=K, var=T) + SE

    t_elapsed = 0.0
    step = 0
    T_maxes = []
    ts = []

    if view:
        plt.ion()
        fig.canvas.draw_idle()
        plt.pause(0.001)

    while t_elapsed < t:
        T.updateOld()

        if callable(k_f):
            K.setValue(k_f(T))
        if callable(C_f):
            rho_C.setValue(rho * C_f(T))
        bcs = BC.update(mesh, T)

        T_old = T.value.copy()

        if dT_target is not None:
            while True:
                eq.solve(var=T, dt=dt, boundaryConditions=bcs)
                dT_inf = np.max(np.abs(T.value - T_old))
                if dT_inf > dT_target:
                    print("Time step too large for required dT, reducing by 50%")
                    dt *= 0.5
                else:
                    break
        else:
            eq.solve(var=T, dt=dt, boundaryConditions=bcs)

        t_elapsed += dt
        step += 1

        # Ramp dt
        if (dt_ramp is not None and dt < dt_max):
            dt *= dt_ramp
        if (dt > dt_max):
            dt = dt_max
        if (t_elapsed + dt > t):
            dt = t - t_elapsed

        # Viewer update every view_freq steps ----
        if view and (step % view_freq == 0):
            A = T.value.reshape((nx, ny, nz), order='F')

            im_xy.set_data(A[:, :, iz].T)
            im_xz.set_data(A[:, iy, :].T)
            im_yz.set_data(A[ix, :, :].T)

            vmin = A.min()
            vmax = A.max()
            im_xy.set_clim(vmin, vmax)
            im_xz.set_clim(vmin, vmax)
            im_yz.set_clim(vmin, vmax)

            fig.canvas.draw_idle()
            plt.pause(0.001)

            T_maxes.append(vmax)
            ts.append(t_elapsed)
            print(f"step={step}  t={t_elapsed:.4e}s  T[min,max]=[{vmin:.2f}, {vmax:.2f}]  ΔT={vmax - vmin:.2f}")

    if view:
        plt.ioff()
        plt.show()

    return T_maxes, ts