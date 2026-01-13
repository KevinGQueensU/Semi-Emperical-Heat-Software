from Medium import Medium
from Beam import Beam
from Boundary_Conditions import BoundaryConditions
import numpy as np
import bisect
import scipy.constants as const
from fipy import FaceVariable, CellVariable, Grid2D, Grid3D, TransientTerm, DiffusionTerm, Viewer
from fipy.boundaryConditions import FixedValue, FixedFlux
from fipy.tools import numerix as nx
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (needed for 3D)

def region_index(x0, x):
    # x0 must be sorted (strictly increasing recommended)
    m = len(x0)
    if m < 2:
        # With fewer than 2 breakpoints, only "(x0[-1], +∞)" exists
        return -1 if x <= x0[-1] else 0

    if x < x0[0]:
        return -1
    i = bisect.bisect_left(x0, x)  # index of first breakpoint >= x

    if i == m - 1:
        # x >= last breakpoint
        return (m - 1) if x > x0[-1] else (m - 2)
    else:
        # x in [x0[i], x0[i+1])
        return i

def compute_SE(x, y, z, alpha, beta, x_ref, beam: Beam,  mediums, dx = 0.1):
    dEddx = 0
    if(x_ref > x):
        dx = -np.abs(dx)

    freeE_flux = beam.PD(x, y, z, alpha, beta)
    E_beam = beam.E_0
    I_beam = beam.I_0
    x_med = mediums.x0
    xi = x_ref
    condition = False
    while(condition == False):
        if(dx < 0 and xi < x) or (dx < 0 and xi < x):
            xi = x
            condition = True
        med_i = region_index(x_med, xi)
        dEddx = mediums[med_i].get_Egrad(xi, dx, E_beam, I_beam)
        dIdx = mediums[med_i].get_dIdx(xi, E_beam, I_beam)
        E_beam = E_beam - dEddx*dx
        I_beam = I_beam + dIdx*dx
        xi = xi + dx
    SE = freeE_flux * (1/beam.E_0) * dEddx
    return SE

def scale_fipy_viewer_axes(viewer,
                           x_units="mm", y_units="mm",
                           x_label="x", y_label="y",
                           decimals=3, equal_aspect=True):

    # 1) get the underlying Matplotlib axes from the viewer
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

    # 3) formatters that multiply the underlying meter values
    def _mk_formatter(f, sig):
        fmt = "{:." + str(sig) + "g}"
        return ticker.FuncFormatter(lambda val, pos: fmt.format(val * f))

    ax.xaxis.set_major_formatter(_mk_formatter(fx, decimals))
    ax.yaxis.set_major_formatter(_mk_formatter(fy, decimals))

    # 4) axis labels with units
    ax.set_xlabel(f"{x_label} [{x_units}]")
    ax.set_ylabel(f"{y_label} [{y_units}]")

    # 5) optionally enforce equal aspect in displayed units
    if equal_aspect:
        ax.set_aspect("equal", adjustable="box")

    # 6) redraw
    return ax

def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect);


_SIGMA_SB = 5.670374419e-8  # Stefan–Boltzmann constant [W/(m^2·K^4)]

def heateq_solid_2d(
    beam: Beam, medium: Medium, BC: BoundaryConditions,
    Lx: float, Ly: float,
    rho: float,
    C_f,                    # callable: Cp(T) [J/(kg·K)]
    k_f,                    # callable: k(T)  [W/(m·K)]
    t_end: float,           # total simulation time [s]
    T0: float = 298,
    SE = None,        # uniform volumetric source, W/m^3
    x_shift=None,
    y_shift=None,
    alpha = 0,
    beta = 0,
    dx: float = 1e-4, dy: float = 1e-4,
    dt: float = 1e-3,
    dt_ramp: float = None,   # optional dt ramp
    view: bool = False,
    view_freq: int = 20,    # update viewer every N steps
    dT_target = None,       # optional early stop on span
    dt_max=1,
    x_units = 'mm',
    y_units = 'mm'):

    # mesh
    nx_cells = int(nx.round(Lx / dx))
    ny_cells = int(nx.round(Ly / dy))
    mesh = Grid2D(dx=dx, dy=dy, nx=nx_cells, ny=ny_cells)
    if x_shift is None:
        x_shift = 0
    if y_shift is None:
        y_shift = -Ly / 2

    mesh += ((x_shift,), (y_shift,))  # shift mesh up
    cx = mesh.cellCenters[0].value
    cy = mesh.cellCenters[1].value
    CX = cx.reshape((nx_cells, ny_cells), order='F')  # shape (nx, ny)
    CY = cy.reshape((nx_cells, ny_cells), order='F')

    if SE is None:  # did not give stopping energy
        cj = CX[:, 0]
        E_0 = beam.E_0
        I_0 = beam.I_0

        E_beam = np.empty_like(cj)
        dIdx = np.empty_like(cj)
        E_inst = np.empty_like(cj)
        I_beam = np.empty_like(cj)
        dEb_dx = np.zeros_like(cj)  # eV/(m·s)
        dEdx = np.zeros_like(cj)
        dEdx_beam = np.zeros_like(cj)

        E_beam[0] = E_0 * I_0  # eV/s
        E_inst[0] = E_0  # eV
        I_beam[0] = I_0  # 1/s
        dIdx[0] = medium.get_dIdx(E_inst[0], I_beam[0])

        for l, j in enumerate(cj):
            if (l == nx_cells - 1):
                break
            dEdx[l] = medium.get_dEdx(E_inst[l])
            dEdx_beam[l] = dEdx[l] * I_beam[l] + E_inst[l] * dIdx[l]
            I_beam[l + 1] = I_beam[l] + dIdx[l] * dx
            E_beam[l + 1] = max(E_beam[l] + dEdx_beam[l] * dx, 0)
            E_inst[l + 1] = E_beam[l + 1] / I_beam[l + 1]
            dIdx[l + 1] = medium.get_dIdx(E_inst[l + 1], I_beam[l + 1])  # (1/s)/m

        for l in range(nx_cells):
            dEb_dx[l] -= I_beam[l] * dEdx[l]

        phi_free = np.array(beam.PD(CX, CY, 0, alpha, beta))
        dEb_dx *= 1.602176634e-19  # ev to J

        SE = dEb_dx[:, None] * phi_free * 1 / E_beam[0]
        SE = SE.reshape(-1, order='F')

    T0 = float(T0)
    SE = float(SE)
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
            viewer = Viewer(vars=(T,), title="Temperature Distribution",
                            datamin = 273, datamax = 510)
        except Exception:
            viewer = None

    # ---------- time loop ----------
    t_elapsed = 0.0
    step = 0
    T.updateOld()

    while t_elapsed < t_end:
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
        if (t_elapsed + dt > t_end):
            dt = t_end - t_elapsed

        # diagnostics
        if (step % max(1, view_freq)) == 0:
            Tamb = float(BC.T_amb)
            err = float(np.max(np.abs(T.value - Tamb)))
            print(f"t={t_elapsed:.3f}s  Tmax={T.value.max():.6f}  Tmin={T.value.min():.6f}  max|T-Tamb|={err:.6e}")

        if viewer is not None:
            Tmin = float(T.value.min())
            Tmax = float(T.value.max())
            print(f"t={t_elapsed + dt:0.3f}s  T[min,max]=[{Tmin:.2f}, {Tmax:.2f}]")
            ax = scale_fipy_viewer_axes(viewer,
                                   x_units=x_units, y_units=y_units,
                                   x_label="x", y_label="y",
                                   decimals=4)
            forceAspect(ax)
            viewer.plot()
        step += 1

def scale_slice_axes(ax,
                     plane="xy",                  # "xy", "xz", or "yz"
                     units=("mm", "mm"),          # (x_units, y_units)
                     decimals=3,                  # tick label precision
                     equal_aspect=True,
                     preserve_limits=True,
                     labelsize=13,                # axis label font size
                     ticklabelsize=11):           # tick label font size

    # ---- validate plane and map to labels ----
    plane = plane.lower()
    if plane not in ("xy", "xz", "yz"):
        raise ValueError("plane must be one of: 'xy', 'xz', 'yz'")

    x_label_char, y_label_char = {
        "xy": ("x", "y"),
        "xz": ("x", "z"),
        "yz": ("y", "z"),
    }[plane]

    # ---- unit multipliers ----
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

    # ---- keep current limits to avoid aspect/layout shrinking the panel ----
    if preserve_limits:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

    # ---- unit-scaled tick formatters ----
    def _mk_formatter(scale, sig):
        fmt = "{:." + str(sig) + "g}"
        return ticker.FuncFormatter(lambda val, pos: fmt.format(val * scale))

    ax.xaxis.set_major_formatter(_mk_formatter(fx, decimals))
    ax.yaxis.set_major_formatter(_mk_formatter(fy, decimals))

    # ---- labels + font sizes ----
    ax.set_xlabel(f"{x_label_char} [{ux}]", fontsize=labelsize)
    ax.set_ylabel(f"{y_label_char} [{uy}]", fontsize=labelsize)
    ax.tick_params(axis="both", labelsize=ticklabelsize)

    # ---- equal aspect in displayed units ----
    if equal_aspect:
        ax.set_aspect("equal", adjustable="box")

    # ---- redraw and restore limits ----
    fig = ax.figure
    if fig is not None and fig.canvas is not None:
        fig.canvas.draw_idle()

    if preserve_limits:
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

    return ax

def _bcs_from_T0_and_rad(mesh, T, T0_faces, rad_bnd, T_amb, eps):
    faces = [mesh.facesTop, mesh.facesBottom, mesh.facesLeft,
             mesh.facesRight, mesh.facesFront, mesh.facesBack]

    # T0_faces can be None, scalar, or iterable
    if T0_faces is None:
        T0_list = [None] * 6
    elif np.iterable(T0_faces):
        if len(T0_faces) != 6:
            raise ValueError("T0_faces must be length-6 if iterable (top, bottom, left, right, front, back).")
        T0_list = [None if v is None else float(v) for v in T0_faces]
    else:
        # broadcast a scalar to all faces
        T0_list = [float(T0_faces)] * 6

    # rad_bnd can be bool or iterable
    if np.iterable(rad_bnd):
        if len(rad_bnd) != 6:
            raise ValueError("rad_bnd must be length-6 if iterable (top, bottom, left, right, front, back).")
        rad_list = [bool(v) for v in rad_bnd]
    else:
        rad_list = [bool(rad_bnd)] * 6

    bcs = []

    # 1) Dirichlet faces
    for tv, f in zip(T0_list, faces):
        if tv is not None:
            bcs.append(FixedValue(faces=f, value=tv))

    # 2) Radiation faces
    needs_rad = any((rb and (tv is None)) for rb, tv in zip(rad_list, T0_list))
    if needs_rad:
        Tface = T.faceValue
        h = 4.0 * float(eps) * const.Stefan_Boltzmann * (nx.maximum(Tface, 0.0) ** 3.0)
        qn = h * (Tface - float(T_amb))  # W/m^2, heat leaving the domain

        # Per-face values: start at 0 everywhere, set qn only on selected faces
        val = FaceVariable(mesh=mesh, value=0.0)
        for rb, tv, f in zip(rad_list, T0_list, faces):
            if rb and (tv is None):
                val.setValue(qn, where=f)

        # Single FixedFlux applies per-face values (0 elsewhere)
        bcs.append(FixedFlux(faces=mesh.exteriorFaces, value=val))

    # 3) Remaining faces (T0=None & rad=False) → no BC (insulated by default)
    return bcs

def heateq_solid_3d_test(beam, medium, Lx, Ly, Lz, rho, C, k, t,
                    T0=298.0,
                    T0_faces: np.ndarray[float] | None = None,
                    rad_bnd: np.ndarray[bool] | bool = False,
                    T_amb=298, eps=1,
                    dx=1e-3, dy=1e-3, dz=1e-3,
                    dt=0.1, dt_ramp=None, dt_max=1, dT_target=None,
                    x_shift=None, y_shift=None, z_shift=None,
                    SE=None, alpha=0, beta=0,
                    view=False, view_freq=0):
    import scipy.constants as const

    if x_shift is None: x_shift = 0
    if y_shift is None: y_shift = -Ly / 2
    if z_shift is None: z_shift = -Lz / 2
    import numpy as np
    nx = int(np.ceil(Lx / dx))
    ny = int(np.ceil(Ly / dy))
    nz = int(np.ceil(Lz / dz))
    mesh = Grid3D(nx=nx, dx=dx, ny=ny, dy=dy, nz=nz, dz=dz)
    mesh += ((x_shift,), (y_shift,), (z_shift,))

    cx = mesh.cellCenters[0].value
    cy = mesh.cellCenters[1].value
    cz = mesh.cellCenters[2].value
    CX = cx.reshape((nx, ny, nz), order='F')
    CY = cy.reshape((nx, ny, nz), order='F')
    CZ = cz.reshape((nx, ny, nz), order='F')

    # --- SE like your 2D (ported to 3D) ---
    if SE is None:
        cj = CX[:, 0, 0]
        E_0 = beam.E_0
        I_0 = beam.I_0

        E_beam = np.empty_like(cj)
        dIdx = np.empty_like(cj)
        E_inst = np.empty_like(cj)
        I_beam = np.empty_like(cj)
        dEb_dx = np.zeros_like(cj)
        dEdx = np.zeros_like(cj)
        dEdx_beam = np.zeros_like(cj)

        E_beam[0] = E_0 * I_0
        E_inst[0] = E_0
        I_beam[0] = I_0
        dIdx[0] = medium.get_dIdx(E_inst[0], I_beam[0])

        for l, _ in enumerate(cj):
            if l == nx - 1: break
            dEdx[l] = medium.get_dEdx(E_inst[l])
            dEdx_beam[l] = dEdx[l] * I_beam[l] + E_inst[l] * dIdx[l]
            I_beam[l + 1] = I_beam[l] + dIdx[l] * dx
            E_beam[l + 1] = max(E_beam[l] + dEdx_beam[l] * dx, 0)
            E_inst[l + 1] = E_beam[l + 1] / I_beam[l + 1]
            dIdx[l + 1] = medium.get_dIdx(E_inst[l + 1], I_beam[l + 1])

        for l in range(nx):
            dEb_dx[l] -= I_beam[l] * dEdx[l]

        phi_free = np.array(beam.PD(CX, CY, CZ, alpha, beta))
        dEb_dx *= 1.602176634e-19  # eV→J

        SE = dEb_dx[:, None, None] * phi_free * 1 / E_beam[0]
        SE = SE.reshape(-1, order='F')

    SE = CellVariable(mesh=mesh, value=SE, name=r"$S_{E}$")
    T = CellVariable(mesh=mesh, value=float(T0), hasOld=True, name="T [K]")

    # k, rho*C can be functions of T (match your 2D pattern)
    if callable(k):
        K = CellVariable(mesh=mesh, value=k(T))
    else:
        K = CellVariable(mesh=mesh, value=k, rank=0)

    if callable(C):
        rho_C = CellVariable(mesh=mesh, value=rho * C(T))
    else:
        rho_C = CellVariable(mesh=mesh, value=rho * C, rank=0)

    # face masks
    m_top = mesh.facesTop
    m_bot = mesh.facesBottom
    m_left = mesh.facesLeft
    m_right = mesh.facesRight
    m_front = mesh.facesFront
    m_back = mesh.facesBack

    # normalize inputs like 2D intent
    # rad_bnd can be bool or array → broadcast to 6 faces [top, bottom, left, right, front(+z), back(-z)]
    rad_arr = np.array(rad_bnd, dtype=bool)
    if rad_arr.ndim == 0:
        rad_arr = np.full(6, rad_arr)

    # T0_faces can be None, scalar, or array length 6
    if T0_faces is None:
        T0_faces = (None, None, None, None, None, None)
    elif not np.iterable(T0_faces):
        T0_faces = (float(T0_faces),) * 6

    bcs = _bcs_from_T0_and_rad(mesh, T, T0_faces, rad_bnd, T_amb, eps)

    # PDE (fully implicit)
    eq = TransientTerm(coeff=rho_C, var=T) == DiffusionTerm(coeff=K, var=T) + SE

    if view:
        # --- three orthogonal mid-slices: xy, xz, yz ---
        ix, iy, iz = 0, ny // 2, nz // 2
        x0, x1 = x_shift, x_shift + Lx
        y0, y1 = y_shift, y_shift + Ly
        z0, z1 = z_shift, z_shift + Lz

        # use manual spacing so we can control gaps
        fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.5), constrained_layout= True)

        A = T.value.reshape((nx, ny, nz), order='F')

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

        # Now scale/label each panel correctly
        scale_slice_axes(axes[0], plane="xy", units=("mm", "mm"), decimals=3,
                         equal_aspect=True, preserve_limits=True, labelsize=14, ticklabelsize=12)
        scale_slice_axes(axes[1], plane="xz", units=("mm", "mm"), decimals=3,
                         equal_aspect=True, preserve_limits=True, labelsize=14, ticklabelsize=12)
        scale_slice_axes(axes[2], plane="yz", units=("mm", "mm"), decimals=3,
                         equal_aspect=True, preserve_limits=True, labelsize=14, ticklabelsize=12)

        # colorbar with larger label & ticks
        cbar = plt.colorbar(im_yz, ax=axes.ravel().tolist(), shrink=0.9)
        cbar.set_label('T [K]', fontsize=14)
        cbar.ax.tick_params(labelsize=12)

        plt.show(block=False)

    t_elapsed = 0.0
    t_next = 0.0
    T_maxes = []
    ts = []
    while t_elapsed < t:
        T.updateOld()

        if callable(k):
            K.setValue(k(T))
        if callable(C):
            rho_C.setValue(rho * C(T))
        bcs = _bcs_from_T0_and_rad(mesh, T, T0_faces, rad_bnd, T_amb, eps)
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

        if (dt_ramp is not None and dt < dt_max):
            dt *= dt_ramp
        if (dt > dt_max):
            dt = dt_max
        if (t_elapsed + dt > t):
            dt = t - t_elapsed

        if view and (t_elapsed > t_next):
            A = T.value.reshape((nx, ny, nz), order='F')

            im_xy.set_data(A[:, :, iz].T)
            im_xz.set_data(A[:, iy, :].T)
            im_yz.set_data(A[ix, :, :].T)

            vmin = A.min(); vmax = A.max()
            im_xy.set_clim(vmin, vmax)
            im_xz.set_clim(vmin, vmax)
            im_yz.set_clim(vmin, vmax)

            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.001)

            T_maxes.append(A.max())
            ts.append(t_elapsed)
            print("Time elapsed:", t_elapsed)
            t_next += view_freq
            A = T.value.reshape((nx, ny, nz), order='F')
            print(f"t={t_elapsed:.4e}s  T[min,max]=[{A.min():.2f}, {A.max():.2f}]  ΔT={A.max() - A.min():.2f}")
    # ----- end of time loop -----
    if view:
        # only touch pyplot if we actually created a figure
        plt.ioff()
        try:
            plt.show()
        except Exception as e:
            # Just in case the backend complains if the window is gone
            print(f"Warning: plt.show() failed at end of run: {e}")

    return T_maxes, ts