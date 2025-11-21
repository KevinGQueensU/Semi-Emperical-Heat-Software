from Medium import atom, Medium
from Beam import Beam
import bisect
import scipy.constants as const
from fipy import FaceVariable, CellVariable, Grid2D, Grid3D, TransientTerm, DiffusionTerm, Viewer, \
    PowerLawConvectionTerm, ImplicitSourceTerm, FixedFlux
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (needed for 3D)
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


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

import matplotlib.ticker as ticker
def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect);

from fipy.boundaryConditions import FixedValue, FixedFlux
from fipy.tools import numerix as nx

_SIGMA_SB = 5.670374419e-8  # Stefan–Boltzmann constant [W/(m^2·K^4)]

def heateq_solid_2d(
    beam, medium,
    Lx: float, Ly: float,
    rho: float,
    C_f,                    # callable: Cp(T) [J/(kg·K)]
    k_f,                    # callable: k(T)  [W/(m·K)]
    t_end: float,           # total simulation time [s]
    SE: float = 0.0,        # uniform volumetric source, W/m^3
    T0: float = 298.0,
    T0_faces = (None, None, None, None),      # (left, right, bottom, top) fixed values or None
    rad_bnd = (False, False, False, False),   # (left, right, bottom, top) radiation on/off
    T_amb: float = 298.0,
    eps: float = 1.0,       # emissivity
    dx: float = 1e-4, dy: float = 1e-4,
    dt: float = 1e-3,
    dt_ramp: float = 0.0,   # optional dt ramp
    view: bool = False,
    view_freq: int = 20,    # update viewer every N steps
    dT_target = None,       # optional early stop on span
    dt_max=1):
    # ---------- mesh ----------
    nx_cells = int(nx.round(Lx / dx))
    ny_cells = int(nx.round(Ly / dy))
    mesh = Grid2D(dx=dx, dy=dy, nx=nx_cells, ny=ny_cells)

    # ---------- temperature variable (ensure float) ----------
    T0f = float(T0)
    T = CellVariable(mesh=mesh, value=T0f, name="Temperature", hasOld=True)

    # ---------- material properties as Variables (depend on T) ----------
    rhoC = CellVariable(mesh=mesh, name="rho*C", value=rho * float(C_f(T0f)))
    k_cell = CellVariable(mesh=mesh, name="k(T)", value=float(k_f(T0f)))

    def _refresh_material_props():
        Cp_val = C_f(T)   # J/(kg·K)
        k_val  = k_f(T)   # W/(m·K)
        rhoC.setValue(rho * Cp_val)
        k_cell.setValue(k_val)

    _refresh_material_props()

    # ---------- faces and masks ----------
    facesL = mesh.facesLeft
    facesR = mesh.facesRight
    facesB = mesh.facesBottom
    facesT = mesh.facesTop
    side_faces = (facesL, facesR, facesB, facesT)

    # Build a face-areas array compatible with UniformGrid2D:
    # vertical faces (normal x≠0) have area = dy; horizontal faces (normal y≠0) have area = dx
    fn = mesh.faceNormals
    vertical = nx.abs(fn[0]) > 0.5   # left/right
    horiz    = nx.abs(fn[1]) > 0.5   # bottom/top
    face_areas = vertical * dy + horiz * dx   # shape (nFaces,)

    # ---------- fixed-value BCs (precedence over radiation on that side) ----------
    fixed_bcs = []
    for tv, faces in zip(T0_faces, side_faces):
        if tv is not None:
            fixed_bcs.append(FixedValue(faces=faces, value=float(tv)))

    # ---------- radiation (Robin) via FixedFlux: -k ∂T/∂n = h (T - T_amb), h = 4 ε σ T^3 ----------
    def _build_radiation_bcs():
        if not any(rad_bnd):
            return []
        Tface = T.faceValue
        h_face = 4.0 * float(eps) * _SIGMA_SB * (Tface ** 3.0)
        qn = h_face * (Tface - float(T_amb))  # W/m^2 (heat leaving domain)
        bcs = []
        for i, (on, faces) in enumerate(zip(rad_bnd, side_faces)):
            if on and (T0_faces[i] is None):
                bcs.append(FixedFlux(faces=faces, value=qn))
        return bcs

    # ---------- equation (fully implicit) ----------
    source_term = float(SE) if SE else 0.0
    eq = TransientTerm(coeff=rhoC) == DiffusionTerm(coeff=k_cell) + source_term

    # ---------- viewer ----------
    viewer = None
    if view:
        try:
            viewer = Viewer(vars=(T,), title="Temperature (K)")
            viewer.plot()
        except Exception:
            viewer = None

    # ---------- diagnostics ----------
    ext_mask = mesh.exteriorFaces.value  # boolean array

    def _net_radiative_power_into_domain():
        """Return net power INTO domain (negative of radiated-out power) on active radiation faces."""
        if not any(rad_bnd):
            return 0.0
        Tface = T.faceValue
        # Use .value to get NumPy arrays for purely algebraic diagnostics
        Tfv = Tface.value
        hfv = 4.0 * float(eps) * _SIGMA_SB * (nx.maximum(Tface, 0.0) ** 3.0).value
        q_out = hfv * (Tfv - float(T_amb))  # W/m^2 (NumPy)

        # build active mask (NumPy bool array over faces)
        active = (mesh.exteriorFaces & False).value
        if rad_bnd[0] and (T0_faces[0] is None): active |= facesL.value
        if rad_bnd[1] and (T0_faces[1] is None): active |= facesR.value
        if rad_bnd[2] and (T0_faces[2] is None): active |= facesB.value
        if rad_bnd[3] and (T0_faces[3] is None): active |= facesT.value
        if not active.any():
            return 0.0

        P_out = float((q_out[active] * face_areas[active]).sum())  # W
        return -P_out

    def _energy_balance(dt_used):
        dT = (T - T.old)
        dV = mesh.cellVolumes
        dU = float(nx.sum((rhoC * dT) * dV))
        Qsrc = float(source_term) * float(nx.sum(dV)) * float(dt_used) if source_term else 0.0
        Qrad = _net_radiative_power_into_domain() * float(dt_used)
        resid = abs(dU - (Qsrc + Qrad))
        print(f"[EB] dU={dU: .3e} J, Qsrc={Qsrc: .3e} J, Qrad={Qrad: .3e} J, residual={resid: .3e} J")

    # ---------- time loop ----------
    t = 0.0
    step = 0
    T.updateOld()
    print(f"Net radiative power into domain [W]: {_net_radiative_power_into_domain()}")

    while t < t_end:
        if (dt_ramp is not None and dt < dt_max):
            dt *= dt_ramp
        if (dt > dt_max):
            dt = dt_max
        if (t + dt > t_end):
            dt = t_end - t
        # update props and BCs for implicit sweep
        _refresh_material_props()
        bcs = fixed_bcs + _build_radiation_bcs()

        # advance one implicit step
        T.updateOld()
        eq.solve(var=T, dt=dt, boundaryConditions=bcs)

        # diagnostics
        if (step % max(1, view_freq)) == 0:
            print(f"Net radiative power into domain [W]: {_net_radiative_power_into_domain()}")
            Tmin = float(T.value.min()); Tmax = float(T.value.max())
            print(f"t={t + dt:0.3f}s  T[min,max]=[{Tmin:.2f}, {Tmax:.2f}]")
            if dT_target is not None and (Tmax - Tmin) >= float(dT_target):
                break

        if viewer is not None:
            viewer.plot()

        t += dt
        print(t)
        step += 1







def heateq_solid_3d(beam, medium, Lx, Ly, Lz, rho, C, k, t,
                    T0 = 298.0,
                    T0_faces: np.ndarray[float] | None = None,
                    rad_bnd: np.ndarray[bool] | bool = False,
                    T_amb = 298, eps = 1,
                    dx = 1e-3, dy = 1e-3, dz = 1e-3,
                    dt = 0.1, dt_ramp = None, dt_max = 1, dT_target = None,
                    x_shift = None, y_shift = None, z_shift = None,
                    SE = None, alpha = 0, beta = 0,
                    view=False, view_freq = 0):


    import scipy.constants as const

    if x_shift is None: x_shift = 0
    if y_shift is None: y_shift = -Ly/2
    if z_shift is None: z_shift = -Lz/2
    import numpy as np
    nx = int(np.ceil(Lx/dx))
    ny = int(np.ceil(Ly/dy))
    nz = int(np.ceil(Lz/dz))
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
        dIdx   = np.empty_like(cj)
        E_inst = np.empty_like(cj)
        I_beam = np.empty_like(cj)
        dEb_dx = np.zeros_like(cj)
        dEdx   = np.zeros_like(cj)
        dEdx_beam = np.zeros_like(cj)

        E_beam[0] = E_0 * I_0
        E_inst[0] = E_0
        I_beam[0] = I_0
        dIdx[0]   = medium.get_dIdx(E_inst[0], I_beam[0])

        for l, _ in enumerate(cj):
            if l == nx - 1: break
            dEdx[l]      = medium.get_dEdx(E_inst[l])
            dEdx_beam[l] = dEdx[l] * I_beam[l] + E_inst[l] * dIdx[l]
            I_beam[l+1]  = I_beam[l] + dIdx[l] * dx
            E_beam[l+1]  = max(E_beam[l] + dEdx_beam[l] * dx, 0)
            E_inst[l+1]  = E_beam[l+1] / I_beam[l+1]
            dIdx[l+1]    = medium.get_dIdx(E_inst[l+1], I_beam[l+1])

        for l in range(nx):
            dEb_dx[l] -= I_beam[l] * dEdx[l]

        phi_free = np.array(beam.PD(CX, CY, CZ, alpha, beta))
        dEb_dx *= 1.602176634e-19 # eV→J

        SE = dEb_dx[:, None, None] * phi_free * 1 / E_beam[0]
        SE = SE.reshape(-1, order='F')

    SE = CellVariable(mesh=mesh, value=SE, name=r"$S_{E}$")
    T  = CellVariable(mesh=mesh, value=float(T0), hasOld=True, name="T [K]")

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
    m_top   = mesh.facesTop
    m_bot   = mesh.facesBottom
    m_left  = mesh.facesLeft
    m_right = mesh.facesRight
    m_front = mesh.facesFront
    m_back  = mesh.facesBack

    # normalize inputs like your 2D intent
    # rad_bnd can be bool or array → broadcast to 6 faces [top, bottom, left, right, front(+z), back(-z)]
    rad_arr = np.array(rad_bnd, dtype=bool)
    if rad_arr.ndim == 0:
        rad_arr = np.full(6, rad_arr)
    rad_int = rad_arr.astype(int)

    # T0_faces can be None, scalar, or array length 6
    if T0_faces is None:
        T0_faces = (T0, T0, T0, T0, T0, T0)
    elif not np.iterable(T0_faces):
        T0_faces = (float(T0_faces),) * 6
    else:
        T0_faces = tuple(map(float, T0_faces))

    sig = const.Stefan_Boltzmann

    # --- Branches like 2D ---
    if np.any(rad_arr):

        # paint boundary-adjacent cells once for all 6 faces (like your 2D masks)
        def paint(face_mask, value):
            cell_ids = mesh.faceCellIDs[0][face_mask]
            cell_mask = np.zeros(mesh.numberOfCells, dtype=bool)
            cell_mask[cell_ids] = True
            T.setValue(value, where=cell_mask)

        face_order = [m_top, m_bot, m_left, m_right, m_front, m_back]
        for fm, v in zip(face_order, T0_faces):
            paint(fm, v)

        # disable diffusion through exterior faces (same idea as 2D)
        K.constrain(0, mesh.exteriorFaces)

        # initial flux assembly
        T_face = T.faceValue.value
        q_all = np.zeros_like(T_face)
        q_all[m_top]   = eps * sig * (T_face[m_top]  ** 4 - T_amb ** 4) * rad_int[0]
        q_all[m_bot]   = eps * sig * (T_face[m_bot]  ** 4 - T_amb ** 4) * rad_int[1]
        q_all[m_left]  = eps * sig * (T_face[m_left] ** 4 - T_amb ** 4) * rad_int[2]
        q_all[m_right] = eps * sig * (T_face[m_right]** 4 - T_amb ** 4) * rad_int[3]
        q_all[m_front] = eps * sig * (T_face[m_front]** 4 - T_amb ** 4) * rad_int[4]
        q_all[m_back]  = eps * sig * (T_face[m_back] ** 4 - T_amb ** 4) * rad_int[5]

        flux_faces = FaceVariable(mesh=mesh, value=0.0)
        flux_faces.setValue(-q_all)

        eq = TransientTerm(coeff=rho_C, var=T) == DiffusionTerm(coeff=K, var=T) + SE + \
             (mesh.exteriorFaces * flux_faces).divergence

        # non-radiating faces get Dirichlet constraints (like your 2D logic)
        for (fm, is_rad, v) in zip(face_order, rad_arr, T0_faces):
            if not is_rad:
                T.constrain(v, where=fm)

    elif T0_faces is not None:
        # Dirichlet on all 6 faces
        eq = TransientTerm(coeff=rho_C, var=T) == DiffusionTerm(coeff=K, var=T) + SE
        for (fm, v) in zip([m_top, m_bot, m_left, m_right, m_front, m_back], T0_faces):
            T.constrain(v, where=fm)
    else:
        eq = TransientTerm(coeff=rho_C, var=T) == DiffusionTerm(coeff=K, var=T) + SE

    if view:
        # --- three orthogonal mid-slices: xy, xz, yz ---
        ix, iy, iz = nx // 2, ny // 2, nz // 2
        x0, x1 = x_shift, x_shift + Lx
        y0, y1 = y_shift, y_shift + Ly
        z0, z1 = z_shift, z_shift + Lz

        fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
        A = T.value.reshape((nx, ny, nz), order='F')

        im_xy = axes[0].imshow(A[:, :, iz].T, origin='lower', extent=[x0, x1, y0, y1], aspect='equal', cmap='turbo')
        axes[0].set_title('xy @ mid z');
        axes[0].set_xlabel('x [m]');
        axes[0].set_ylabel('y [m]')

        im_xz = axes[1].imshow(A[:, iy, :].T, origin='lower', extent=[x0, x1, z0, z1], aspect='equal', cmap='turbo')
        axes[1].set_title('xz @ mid y');
        axes[1].set_xlabel('x [m]');
        axes[1].set_ylabel('z [m]')

        im_yz = axes[2].imshow(A[ix, :, :].T, origin='lower', extent=[y0, y1, z0, z1], aspect='equal', cmap='turbo')
        axes[2].set_title('yz @ mid x');
        axes[2].set_xlabel('y [m]');
        axes[2].set_ylabel('z [m]')

        plt.colorbar(im_yz, ax=axes.ravel().tolist(), shrink=0.9, label='T [K]', cmap='turbo')
        plt.show(block=False)

    t_elapsed = 0.0
    t_next = 0.0

    while t_elapsed < t:
        T.updateOld()

        if callable(k):
            K.setValue(k(T))

        if callable(C):
            rho_C.setValue(rho * C(T))

        if np.any(rad_arr):
            # rebuild radiation flux on all 6 faces each step

            q_all[m_top]   = eps * sig * (T_face[m_top]  ** 4 - T_amb ** 4) * rad_int[0]
            q_all[m_bot]   = eps * sig * (T_face[m_bot]  ** 4 - T_amb ** 4) * rad_int[1]
            q_all[m_left]  = eps * sig * (T_face[m_left] ** 4 - T_amb ** 4) * rad_int[2]
            q_all[m_right] = eps * sig * (T_face[m_right]** 4 - T_amb ** 4) * rad_int[3]
            q_all[m_front] = eps * sig * (T_face[m_front]** 4 - T_amb ** 4) * rad_int[4]
            q_all[m_back]  = eps * sig * (T_face[m_back] ** 4 - T_amb ** 4) * rad_int[5]
            flux_faces.setValue(-q_all)

            eq = TransientTerm(coeff=rho_C, var=T) == DiffusionTerm(coeff=K, var=T) + SE + \
                 (mesh.exteriorFaces * flux_faces).divergence

        T_old = T.value.copy()

        if dT_target is not None:
            while True:
                eq.solve(var=T, dt=dt)
                dT_inf = np.max(np.abs(T.value - T_old))
                if dT_inf > dT_target:
                    print("Time step too large for required dT, reducing by 50%")
                    dt *= 0.5
                else:
                    break
        else:
            eq.solve(var=T, dt=dt)

        t_elapsed += dt

        if (dt_ramp is not None and dt < dt_max):
            dt *= dt_ramp
        if (dt > dt_max):
            dt = dt_max
        if(t_elapsed + dt > t):
            dt = t - t_elapsed

        if view and (t_elapsed > t_next):
            A = T.value.reshape((nx, ny, nz), order='F')

            im_xy.set_data(A[:, :, iz].T)
            im_xz.set_data(A[:, iy, :].T)
            im_yz.set_data(A[ix, :, :].T)

            # keep all panels on the same dynamic scale (remove these two lines for fixed scale)
            vmin = A.min();
            vmax = A.max()
            im_xy.set_clim(vmin, vmax)
            im_xz.set_clim(vmin, vmax)
            im_yz.set_clim(vmin, vmax)

            # force an actual refresh in SciView / non-GUI contexts
            fig.canvas.draw()  # draw the canvas now
            fig.canvas.flush_events()  # process GUI events if any
            plt.pause(0.001)  # yield control briefly

            print("Time elapsed:", t_elapsed)
            t_next += view_freq
            A = T.value.reshape((nx, ny, nz), order='F')
            print(f"t={t_elapsed:.4e}s  T[min,max]=[{A.min():.2f}, {A.max():.2f}]  ΔT={A.max() - A.min():.2f}")
    plt.ioff()
    plt.show()
    return


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

    # normalize inputs like your 2D intent
    # rad_bnd can be bool or array → broadcast to 6 faces [top, bottom, left, right, front(+z), back(-z)]
    rad_arr = np.array(rad_bnd, dtype=bool)
    if rad_arr.ndim == 0:
        rad_arr = np.full(6, rad_arr)

    # T0_faces can be None, scalar, or array length 6
    if T0_faces is None:
        T0_faces = (T0, T0, T0, T0, T0, T0)
    elif not np.iterable(T0_faces):
        T0_faces = (float(T0_faces),) * 6

    # --- Branches like 2D (Robin radiation per FiPy docs) ---
    if np.any(rad_arr):
        # --- Robin radiation on selected faces, no ImplicitSourceTerm ---

        from fipy.tools import numerix
        MA = numerix.MA

        # 1) Build a FiPy face mask (binOp), not a numpy array
        face_order = [m_top, m_bot, m_left, m_right, m_front, m_back]
        mask_faces = face_order[0] & False
        for fm, is_rad in zip(face_order, rad_arr):
            if is_rad:
                mask_faces = mask_faces | fm

        if T0_faces is not None:
            face_order = [m_top, m_bot, m_left, m_right, m_front, m_back]
            for fm, is_rad, val in zip(face_order, rad_arr, T0_faces):
                if not is_rad:
                    T.constrain(val, where=fm)

        # 2) Geometry helpers for the FiPy recipe (numeric snapshots)
        fc = mesh.faceCenters  # (dim, nFaces)
        cc = mesh.cellCenters  # (dim, nCells)
        c2f = numerix.take(cc, mesh.faceCellIDs, axis=1)  # (dim, 2, nFaces)
        fc_rep = MA.repeat(fc[..., numerix.NewAxis, :], 2, axis=1)
        cellToFaceDistanceVectors = fc_rep - c2f  # (dim, 2, nFaces)
        dcc = c2f[..., 1, :] - c2f[..., 0, :]  # (dim, nFaces)
        owner_to_face = cellToFaceDistanceVectors[:, 0, :]  # (dim, nFaces)
        cellDistanceVectors = MA.filled(MA.where(MA.getmaskarray(dcc), owner_to_face, dcc))
        dPf = FaceVariable(mesh=mesh, value=mesh._faceToCellDistanceRatio * cellDistanceVectors)  # vector

        # 3) Numeric snapshots for material & normals
        Kf = K.faceValue.value
        n_vec = getattr(mesh.faceNormals, "value", mesh.faceNormals)  # (dim, nFaces)

        # 4) Linearize radiation at current T_face (numeric)
        T_face_prev = T.faceValue.value
        hr_all = eps * const.Stefan_Boltzmann * (T_face_prev + T_amb) * (T_face_prev ** 2 + T_amb ** 2)

        # 5) Robin scalar factor on faces:  C = K / (dPf·(h n) + K)
        a_vec_tmp = n_vec * hr_all  # (dim, nFaces) numeric
        a_tmp = FaceVariable(mesh=mesh, rank=1, value=0.0)
        a_tmp.setValue(a_vec_tmp, where=mask_faces)  # a = h n only on masked faces

        denom = (dPf.dot(a_tmp)).value + Kf  # scalar (nFaces,)
        C_val = Kf / denom  # scalar (nFaces,)

        # 6) Build convection vector q = C * a = C * (h n) on masked faces (0 elsewhere)
        q = FaceVariable(mesh=mesh, rank=1, value=0.0)
        q.setValue((a_vec_tmp * C_val), where=mask_faces)  # vector (dim, nFaces) on mask

        # 7) Build source flux s = C * g = C * (h T_amb) on masked faces
        s = FaceVariable(mesh=mesh, value=0.0)  # scalar on faces
        s.setValue(C_val * (hr_all * T_amb), where=mask_faces)

        # 8) Final equation (no ImplicitSourceTerm, no mixing):
        eq = (TransientTerm(coeff=rho_C, var=T)
              == DiffusionTerm(coeff=K, var=T) + SE
              + PowerLawConvectionTerm(coeff=q, var=T)  # handles a·∇T part
              + (s).divergence)  # handles g part

    elif T0_faces is not None:
        # Dirichlet on all 6 faces, no radiation
        eq = TransientTerm(coeff=rho_C, var=T) == DiffusionTerm(coeff=K, var=T) + SE
        for (fm, v) in zip([m_top, m_bot, m_left, m_right, m_front, m_back], T0_faces):
            T.constrain(v, where=fm)
    else:
        eq = TransientTerm(coeff=rho_C, var=T) == DiffusionTerm(coeff=K, var=T) + SE

    if view:
        # --- three orthogonal mid-slices: xy, xz, yz ---
        ix, iy, iz = 0, ny // 2, nz // 2
        x0, x1 = x_shift, x_shift + Lx
        y0, y1 = y_shift, y_shift + Ly
        z0, z1 = z_shift, z_shift + Lz

        fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
        A = T.value.reshape((nx, ny, nz), order='F')

        im_xy = axes[0].imshow(A[:, :, iz].T, origin='lower', extent=[x0, x1, y0, y1], aspect='equal', cmap='turbo')
        axes[0].set_title('XY plane at Lz/2');
        axes[0].set_xlabel('x [m]');
        axes[0].set_ylabel('y [m]')

        im_xz = axes[1].imshow(A[:, iy, :].T, origin='lower', extent=[x0, x1, z0, z1], aspect='equal', cmap='turbo')
        axes[1].set_title('XZ plane at Ly/2');
        axes[1].set_xlabel('x [m]');
        axes[1].set_ylabel('z [m]')

        im_yz = axes[2].imshow(A[ix, :, :].T, origin='lower', extent=[y0, y1, z0, z1], aspect='equal', cmap='turbo')
        axes[2].set_title('YZ plane at x = 0');
        axes[2].set_xlabel('y [m]');
        axes[2].set_ylabel('z [m]')

        plt.colorbar(im_yz, ax=axes.ravel().tolist(), shrink=0.9, label='T [K]', cmap='turbo')
        plt.show(block=False)

    t_elapsed = 0.0
    t_next = 0.0

    while t_elapsed < t:
        T.updateOld()

        if callable(k):
            K.setValue(k(T))

        if callable(C):
            rho_C.setValue(rho * C(T))

        if np.any(rad_arr):
            # refresh T_face snapshot
            T_face_prev = T.faceValue.value
            hr_all = eps * const.Stefan_Boltzmann * (T_face_prev + T_amb) * (T_face_prev ** 2 + T_amb ** 2)

            # refresh a, C, q, s (same formulas as above, concise):
            a_vec_tmp = n_vec * hr_all
            a_tmp.setValue(0.0)
            a_tmp.setValue(a_vec_tmp, where=mask_faces)
            denom = (dPf.dot(a_tmp)).value + Kf
            C_val = Kf / denom
            q.setValue(0.0);
            q.setValue((a_vec_tmp * C_val), where=mask_faces)
            s.setValue(0.0);
            s.setValue(C_val * (hr_all * T_amb), where=mask_faces)

            # reassemble eq if you prefer (or just keep the same object; coeffs updated in-place)
            eq = (TransientTerm(coeff=rho_C, var=T)
                  == DiffusionTerm(coeff=K, var=T) + SE
                  + PowerLawConvectionTerm(coeff=q, var=T)
                  + (s).divergence)

        T_old = T.value.copy()

        if dT_target is not None:
            while True:
                eq.solve(var=T, dt=dt)
                dT_inf = np.max(np.abs(T.value - T_old))
                if dT_inf > dT_target:
                    print("Time step too large for required dT, reducing by 50%")
                    dt *= 0.5
                else:
                    break
        else:
            eq.solve(var=T, dt=dt)

        # if using radiation, also check SE and face flux sign

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

            # keep all panels on the same dynamic scale (remove these two lines for fixed scale)
            vmin = A.min();
            vmax = A.max()
            im_xy.set_clim(vmin, vmax)
            im_xz.set_clim(vmin, vmax)
            im_yz.set_clim(vmin, vmax)

            # force an actual refresh in SciView / non-GUI contexts
            fig.canvas.draw()  # draw the canvas now
            fig.canvas.flush_events()  # process GUI events if any
            plt.pause(0.001)  # yield control briefly

            print("Time elapsed:", t_elapsed)
            t_next += view_freq
            A = T.value.reshape((nx, ny, nz), order='F')
            print(f"t={t_elapsed:.4e}s  T[min,max]=[{A.min():.2f}, {A.max():.2f}]  ΔT={A.max() - A.min():.2f}")
    plt.ioff()
    plt.show()
    return
