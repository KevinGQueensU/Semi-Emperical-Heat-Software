from matplotlib import pyplot as plt

from Medium import atom, Medium
from Beam import Beam
import bisect
import numpy as np
import scipy.constants as const
from mayavi import mlab
from fipy import FaceVariable, CellVariable, Grid2D, Grid3D, TransientTerm, DiffusionTerm, Viewer


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

def heateq_solid_2d(beam, medium, Lx, Ly, rho, C, k,
                    t, T0 = 298.0, T0_faces = 0, T0_top = None, T0_bot = None,
                    T0_left = None, T0_right = None, T_amb = 0, rad_bnd = False,
                    eps = 1, dx = 1e-3, dy = 1e-3, dt = 0.1, res = 1e-8,
                    x_shift = None, y_shift = None, SE = None,
                    alpha = 0, beta = 0, view=False, view_freq = 0):

    if x_shift is None:
        x_shift = 0
    if y_shift is None:
        y_shift = -Ly/2

    # creating CFD meshgrid
    nx = int(np.ceil(Lx/dx))
    ny = int(np.ceil(Ly/dy))
    mesh = Grid2D(nx=nx, dx=dx, ny=ny, dy=dy)
    mesh += ((x_shift,), (y_shift,)) # shift mesh up
    cx = mesh.cellCenters[0].value
    cy = mesh.cellCenters[1].value
    CX = cx.reshape((nx, ny), order='F')  # shape (nx, ny)
    CY = cy.reshape((nx, ny), order='F')

    if SE is None: # did not give stopping energy
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

        for k, j in enumerate(cj):
            if (k == nx - 1):
                break
            # Get energy gradient
            dEdx[k] = medium.get_dEdx(E_inst[k])
            dEdx_beam[k] = dEdx[k] * I_beam[k] + E_inst[k] * dIdx[k]
            I_beam[k + 1] = I_beam[k] + dIdx[k] * dx
            E_beam[k + 1] = max(E_beam[k] + dEdx_beam[k] * dx, 0)
            E_inst[k + 1] = E_beam[k + 1] / I_beam[k + 1]
            dIdx[k + 1] = medium.get_dIdx(E_inst[k + 1], I_beam[k + 1])  # (1/s)/m

        for k in range(nx):
            dEb_dx[k] -= I_beam[k] * dEdx[k]

        phi_free = np.array(beam.PD(CX, CY, 0, alpha, beta))
        dEb_dx *= 1.602176634e-19 # ev to J

        SE = dEb_dx[:, None] * phi_free * 1 / E_beam[0]
        SE = SE.reshape(-1, order='F')

    SE = CellVariable(mesh=mesh, value=SE, name=r"$S_{E}$")
    T = CellVariable(mesh=mesh, value=float(T0), hasOld=True, name="T [K]")

    if T0_top is None:
        T0_top = T0_faces
    if T0_bot is None:
        T0_bot = T0_faces
    if T0_left is None:
        T0_left = T0_faces
    if T0_right is None:
        T0_right = T0_faces

    K = CellVariable(mesh = mesh, value = float(k), rank = 0)
    rho_C = CellVariable(mesh = mesh, value = float(rho * C), rank = 0)

    if(rad_bnd):
        m_top = mesh.facesTop
        m_bot = mesh.facesBottom
        m_left = mesh.facesLeft
        m_right = mesh.facesRight
        top_cells = mesh.faceCellIDs[0][m_top]
        bottom_cells = mesh.faceCellIDs[0][m_bot]
        left_cells = mesh.faceCellIDs[0][m_left]
        right_cells = mesh.faceCellIDs[0][m_right]

        top_mask = np.zeros(mesh.numberOfCells, dtype=bool)
        bottom_mask = np.zeros(mesh.numberOfCells, dtype=bool)
        left_mask = np.zeros(mesh.numberOfCells, dtype=bool)
        right_mask = np.zeros(mesh.numberOfCells, dtype=bool)

        top_mask[top_cells] = True
        bottom_mask[bottom_cells] = True
        left_mask[left_cells] = True
        right_mask[right_cells] = True

        T.setValue(T0_top, where=top_mask)
        T.setValue(T0_bot, where=bottom_mask)
        T.setValue(T0_left, where=left_mask)
        T.setValue(T0_right, where=right_mask)


        sig = const.Stefan_Boltzmann
        T_face = T.faceValue.value
        q_all = np.zeros_like(T_face)

        q_all[m_top] = eps * sig * (T_face[m_top] - T_amb) ** 4
        q_all[m_bot] = eps * sig * (T_face[m_bot]- T_amb) ** 4
        q_all[m_left] = eps * sig * (T_face[m_left]- T_amb) ** 4
        q_all[m_right] = eps * sig * (T_face[m_right]- T_amb) ** 4

        flux_faces = FaceVariable(mesh=mesh, value=0.0)
        flux_faces.setValue(-q_all)
        eq = TransientTerm(coeff=rho_C, var=T) == DiffusionTerm(coeff=K, var=T) + SE + \
             (mesh.exteriorFaces * flux_faces).divergence
        K.constrain(0, mesh.exteriorFaces)
    else:
        eq = TransientTerm(coeff = rho * C, var = T) == DiffusionTerm(coeff=k, var=T) + SE
        T.constrain(T0_top, mesh.facesTop)
        T.constrain(T0_bot, mesh.facesBottom)
        T.constrain(T0_left, mesh.facesLeft)
        T.constrain(T0_right, mesh.facesRight)

    if view:
        viewer = Viewer(vars=T)
    t_elapsed = 0.0
    t_end = t  # seconds
    t_next = 0
    res_temp = 1e10
    while t_elapsed < t_end:
        T.updateOld()
        if (rad_bnd):
            T_face = T.faceValue.value  # shape = (nFaces,)
            q_all[m_top] = eps * sig * (T_face[m_top] - T_amb) ** 4
            q_all[m_bot] = eps * sig * (T_face[m_bot] - T_amb) ** 4
            q_all[m_left] = eps * sig * (T_face[m_left] - T_amb) ** 4
            q_all[m_right] = eps * sig * (T_face[m_right] - T_amb) ** 4
            flux_faces.setValue(-q_all)
            eq = TransientTerm(coeff=rho_C, var=T) == DiffusionTerm(coeff=K, var=T) + SE + \
                 (mesh.exteriorFaces * flux_faces).divergence
        res_arr = np.empty(0)
        while res_temp > res:
            res_temp = eq.sweep(var = T, dt = dt)
            res_arr = np.append(res_arr, res_temp)
            if(len(res_arr) > 1):
                if(res_arr[-1] == res_arr[-2]):
                    print("Time step too large for desired residual, reducing by 10%")
                    dt *= 0.9
        t_elapsed += dt
        if(dt < 1):
            dt *= 1.01
        if view:
            if t_elapsed > t_next:
                if(rad_bnd):
                    print(f"t={t_elapsed + dt:.3f}s  mean face temp ={T_face.mean():.4f} K, max={T_face.max():.4f} K)")
                t_next += view_freq
                viewer.plot()
                print("Time elapsed: " + str(t_elapsed))
    return

def heateq_solid_3d(beam, medium, Lx, Ly, Lz, rho, C, k,
                    t, T0=298.0, T0_faces=0, T0_top=None, T0_bot=None,
                    T0_left=None, T0_right=None, T0_front = None, T0_back = None, T_amb=0.0, rad_bnd=False,
                    eps = 1, dx = 1e-3, dy = 1e-3, dz = 1e-3, dt = 0.1, res = 1e-9,
                    x_shift = None, y_shift = None, z_shift = None, SE = None,
                    alpha = 0, beta = 0, view=False, view_freq = 0):
    if x_shift is None:
        x_shift = 0
    if y_shift is None:
        y_shift = -Ly/2
    if z_shift is None:
        z_shift = -Lz/2

    # creating CFD meshgrid
    nx = int(np.ceil(Lx/dx))
    ny = int(np.ceil(Ly/dy))
    nz = int(np.ceil(Lz/dz))
    mesh = Grid3D(nx=nx, dx=dx, ny=ny, dy=dy, nz = nz, dz = dz)
    mesh += ((x_shift,), (y_shift,), (z_shift,)) # shift mesh up
    cx = mesh.cellCenters[0].value
    cy = mesh.cellCenters[1].value
    cz = mesh.cellCenters[2].value
    CX = cx.reshape((nx, ny, nz), order='F')  # shape (nx, ny)
    CY = cy.reshape((nx, ny, nz), order='F')
    CZ = cz.reshape((nx, ny, nz), order='F')

    if SE is None: # did not give stopping energy
        cj = CX[:, 0, 0]
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

        for k, j in enumerate(cj):
            if (k == nx - 1):
                break
            # Get energy gradient
            dEdx[k] = medium.get_dEdx(E_inst[k])
            dEdx_beam[k] = dEdx[k] * I_beam[k] + E_inst[k] * dIdx[k]
            I_beam[k + 1] = I_beam[k] + dIdx[k] * dx
            E_beam[k + 1] = max(E_beam[k] + dEdx_beam[k] * dx, 0)
            E_inst[k + 1] = E_beam[k + 1] / I_beam[k + 1]
            dIdx[k + 1] = medium.get_dIdx(E_inst[k + 1], I_beam[k + 1])  # (1/s)/m

        for k in range(nx):
            dEb_dx[k] -= I_beam[k] * dEdx[k]

        phi_free = np.array(beam.PD(CX, CY, CZ, alpha, beta))
        dEb_dx *= 1.602176634e-19 # ev to J

        SE = dEb_dx[:, None, None] * phi_free * 1 / E_beam[0]
        SE = SE.reshape(-1, order='F')

    SE = CellVariable(mesh=mesh, value=SE, name=r"$S_{E}$")
    T = CellVariable(mesh=mesh, value=float(T0), hasOld=True, name="T [K]")

    # defaults for 4 named faces (we’ll set z-faces from T0_faces below)
    if T0_top is None:
        T0_top = T0_faces
    if T0_bot is None:
        T0_bot = T0_faces
    if T0_left is None:
        T0_left = T0_faces
    if T0_right is None:
        T0_right = T0_faces
    if T0_front is None:
        T0_front = T0_faces  # +z
    if T0_back is None:
        T0_back = T0_faces  # -z

    K = CellVariable(mesh=mesh, value=float(k), rank=0)
    rho_C = CellVariable(mesh=mesh, value=float(rho * C), rank=0)

    if rad_bnd:
        # --- build face masks (x,y,z) ---
        m_top = mesh.facesTop
        m_bot = mesh.facesBottom
        m_left = mesh.facesLeft
        m_right = mesh.facesRight
        m_front = mesh.facesFront
        m_back = mesh.facesBack

        # --- paint boundary-adjacent cells once (all 6 faces) ---
        # owner-cell IDs for faces are in row 0
        def paint(mask, value):
            cell_ids = mesh.faceCellIDs[0][mask]
            cell_mask = np.zeros(mesh.numberOfCells, dtype=bool)
            cell_mask[cell_ids] = True
            T.setValue(float(value), where=cell_mask)

        paint(m_top, T0_top)
        paint(m_bot, T0_bot)
        paint(m_left, T0_left)
        paint(m_right, T0_right)
        paint(m_front, T0_front)
        paint(m_back, T0_back)

        # zero diffusion “BC” on exterior faces so only the explicit flux applies
        K.constrain(0.0, mesh.exteriorFaces)

        # base equation; we’ll add radiation term each step
        baseEq = (TransientTerm(coeff=rho_C, var=T)
                  == DiffusionTerm(coeff=K, var=T) + SE)

        # preallocate containers used in loop
        sig = const.Stefan_Boltzmann
        T_face = T.faceValue.value
        q_all = np.zeros_like(T_face)
        flux_faces = FaceVariable(mesh=mesh, value=0.0)

    else:
        # Dirichlet faces (no radiation)
        T.constrain(float(T0_top), mesh.facesTop)
        T.constrain(float(T0_bot), mesh.facesBottom)
        T.constrain(float(T0_left), mesh.facesLeft)
        T.constrain(float(T0_right), mesh.facesRight)
        T.constrain(float(T0_faces), mesh.facesFront)
        T.constrain(float(T0_faces), mesh.facesBack)
        eq = (TransientTerm(coeff=rho * C, var=T)
              == DiffusionTerm(coeff=k, var=T) + SE)

    # optional viewer (3D viewing may require Mayavi)
    if view:
        viewer = Viewer(vars=T)

    t_elapsed = 0.0
    t_next = 0.0
    while t_elapsed < t:
        T.updateOld()

        if rad_bnd:
            # recompute radiation flux on ALL 6 exterior face sets
            T_face = T.faceValue.value
            q_all[:] = 0.0

            # correct ambient radiation: eps*sigma*(T^4 - T_amb^4)
            q_all[m_top] = eps * sig * (T_face[m_top] ** 4 - T_amb** 4)
            q_all[m_bot] = eps * sig * (T_face[m_bot] ** 4 - T_amb** 4)
            q_all[m_left] = eps * sig * (T_face[m_left] ** 4 - T_amb** 4)
            q_all[m_right] = eps * sig * (T_face[m_right] ** 4 - T_amb** 4)
            q_all[m_front] = eps * sig * (T_face[m_front] ** 4 - T_amb** 4)
            q_all[m_back] = eps * sig * (T_face[m_back] ** 4 - T_amb** 4)

            # FiPy divergence takes inward flux; outward radiation is negative
            flux_faces.setValue(-q_all)

            eq = baseEq + (mesh.exteriorFaces * flux_faces).divergence

        # simple residual-controlled sweep
        res_temp = 1e9
        while res_temp > res:
            res_temp = eq.sweep(var=T, dt=dt)
            # basic guard: if it stalls, reduce dt a bit
            # (optional; keep if you’ve been using this pattern)
            # if res_temp_old == res_temp: dt *= 0.9
            # res_temp_old = res_temp

        t_elapsed += dt
        if view and t_elapsed >= t_next:
            viewer.plot()
            t_next += max(view_freq, 0.0)

    return
