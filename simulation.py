from matplotlib import pyplot as plt

from Medium import atom, Medium
from Beam import Beam
import bisect
import numpy as np
import scipy.constants as const
from fipy import Variable, FaceVariable, CellVariable, Grid2D, TransientTerm, DiffusionTerm, DummySolver, Viewer, \
    ExplicitDiffusionTerm

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
                    eps = 1, dx = 1e-3, dy = 1e-3, dt_fact = 0.1,
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
    T = CellVariable(mesh=mesh, value=T0, hasOld=True, name="T [K]")

    if T0_top is None:
        T0_top = T0_faces
    if T0_bot is None:
        T0_bot = T0_faces
    if T0_left is None:
        T0_left = T0_faces
    if T0_right is None:
        T0_right = T0_faces

    K = CellVariable(mesh = mesh, value = k, rank = 0)
    rho_C = CellVariable(mesh = mesh, value = rho * C, rank = 0)

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
        eq = TransientTerm(coeff=rho_C, var=T) == ExplicitDiffusionTerm(coeff=K, var=T) + SE + \
             (mesh.exteriorFaces * flux_faces).divergence
        K.constrain(0, mesh.exteriorFaces)
    else:
        eq = TransientTerm(coeff = rho * C, var = T) == ExplicitDiffusionTerm(coeff=k, var=T) + SE
        T.constrain(T0_top, mesh.facesTop)
        T.constrain(T0_bot, mesh.facesBottom)
        T.constrain(T0_left, mesh.facesLeft)
        T.constrain(T0_right, mesh.facesRight)

    if view:
        viewer = Viewer(vars=T)

    alpha = k / (rho * C)

    t_start = 0.0
    t_end = t  # seconds
    dt = dt_fact * min(dx, dy) ** 2 / alpha

    ts = np.array([])
    Ts = np.array([])
    t_next = 0
    while t_start < t_end:
        ts = np.append(ts, t_start)
        Ts = np.append(Ts, T)
        T.updateOld()
        if(rad_bnd):
            T_face = T.faceValue.value  # shape = (nFaces,)
            q_all[m_top] = eps * sig * (T_face[m_top] - T_amb) ** 4
            q_all[m_bot] = eps * sig * (T_face[m_bot] - T_amb) ** 4
            q_all[m_left] = eps * sig * (T_face[m_left] - T_amb) ** 4
            q_all[m_right] = eps * sig * (T_face[m_right] - T_amb) ** 4
            flux_faces.setValue(-q_all)
            eq = TransientTerm(coeff=rho_C, var=T) == ExplicitDiffusionTerm(coeff=K, var=T) + SE +\
            (mesh.exteriorFaces * flux_faces).divergence
        eq.solve(var=T, dt=dt)
        t_start += dt
        if view:
            if t_start > t_next:
                print(f"t={t_start + dt:.3f}s  mean face temp ={T_face.mean():.4f} K, max={T_face.max():.4f} K)")
                t_next += view_freq
                viewer.plot()
    return ts, Ts