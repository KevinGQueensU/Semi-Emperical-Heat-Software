from matplotlib import pyplot as plt

from Medium import atom, Medium
from Beam import Beam
import bisect
import numpy as np
from fipy import Variable, FaceVariable, CellVariable, Grid2D, TransientTerm, DiffusionTerm, DummySolver, Viewer, \
    ExplicitDiffusionTerm
from fipy.tools import numerix

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
                    t, T0 = 298, dx = 1e-3, dy = 1e-3, dt_fact = 0.1,
                    x_shift = None, y_shift = None,
                    SE = None, alpha = 0, beta = 0, view=False):
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
    T.constrain(T0, mesh.exteriorFaces)
    eq = TransientTerm(coeff = rho * C, var = T) == ExplicitDiffusionTerm(coeff=k, var=T) + SE
    if view:
        viewer = Viewer(vars=T)

    alpha = k / (rho * C)

    t_start = 0.0
    t_end = t  # seconds
    dt = dt_fact * min(dx, dy) ** 2 / alpha

    ts = np.array([])
    Ts = np.array([])

    while t_start < t_end:
        ts = np.append(ts, t_start)
        Ts = np.append(Ts, T)
        T.updateOld()
        eq.solve(var=T, dt=dt)
        t_start += dt
        print(t_start)
        if view:
            viewer.plot()
    return ts, Ts

def heateq_solid_2d_temp(rho_Ni, C, k):
    Lx = 350e-3

    # Parameters for the beam and values
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
    Ni = atom('Ni', Z_Ni, A_Ni, 1, "Ni//Cross_Sections//Ni_px.txt")
    rho = 8.908  # g/cm^3
    n = 2.56e30 # 1/m^3
    H = 140e-3
    W = 10e-3  # 10 mm square slab
    A_xsec = W * H
    P_perim = 2 * (W + H)
    medium = Medium(n, rho, Ni, Lx, A_xsec, P_perim, "Ni//Stopping_Power//H.txt", beam,
                    x0 = x0)
    Lx = 140e-3
    dx = 1e-3
    nx = int(np.ceil(Lx/dx))

    Ly = 140e-3
    dy = 1e-3
    ny = int(np.ceil(Ly/dy))

    mesh = Grid2D(nx=nx, dx=dx, ny=ny, dy=dy)
    mesh = mesh + ((0,), (-Ly/2,))
    cx = mesh.cellCenters[0].value
    cy = mesh.cellCenters[1].value
    CX = cx.reshape((nx, ny), order='F')  # shape (nx, ny)
    CY = cy.reshape((nx, ny), order='F')
    alpha = beta = 0.0
    z = 0
    cx_test = cx
    cx = CX[:, 0]
    E_beam = np.empty_like(cx)
    E_beam[0] = E_0*I_0  # eV/s
    dIdx = np.empty_like(cx)
    E_inst = np.empty_like(cx)
    I_beam = np.empty_like(cx)
    dEb_dx = np.zeros_like(cx)  # eV/(m·s)
    dEdx = np.zeros_like(cx)
    dEdx_beam = np.zeros_like(cx)
     # Free energy flux eV/(m^2·s)

    E_inst[0] = E_0 # eV
    I_beam[0] = I_0  # 1/s
    dIdx[0] = medium.get_dIdx(E_inst[0], I_beam[0])

    for k, j in enumerate(cx):
        if(k == nx - 1):
            break
        # Get energy gradient
        dEdx[k] = medium.get_dEdx(E_inst[k])
        dEdx_beam[k] = dEdx[k] * I_beam[k] + E_inst[k] * dIdx[k]
        I_beam[k + 1] = I_beam[k] + dIdx[k] * dx
        E_beam[k + 1] = max(E_beam[k] + dEdx_beam[k] * dx, 0)
        E_inst[k + 1] = E_beam[k+1]/I_beam[k+1]
        dIdx[k+1] = medium.get_dIdx(E_inst[k+1], I_beam[k+1])  # (1/s)/m

    for k in range(nx):
        dEb_dx[k] -= I_beam[k] * dEdx[k]
    print("initial finished")
    eV_to_J = 1.602176634e-19 # shape (nx, ny)
    phi_free = np.array(beam.PD(CX, CY, z, alpha, beta))
    dEb_dx *= eV_to_J
    SE_W_per_m2 = dEb_dx[:, None] * phi_free * 1/E_beam[0]
    SE_W_per_m2 = SE_W_per_m2.reshape(-1, order='F')
    SE = CellVariable(mesh=mesh, value=    SE_W_per_m2, name=r"$S_{E}$")

    # Initial conditions
    T0 = 298 # K
    T = CellVariable(mesh=mesh, value=T0, hasOld=True, name="T [K]")
    T.constrain(T0, mesh.exteriorFaces) # constrain the surfaces of box
    eq = TransientTerm(coeff = rho_Ni * C, var = T) == ExplicitDiffusionTerm(coeff=k, var=T) + SE * Lx * Ly
    viewer = Viewer(vars=T)

    alpha = k / (rho_Ni * C)
    dt = 0.1 * min(dx, dy) ** 2 / alpha
    t_end = 300  # seconds
    t = 0.0

    while t < t_end:
        T.updateOld()
        eq.solve(var=T, dt=dt)
        t += dt
        print(t)
        viewer.plot()
    viewer.plot()
