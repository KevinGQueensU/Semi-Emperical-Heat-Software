import fipy
import numpy as np
import scipy.constants as const
from fipy import FaceVariable, CellVariable, Grid2D, Grid3D, TransientTerm, DiffusionTerm, Viewer, meshes
from fipy.meshes.uniformGrid import UniformGrid
from fipy.meshes.uniformGrid3D import UniformGrid3D
from fipy.boundaryConditions import FixedValue, FixedFlux
from fipy.tools import numerix as nx

class BoundaryConditions:
    TYPES = ('None', 'Fixed', 'BBR') # BBR = Blackbody radiation

    def __init__(self,
                types: np.ndarray[str],                 # Specify type for each face. Formatted as [top, bottom, left, right] (2D)
                                                        # or [top, bottom, left, right, front, back] (3D)
                 T0: np.ndarray[float] | float,         # The initial temperature(s) for each face [K]
                 T_amb: float | None = None,            # The ambient temperature(s) for BBR
                 eps: np.ndarray[float] | float = 1,    # BBR emissivity coefficient(s) [0, 1]
                 ) -> None:

        # Test cases
        if(np.any(T0 <= 0) or (T_amb != None and T_amb <= 0)):
            raise ValueError('Temperature must be positive')
        if(np.iterable(T0) and len(T0) != len(types)):
            raise ValueError("Number of T0s != number of BC's")
        if(np.iterable(eps) and len(eps) != len(types)):
            raise ValueError("Number of emissivity coeff. != number of BC's")
        for i, type in enumerate(types):
            if type not in self.TYPES:
                raise ValueError('BC' + str(i) +  'is not one of {}'.format(self.TYPES))
            if type == 'BBR' and T_amb is None:
                raise ValueError('No ambient temperature given for radiation boundary')

        # Making single values into arrays for easier iteration
        if not np.iterable(T0):
            T0 = np.array(([float(T0)] * 6))
        if not np.iterable(eps):
            eps = np.array(([float(eps)] * 6))

        self.types = types
        self.T0s = T0
        self.eps = eps
        self.T_amb = T_amb
        self.eps = self.eps

        self.faces = None

    # FUNCTION: Pass in custom FiPy masks where BCs are to be applied
    def set_masks(self,
                  masks: np.ndarray[UniformGrid] | np.ndarray[UniformGrid3D]
                  ) -> None:
        if(len(masks) != len(self.types)):
            raise ValueError("Number of masks != number of BC's")
        self.faces = masks

    # FUNCTION: Update boundary conditions based on current temperature, returns and stores BCs.
    def update(self,
                  mesh: meshes,    # Pass in the FiPy mesh
                  T: CellVariable  # CellVariable temperature in [K]
               ) -> list[fipy.boundaryConditions]:

        # Initializing masks if it has not already been set.
        if(self.faces == None):
            self.faces = [mesh.facesTop, mesh.facesBottom, \
                          mesh.facesLeft, mesh.facesRight]

            if(mesh.dim == 3):
                self.faces.append([mesh.facesFront, mesh.facesBack])


        # Init. face values -> set to 0 everywhere, set qn only on selected faces
        val = FaceVariable(mesh=mesh, value=0.0)
        bcs = []

        for i, type in enumerate(self.types):

            if(type == 'Fixed'): # Append individually for Dirhlect
                bcs.append(FixedValue(faces=self.faces[i], value=self.T0s[i]))

            elif(type == 'BBR'): # Apply values to each face and then append afterwards for BBR
                Tface = T.faceValue
                h = 4.0 * float(self.eps[i]) * const.Stefan_Boltzmann * (nx.maximum(Tface, 0.0) ** 3.0)
                qn = h * (Tface - float(self.T_amb))  # W/m^2, heat leaving the domain

                # Set qn only on selected faces
                val.setValue(qn, where = self.faces[i])

        if(self.T_amb != None):
            bcs.append(FixedFlux(faces=mesh.exteriorFaces, value=val))

        self.last_bcs = bcs

        return bcs


if __name__ == "__main__":
    ##OLIVINE
    from Beam import Beam
    from Medium import Atom, Medium
    import simulation as sim

    def C_f(T):
        # constants as floats
        a = 87.36
        b = 8.717e-2
        c = -3.699e6
        d = 8.436e2
        e = -2.237e-5
        M = 0.14069  # kg/mol (forsterite)

        # ensure floating FiPy variable inside the expression
        Tf = T + 0.0

        # avoid negative integer exponents: use reciprocals with float powers
        Cp = a + b * Tf + c * (1.0 / (Tf ** 2.0)) + d * (1.0 / (Tf ** 0.5)) + e * (Tf ** 2.0)

        # guard against tiny negatives from roundoff
        return nx.maximum(Cp / M, 0.0)


    def k_f(T, k0=1.7, *, Tmin=160, Tmax=6000.0, kmin=1e-4, kmax=50.0):
        # ensure float and keep T in a numerically safe range
        Tf = nx.clip(1.0 * T, Tmin, Tmax)

        # your original law, evaluated on the safe Tf
        k = k0 * nx.exp(-(Tf - 298.0) / 300.0)

        # keep operator strictly elliptic & avoid overflow
        return nx.clip(k, kmin, kmax)


    Z_Mg = 12
    A_Mg = 24.305  # g/mol
    Mg = Atom('Mg', Z_Mg, A_Mg, 0.2222, )

    Z_Fe = 26
    A_Fe = 55.845  # g/mol
    Fe = Atom('Fe', Z_Fe, A_Fe, 0.2222)

    Z_Si = 14
    A_Si = 28.086  # g/mol
    Si = Atom('Si', Z_Si, A_Si, 0.1111)

    Z_O = 8
    A_O = 15.999  # g/mol
    O = Atom('O', Z_O, A_O, 0.4444)

    I_0 = 6.24e11 * 5  # [s^-1]
    E_0 = 8e6  # [MeV]
    r = 1e-4  # m
    sig_ga_y0 = r
    sig_ga_z0 = r
    Z = 1
    x0 = 0
    beam = Beam(E_0, I_0, Z, sig_ga_y0=sig_ga_y0, sig_ga_z0=sig_ga_z0)

    rho = 3.3  # g/cm^3
    Lx = 0.5e-3
    dx = 0.5e-4
    Ly = 1.0e-3
    dy = 0.01e-3
    Lz = Ly
    medium = Medium(rho, [Mg, Fe, Si, O], Lx, Ly, Lz, "Olivine//Hydrogen in Olivine.txt", beam)

    # %%
    if True:
        xk = np.linspace(0, Lx, 10000)  # cell edges
        cell_width = xk[1] - xk[0]
        cj = 0.5 * (xk[:-1] + xk[1:])  # cell centers
        alpha = beta = 0.0
        y = z = 0

        E_beam = np.empty(xk.size + 1)  #
        dIdx = np.empty(xk.size + 1)
        E_inst = np.empty(xk.size + 1)
        I_beam = np.empty(xk.size + 1)
        dEb_dx = np.zeros_like(xk)  # eV/(m·s)
        dEdx = np.zeros_like(xk)
        dEdx_beam = np.zeros_like(xk)
        phi_free = np.array([beam.PD(x, y, z, alpha, beta) for x in xk])  # Free energy flux eV/(m^2·s)

        E_beam[0] = E_0 * I_0
        E_inst[0] = E_0  # eV
        I_beam[0] = I_0  # 1/s

        for k, j in enumerate(cj):
            # Get energy gradient
            dEdx[k] = medium.get_dEdx(E_inst[k])
            dEdx_beam[k] = dEdx[k] * I_beam[k] + E_inst[k] * dIdx[k]
            I_beam[k + 1] = I_beam[k]
            E_beam[k + 1] = max(E_beam[k] + dEdx_beam[k] * cell_width, 0)
            E_inst[k + 1] = E_beam[k + 1] / I_beam[k + 1]

        print("initial finished")
        dx = cell_width
        N = len(cj)
        dEb_dx_test = np.zeros_like(dEb_dx)
        for k in range(N):
            dEb_dx[k] -= I_beam[k] * dEdx[k]

        eV_to_J = 1.602176634e-19
        dEb_dx_W = dEb_dx * eV_to_J  # W/m
        dEb_dx_kW_mm = dEb_dx_W / 1e6  # kW/mm

    # 2D BB Radiation
    if True:
        Lx = 10e-4
        dx = 0.05e-4
        Ly = 10e-4
        dy = 0.1e-4
        Lz = Ly

        # Parameters for the beam and values
        I_0 = 1.00  # [s^-1]
        E_0 = 1e6  # [MeV]
        r = 0.5e-6  # m
        sig_ga_y0 = r
        sig_ga_z0 = r
        Z = 1
        x0 = 0
        beam = Beam(E_0, I_0, Z, sig_ga_y0=sig_ga_y0, sig_ga_z0=sig_ga_z0)

        # Parameters for the medium and values
        Z_Mg = 12
        A_Mg = 24.305  # g/mol
        Mg = Atom('Mg', Z_Mg, A_Mg, 0.2222, )

        Z_Fe = 9
        A_Fe = 18.998  # g/mol
        Fe = Atom('Mg', Z_Fe, A_Fe, 0.2222)

        Z_Si = 14
        A_Si = 28.086  # g/mol
        Si = Atom('Mg', Z_Si, A_Si, 0.1111)

        Z_O = 8
        A_O = 15.999  # g/mol
        O = Atom('Mg', Z_O, A_O, 0.4444)

        rho = 3.3*1e3  # g/cm^3
        n = 9.8e23 * 1e6  # 1/m^3
        W = H = 10e-6  # 10 mm square slab
        A_xsec = W * H
        P_perim = 2 * (W + H)
        medium = Medium(rho, [Mg, Fe, Si, O], Lx, Ly, Lz, "Olivine//Hydrogen in Olivine.txt", beam,
                        x0 = 0)

        BC = BoundaryConditions(['BBR', 'BBR', 'BBR', 'BBR'],
                                298, 200)

        sim.heateq_solid_2d(beam, medium, BC, Lx, Ly, rho, C_f, k_f, SE=1, t_end = 1000, T0 = 500,
                            dx=dx, dy=dy, dt=0.00001, view=True, view_freq=0.0001, dt_ramp = 1.1)