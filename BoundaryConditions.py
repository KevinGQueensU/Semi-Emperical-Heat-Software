import fipy
import numpy as np
import scipy.constants as const
import scipy.interpolate
from fipy import FaceVariable, CellVariable, Gmsh3D, Grid2D, Grid3D, TransientTerm, DiffusionTerm, Viewer, meshes
from fipy.meshes.uniformGrid import UniformGrid
from fipy.meshes.uniformGrid3D import UniformGrid3D
from fipy.boundaryConditions import FixedValue, FixedFlux
from fipy.tools import numerix as nx

TYPES = ('None', 'Fixed', 'BBR')  # BBR = Blackbody radiation
class BoundaryConditions:


    def __init__(self,
                 mesh = None,
                 types: np.ndarray[str] | None = None,                 # Specify type for each face. Formatted as [top, bottom, left, right] (2D)
                                                        # or [top, bottom, left, right, front, back] (3D)
                 T0: float | np.ndarray[float] | None = None,       # The initial temperature(s) for each fixed face [K]
                 T_amb: float | np.ndarray[float] | None = None,    # The ambient temperature(s) for BBR
                 eps: float | np.ndarray[float] = 1,    # BBR emissivity coefficient(s) [0, 1]
                 ) -> None:
        n_BBR = 0
        n_fixed = 0

        # Test cases
        if mesh is None:
            for i, type in enumerate(types):
                if type not in self.TYPES:
                    raise ValueError('BC' + str(i) + 'is not one of {}'.format(self.TYPES))
                if type == 'BBR':
                    if(T_amb is None):
                        raise ValueError('No ambient temperature given for radiation boundary')
                    n_BBR += 1
                if type == 'Fixed':
                    if(T0 is None):
                        raise ValueError('No surface temperature given for fixed boundary')
                    n_fixed += 1
            if(np.any(T0 <= 0) or (T_amb != None and T_amb <= 0)):
                raise ValueError('Temperature must be positive')
            if(np.iterable(T0) and len(T0) != len(n_fixed)):
                raise ValueError("Number of T0s != number of BC's")
            if(np.iterable(eps) and len(eps) != len(n_BBR)):
                raise ValueError("Number of emissivity coeff. != number of BC's")

            # Making single values into arrays for easier iteration
            if not np.iterable(T0):
                T0 = np.array(([float(T0)] * n_fixed))
            if not np.iterable(eps):
                eps = np.array(([float(eps)] * n_BBR))
        self.types = types
        self.T0s = T0
        self.eps = eps
        self.T_amb = T_amb
        self.eps = self.eps
        self.n_BBR = n_BBR
        self.n_fixed = n_fixed
        self.n_none = len(self.types) - n_BBR - n_fixed

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
                self.faces.append(mesh.facesFront)
                self.faces.append(mesh.facesBack)


        # Init. face values -> set to 0 everywhere, set qn only on selected faces
        val = FaceVariable(mesh=mesh, value=0.0)
        bcs = []
        j_fixed = 0
        bbr_faces = None # Only apply flux boundary condition to faces with BBR
        j_bbr = 0

        for i, type in enumerate(self.types):

            if(type == 'Fixed'): # Append individually for Dirhlect
                bcs.append(FixedValue(faces=self.faces[i], value=self.T0s[j_fixed]))
                j_fixed += 1

            elif(type == 'BBR'): # Apply values to each face and then append afterwards for BBR
                Tface = T.faceValue
                h = 4.0 * float(self.eps[j_bbr]) * const.Stefan_Boltzmann * (nx.maximum(Tface, 0.0) ** 3.0)
                qn = h * (Tface - float(self.T_amb))  # W/m^2, heat leaving the domain

                # Set qn only on selected faces
                val.setValue(qn, where = self.faces[i])
                bbr_faces = self.faces[i] if bbr_faces is None else (bbr_faces | self.faces[i])
                j_bbr += 1

        if(self.T_amb != None):
            bcs.append(FixedFlux(faces=bbr_faces, value=val))

        self.last_bcs = bcs

        return bcs
import numpy as np
import numpy.ma as ma
class BoundaryConditionsGmsh:
    def __init__(
            self,
            mesh: Gmsh3D,
            T0=None,          # float | array-like | None
            T_amb=None,       # float | None
            eps: float | np.ndarray[float] =1,            # float | array-like
                ) -> None:
        materials = {}
        T0 = np.array(T0).astype(np.float64)
        if T_amb is not None:
            T_amb = float(T_amb)
        eps = np.array(eps).astype(np.float64)

        # Ensure this is a plain bool array (FiPy may return a FaceVariable-like object)
        exterior_faces = mesh.exteriorFaces

        for mat in mesh.physicalCells:
            cellMask = mesh.physicalCells[mat]  # (nCells,)
            selectedCellIDs = np.where(cellMask)[0]

            # Faces belonging to selected cells
            face_ids = mesh.cellFaceIDs[:, selectedCellIDs]

            matFaces = np.zeros(mesh.numberOfFaces, dtype=bool)
            matFaces[face_ids] = True

            # dict per material
            boundary_masks = {}

            # Build masks for every named physical face group in the mesh
            for face_name in mesh.physicalFaces:
                faceMask = mesh.physicalFaces[face_name]
                boundary_masks[face_name] = (faceMask & matFaces) & exterior_faces

            materials[mat] = boundary_masks

        self.T0s = None if T0 is None else np.asarray(T0)
        self.eps = np.asarray(eps)
        self.T_amb = T_amb

        # material -> boundary_name -> face mask
        self.mats = materials

    def update(self,
               mesh: meshes,
               T: CellVariable ) -> list[fipy.boundaryConditions]:
        # Init. face values -> set to 0 everywhere, set qn only on selected faces
        val = FaceVariable(mesh=mesh, value=0.0)
        bcs = []
        j_fixed = 0
        j_bbr = 0
        bbr_faces = None  # Only apply flux boundary condition to faces with BBR
        for mat in self.mats:
            for bnd in self.mats[mat]:
                if(bnd ==  'None' or bnd == 'Interface'):
                    continue
                faces = FaceVariable(mesh=mesh, value=self.mats[mat][bnd])
                if (bnd == 'Fixed'):  # Append individually for Dirhlect
                    bcs.append(FixedValue(faces=faces, value=self.T0s[j_fixed]))
                elif (bnd == 'BBR'):  # Apply values to each face and then append afterwards for BBR
                    Tface = T.faceValue
                    h = 4.0 * float(self.eps[j_bbr]) * const.Stefan_Boltzmann * (nx.maximum(Tface, 0.0) ** 3.0)
                    qn = h * (Tface - float(self.T_amb))  # W/m^2, heat leaving the domain

                    # Set qn only on selected faces
                    val.setValue(qn, where=self.mats[mat][bnd])
                    bbr_faces = faces if bbr_faces is None else (bbr_faces | faces)
                else:
                    print(f"Boundary not recongized : {bnd}")
            j_fixed += 1
            j_bbr += 1
        if(self.T_amb != None):
            bcs.append(FixedFlux(faces=bbr_faces, value=val))

        self.last_bcs = bcs
        return bcs
#%%
if True:
    #%%
    import numpy as np
    from Beam import Beam
    from Medium import Atom, Medium
    from fipy.tools import numerix as nx
    import scipy as sp

    gmsh_mesh = Gmsh3D("Holder_V3.geo")
    mesh = gmsh_mesh * 1e-2

    # force physical masks to be numpy bool arrays (no mesh reference inside)
    mesh.physicalCells = {k: np.asarray(v, dtype=bool) for k, v in gmsh_mesh.physicalCells.items()}
    mesh.physicalFaces = {k: np.asarray(v, dtype=bool) for k, v in gmsh_mesh.physicalFaces.items()}

    # also detach exteriorFaces
    mesh.exteriorFaces = np.asarray(gmsh_mesh.exteriorFaces, dtype=bool)

    BC = BoundaryConditionsGmsh(mesh, T0 = [298, 298, 298], T_amb = 298, eps = [0.2, 0.2, 0.2])

    I_0 = 3.12e20  # [s^-1] 1 muAmps
    E_0 = 4e6  # [MeV]
    L = 1e-2
    Z = 1
    beam = Beam(E_0, I_0, Z, L = L, W = L, dim=3, type = 'Rectangular')


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
        f_low = scipy.interpolate.PchipInterpolator(Ts_low, Cp_low_J_per_molK)
        val1 = f_low(T_safe) / M_Ta  # -> J/(kg K)


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
                      "Ta//Hydrogen in Tantalum.txt", name='Tantalum')
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
        return nx.maximum(Cp, 0.0)


    # https://link.springer.com/article/10.1007/BF00514474
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
        return nx.maximum(k_combined, 1.0)

    # Define aluminum
    Z_Al = 13
    A_Al = 27  # g/mol
    Al = Atom('Al', Z_Al, A_Al, 1)
    rho_Al = 2700  # kg/m^3

    aluminum = Medium(rho_Al, C_Al, k_Al,
                      Al,
                      "Al//Hydrogen in Aluminum.txt", name='Aluminum')

    def C_olivine(T):
        M = 0.14069  # kg/mol
        T_safe = nx.clip(T, 200, 3000)

        a, b, c, d, e = 87.36, 8.717e-2, -3.699e6, 8.436e2, -2.237e-5

        Cp = a + b * T_safe + c / (T_safe ** 2) + d / (T_safe ** 0.5) + e * (T_safe ** 2)
        return nx.maximum(Cp / M, 10.0)


    def k_olivine(T):
        # ensure float and keep T in a numerically safe range
        Tf = nx.clip(1.0 * T, 160, 6000)
        k = 1.7 * nx.exp(-(Tf - 298.0) / 300.0)

        return nx.clip(k, 1e-4, 50.0)


    rho_olivine = 3.3e3  # kg/m^3

    # Parameters for the medium and values
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

    olivine = Medium(rho_olivine, C_olivine, k_olivine,
                     [Mg, Fe, Si, O],
                     "Olivine//Hydrogen in Olivine.txt",
                     name='Olivine')
    #%%
    if __name__ == "__main__":
        from Simulation import heateq_solid_3d_test

        mediums = [tantalum, olivine, aluminum]
        print(f"N_tot olivine: {olivine.N_tot:.3e}")  # should be ~7e28 atoms/m³
        print(f"N_tot aluminum: {aluminum.N_tot:.3e}")  # should be ~6e28 atoms/m³
        print(f"Se at 5MeV: {olivine.get_Se_ev_m(5e6):.3e} eV/m")  # should be ~1e9 eV/m
        ts, Ts = heateq_solid_3d_test(
            beam, mediums, mesh, BC,
            10000, T0=298.0,
            x_scale = 1e3, x_scale_min = 0,
            dt=1e-6, alpha=0, beta=0,
            view=True, dt_ramp=2, dT_target=5000,
        )
        print("hello")