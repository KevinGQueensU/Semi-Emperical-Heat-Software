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
                 T0: float | np.ndarray[float] | None = None,       # The initial temperature(s) for each fixed face [K]
                 T_amb: float | np.ndarray[float] | None = None,    # The ambient temperature(s) for BBR
                 eps: float | np.ndarray[float] = 1,    # BBR emissivity coefficient(s) [0, 1]
                 ) -> None:
        n_BBR = 0
        n_fixed = 0

        # Test cases
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