import fipy
import numpy as np
import scipy.constants as const
import scipy.interpolate
from fipy import FaceVariable, CellVariable, Gmsh3D, Grid2D, Grid3D, TransientTerm, DiffusionTerm, Viewer, meshes, \
    ImplicitSourceTerm
from fipy.meshes.uniformGrid import UniformGrid
from fipy.meshes.uniformGrid3D import UniformGrid3D
from fipy.boundaryConditions import FixedValue, FixedFlux
from fipy.tools import numerix as nx

class BoundaryConditions:
    TYPES = ('None', 'Fixed', 'BBR')
    def __init__(self,
                 types: np.ndarray[str] | None = None,
                 T0: float | np.ndarray[float] | None = None,
                 T_amb: float | np.ndarray[float] | None = None,
                 eps: float | np.ndarray[float] = 1,
                 ) -> None:

        n_BBR = 0
        n_fixed = 0

        # Count types first
        for i, type in enumerate(types):
            if type not in self.TYPES:
                raise ValueError('BC' + str(i) + ' is not one of {}'.format(self.TYPES))
            if type == 'BBR':
                if T_amb is None:
                    raise ValueError('No ambient temperature given for radiation boundary')
                n_BBR += 1
            if type == 'Fixed':
                if T0 is None:
                    raise ValueError('No surface temperature given for fixed boundary')
                n_fixed += 1

        # Convert scalars for easier iteration
        if not np.iterable(T0):
            T0 = np.array([float(T0)] * n_fixed) if n_fixed > 0 else np.array([])
        else:
            T0 = np.asarray(T0, dtype=float)
        if not np.iterable(eps):
            eps = np.array([float(eps)] * n_BBR) if n_BBR > 0 else np.array([])
        else:
            eps = np.asarray(eps, dtype=float)

        # Catch value errors
        if np.any(T0 <= 0) or (T_amb is not None and T_amb <= 0):
            raise ValueError('Temperature must be positive')
        if len(T0) != n_fixed:
            raise ValueError("Number of T0s != number of Fixed BC's")
        if len(eps) != n_BBR:
            raise ValueError("Number of emissivity coeff. != number of BBR BC's")

        self.types = types
        self.T0s = T0
        self.eps = eps
        self.T_amb = T_amb
        self.n_BBR = n_BBR
        self.n_fixed = n_fixed
        self.n_none = len(self.types) - n_BBR - n_fixed
        self.last_bcs = None
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

        for i, type_i in enumerate(self.types):
            if type_i == 'Fixed': # Append individually for Dirichlet
                bcs.append(FixedValue(faces=self.faces[i], value=self.T0s[j_fixed]))
                j_fixed += 1

            elif type_i == 'BBR': # Apply values to each face and then append afterwards for BBR
                Tface = T.faceValue
                h = float(self.eps[j_bbr]) * const.Stefan_Boltzmann * (Tface ** 2 + self.T_amb ** 2) * (Tface + self.T_amb)
                qn = h * (Tface - float(self.T_amb))  # W/m^2, heat leaving the domain
                qn = nx.maximum(qn, 0.0)
                # Set qn only on selected faces
                val.setValue(qn, where = self.faces[i])
                bbr_faces = self.faces[i] if bbr_faces is None else (bbr_faces | self.faces[i])
                j_bbr += 1

        if self.T_amb != None:
            bcs.append(FixedFlux(faces=bbr_faces, value=val))

        self.last_bcs = bcs

        return bcs

class BoundaryConditionsGmsh:
    def __init__(
            self,
            mesh: Gmsh3D,
            T0: float | np.ndarray[float] = None,         # Fixed temperature(S)
            T_amb: float = None ,                       # Ambient temperature for BBR
            eps: float | np.ndarray[float] = None,      # BBR emissivity coefficients
            h: float | np.ndarray[float] = None,        # Interface thermal conductivity
            R: float | np.ndarray[float] = None,        # Internal boundary thermal contact resistance
                ) -> None:

        materials = {}

        # Unit test cases
        if(T0 is not None and not np.iterable(T0)):
            T0 = np.asarray([T0]).astype(np.float64)
        if(h is not None and not np.iterable(h)):
            h = np.asarray([h]).astype(np.float64)
        if T_amb is not None:
            T_amb = float(T_amb)
        if(eps is not None and not np.iterable(eps)):
            eps = np.asarray([eps]).astype(np.float64)
        if(R is not None and not np.iterable(R)):
            R = np.asarray([R]).astype(np.float64)


        exterior_faces = mesh.exteriorFaces

        # Mat are the physical volume names
        for mat in mesh.physicalCells:
            cellMask = mesh.physicalCells[mat]  # (nCells,)
            selectedCellIDs = np.where(cellMask)[0]

            # Faces belonging to selected cells and materials
            face_ids = mesh.cellFaceIDs[:, selectedCellIDs]
            matFaces = np.zeros(mesh.numberOfFaces, dtype=bool)
            matFaces[face_ids] = True

            # Build dict per material
            boundary_masks = {}

            # Build masks for every named physical face group in the mesh
            for face_name in mesh.physicalFaces:
                faceMask = mesh.physicalFaces[face_name]
                boundary_masks[face_name] = (faceMask & matFaces) & exterior_faces

            materials[mat] = boundary_masks

        # FIXED
        self.T0s = T0

        # BBR
        self.T_amb = T_amb
        self.eps = eps
        self.bbr_explicit = None
        self.bbr_implicit = None

        # Interface
        self.h = h

        # Thermal contact/Internal
        self.R = R

        # Materials and their corresponding mask
        self.mats = materials

        self.last_bcs = None

    # FUNCTION: To work with FiPy's implicit solver, we need to build BBR as an implicit source + explicit term, so we initialize cellvariables
    def init_implicit_bbr(self, mesh):
        from fipy import ImplicitSourceTerm
        self._bbr_h_cell = CellVariable(mesh=mesh, value=0.0)
        self._bbr_src_cell = CellVariable(mesh=mesh, value=0.0)
        self.bbr_implicit = ImplicitSourceTerm(coeff=-self._bbr_h_cell)
        self.bbr_explicit = self._bbr_src_cell

    #FUNCTION: Update the value of the boundary conditions given the mesh and tmeperature field
    def update(self, mesh, T):

        bcs = []
        val_interface = FaceVariable(mesh=mesh, value=0.0)
        interface_faces = None

        self._bbr_h_cell.setValue(0.0)
        self._bbr_src_cell.setValue(0.0)

        for i_mat, mat in enumerate(self.mats):
            for bnd in self.mats[mat]:
                bnd_check = bnd.replace(" ", "").lower()

                if bnd_check == 'none' or bnd_check == 'internal':
                    continue

                face_mask = self.mats[mat][bnd]

                if bnd_check == 'fixed':
                    bcs.append(FixedValue(faces=FaceVariable(mesh=mesh, value=face_mask),
                                          value=self.T0s[i_mat]))

                elif bnd_check == 'bbr':
                    # Calculate BBR value
                    Tface = T.faceValue
                    eps = float(self.eps[i_mat])
                    T_amb = float(self.T_amb)
                    h_face = eps * const.Stefan_Boltzmann * (Tface ** 2 + T_amb ** 2) * (Tface + T_amb)
                    # Get the ID's of the cells and faces with BBR
                    face_ids = np.where(np.asarray(face_mask))[0]
                    id0 = np.asarray(mesh.faceCellIDs[0])[face_ids]
                    id1 = np.asarray(mesh.faceCellIDs[1])[face_ids]
                    cell_ids = np.where(id1 >= 0, id0, id0)
                    cell_ids = np.where(id0 < 0, id1, cell_ids)
                    cell_ids = np.where(id0 >= mesh.numberOfCells, id1, cell_ids)

                    # Compute the area and volumes of those cells
                    face_areas = np.asarray(mesh._faceAreas)[face_ids]
                    cell_vols = np.asarray(mesh.cellVolumes)[cell_ids]
                    ratio = face_areas / cell_vols

                    # Build on temp arrays then setValue
                    h_vals = np.asarray(h_face)[face_ids]
                    h_tmp = np.asarray(self._bbr_h_cell.value).copy()
                    src_tmp = np.asarray(self._bbr_src_cell.value).copy()
                    np.add.at(h_tmp, cell_ids, h_vals * ratio)
                    np.add.at(src_tmp, cell_ids, h_vals * T_amb * ratio)

                    self._bbr_h_cell.setValue(h_tmp)
                    self._bbr_src_cell.setValue(src_tmp)

                elif bnd_check == 'interface':
                    Tface = T.faceValue
                    face_var = FaceVariable(mesh=mesh, value=face_mask)
                    qn = self.h[i_mat] * (Tface - float(self.T0s[i_mat]))
                    val_interface.setValue(qn, where=face_var)
                    interface_faces = face_var if interface_faces is None else (interface_faces | face_var)

                else:
                    print(f"Boundary not recognized: {bnd}")

        if interface_faces is not None:
            bcs.append(FixedFlux(faces=interface_faces, value=val_interface))

        self.last_bcs = bcs
        return bcs

