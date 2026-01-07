import numpy as np
import re
import scipy.constants
import scipy.integrate
import scipy.interpolate as interpolate
from Beam import Beam

# CONVERSION FACTORS TO eV
E_TO_EV = {"ev": 1,
              "kev": 1e3,
              "mev": 1e6,
              "gev": 1e9}


#### HELPER FUNCTIONS ####

# FUNCTION: Sort 2 arrays based on input array x values and removes duplicates
def sorted_unique_xy(x: np.ndarray, y: np.ndarray,  # Input Arrays
                     agg: str ="last"  # Determines which value of y to take for a duplicate x. 'first' takes the first element and 'last' takes the last.
                     ):
    # Convert to numpy arrays
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    # Mask out infinite values
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]

    # Sort by x-array
    idx = np.argsort(x, kind="mergesort") # Gives back indices that sort the x-array
    x = x[idx]
    y = y[idx]

    # Collapse all the duplicates values of x into one spot
    ux, start_idx, counts = np.unique(x, return_index=True, return_counts=True)
    if agg == "first":
        # Only take the first value of y for the duplicate x
        uy = y[start_idx]
    elif agg == "last":
        # Only take the last index of y for the duplicate x
        last_idx = start_idx + counts - 1
        uy = y[last_idx]
    else:
        # Take the mean value of y among the duplicate x's
        uy = np.add.reduceat(y, start_idx) / counts

    # Return the unique sorted arrays
    return ux, uy

# FUNCTION: Read a file and return the first two cols
def read_file(filename):

    try:
        file = open(filename, 'r')
    except:
        print('File not found')

    #init columns
    col1 = np.empty(0)
    col2 = np.empty(0)

    for line in file:
        # Strip leading white space and check if it's a comment
        stripped_line = line.strip()
        if not stripped_line or stripped_line.startswith('#'):
            continue  # Skip empty lines and comment lines

        # Split the line by tab delimiter
        data_elements = stripped_line.split()
        col1 = np.append(col1, float(data_elements[0]))
        col2 = np.append(col2, float(data_elements[1]))

    return col1, col2

# FUNCTION: Read a SRIM stopping power file and return the energies [MeV], electronic and nuclear stopping powers [eV/(m^2)]
def read_SRIM_ev_atom_m2(filename):

    with open(filename, "r", encoding="utf-8", errors="ignore") as f:
        txt = f.read()

    # find multiplier for eV / (1E15 atoms/cm2) in the footer
    target_label = "eV / (1E15 atoms/cm2)"
    mul_block_start = re.search(r"Multiply Stopping by", txt, flags=re.IGNORECASE)
    if not mul_block_start:
        raise ValueError("Multiplier table not found in SRIM file.")

    factor = None

    # Scan through each line looking for the conversion factor
    for line in txt[mul_block_start.end():].splitlines():

        s = line.strip()
        if not s or s.startswith("=") or s.startswith("(C)"):
            if factor is not None:
                break
            continue

        # checking for the format (number -> exponent(?) -> text)
        m = re.match(r"^\s*([0-9.]+(?:[Ee][+-]?\d+)?)\s+(.+?)\s*$", line)
        if not m:
            continue

        val = float(m.group(1)) # pull the conversion value
        label = " ".join(m.group(2).split()) # pull the type of conversion x to y
        if label == target_label:
            factor = val
            break
    if factor is None:
        raise ValueError(f"Unit '{target_label}' not found in multiplier table.")

    factor *= 1e-4 * 1e-15 # convert to m^2 and per atom

    # Format for stopping power (Energy value (decimal or integer) -> Energy unit (str) -> Elec. dEdx (scientific) -> Nucl dEdx (scientific))
    row_re = re.compile(
        r"""^\s*
            (?P<E>\d+(?:\.\d+)?)\s*
            (?P<Unit>eV|keV|MeV|GeV)
            \s+
            (?P<Elec>[+-]?\d+(?:\.\d+)?[Ee][+-]?\d+)
            \s+
            (?P<Nucl>[+-]?\d+(?:\.\d+)?[Ee][+-]?\d+)
        """,
        re.IGNORECASE | re.VERBOSE
    )

    # Scan through rest of text and store (E, dEdx_elec, dEdx_nucl)
    E_list, elec_list, nucl_list = [], [], []
    for line in txt.splitlines():
        m = row_re.match(line)
        if not m:
            continue
        E = float(m.group("E"))
        unit = m.group("Unit").lower()
        elec = float(m.group("Elec"))
        nucl = float(m.group("Nucl"))

        E_list.append(E * E_TO_EV[unit])    # -> MeV
        elec_list.append(elec * factor)     # -> eV/(m^2)
        nucl_list.append(nucl * factor)     # -> eV/(m^2)

    if not E_list:
        raise ValueError("No SRIM numeric rows found in file.")

    return np.array(E_list), np.array(elec_list), np.array(nucl_list)


#### CLASSES ####

## ATOM CLASS used to define constituent atoms of a medium
class Atom:
    # CONSTRUCTOR: that takes in name, atomic number Z, atomic mass A (g/mol), atomic fraction f,
    # filepath with cross sections (p,x)
    def __init__(self,
                 name: str,     # Atomic symbol (or arbitrary)
                 Z: int,        # Atomic number
                 A: int,        # Atomic mass (g/mol)
                 f,             # Atomic fraction of medium
                 sig_file = None,   # OPTIONAL filepath with cross-sections (p,x) (nuclear reactions)
                 E_conv = 1,        # OPTIONAL conversion factor for (p, x) energy values to get eV
                 sig_conv = 1e-28   # OPTIONAL conversion factor for (p, x) cross-section values to get m^2 (default is barns -> m^2)
                 ):
        # Setting parameters
        self.name = name
        self.Z = Z # > 0
        self.A = A # g/mol
        self.f = f # 0 < f < 1
        self.sig_file = sig_file

        # Store values of energies and cross-sections in (eV, m^2)
        if(sig_file is not None):
            self.Es, self.sig_bs = read_file(sig_file)
            self.Es = np.r_[0, self.Es] * E_conv  # -> eV
            self.sig_bs =  np.r_[0, self.sig_bs] * sig_conv # -> m^2
            self.Es, self.sig_bs = sorted_unique_xy(self.Es, self.sig_bs)
            self.sig_interp = interpolate.PchipInterpolator(self.Es, self.sig_bs, extrapolate=True) # KIND OF CRAPPY WAY TO EXTRAPOLATE, DATA HAS TO BE GOOD OTHERWISE BLOWS UP

    # FUNCTION: Multiply stored cross-sections by some factor
    def conv_sig(self, factor):
        if(self.sig_file is None):
            raise ValueError("Cross section file is not defined")
        self.sig_bs *= factor

    # FUNCTION: Multiply stored energies by some factor
    def conv_E(self, factor):
        if(self.sig_file is None):
            raise ValueError("Cross section file is not defined")
        self.Es *= factor

    # FUNCTION: Read in a new cross-section file
    def set_sig(self, file_name, E_conv = 1, sig_conv = 1e-28):
        self.Es, self.sig_bs = read_file(file_name)
        self.sig_bs = np.r_[0, self.sig_bs] * sig_conv  # -> m^2
        self.Es, self.sig_bs = sorted_unique_xy(self.Es, self.sig_bs)
        self.sig_interp = interpolate.PchipInterpolator(self.Es, self.sig_bs, extrapolate=True)

    # FUNCTION: Input an energy, return the extrapolated cross-section (p,x)
    def get_sig(self, E):
        if(self.sig_file is None):
            return 0
        return max(self.sig_interp(E), 0.0) # CANNOT BE NEGATIVE


## MEDIUM CLASS: Used in heat generation modelling to calculate energy gradient
## EQUATIONS BASED OFF https://doi.org/10.1016/j.nimb.2019.09.016
class Medium:
    RFWD = 0.6 # Optimum forward to back scattering ratio from paper for (p, x) nuclear reactions

    ## CONSTRUCTOR: Initializes the medium dimensions and material properties
    def __init__(self,
                 rho: float,          # Bulk density [g/cm^3]
                 atoms: np.ndarray(Atom) | list(Atom) | tuple(Atom) | Atom,  # Constituent atoms that make up the material
                 Lx: float, Ly: float, Lz: float, # Dimensions of medium
                 dEdx_filename: str,  # SRIM file for stopping powers
                 beam: Beam,          # Particle beam being shot into material
                 x0: float = 0        # Starting location of Medium
                 ):

        if isinstance(atoms, (list, tuple, np.ndarray)):
            self.atoms = np.array(list(atoms), dtype=object).reshape(-1)
        else:
            # if only a single atom is passed
            self.atoms = np.array([atoms], dtype=object)

        # Defining parameters
        self.rho = rho
        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz
        self.x0 = x0
        self.beam = beam
        self.filename = dEdx_filename

        self.A = Ly * Lz            # (y, z) cross-sectional area [m^2]
        self.P = 2*(Ly + Lz)        # (y, z) perimeter [m]
        self.r = 2*self.A/self.P    # (y, z) cross-sectional radius m
        self.LBD = 0                # fitting parameter, initially set to 0

        self.Es, self.Se_elec, self.Se_nucl = read_SRIM_ev_atom_m2(dEdx_filename)
        self.Se = self.Se_elec + self.Se_nucl   # total stopping power
        self.Es = np.r_[0, self.Es]             # clip so that at 0 energy there is 0 stopping power
        self.Se = np.r_[0, self.Se]
        self.Es, self.Se = sorted_unique_xy(self.Es, self.Se)
        self.dEdx_interp = interpolate.PchipInterpolator(self.Es, self.Se, extrapolate = True)

        self._compute_number_densities()

    # PRIVATE FUNCTION: Compute Nd for each atom based on fraction and mass, store as dict
    def _compute_number_densities(self):
        # Average molar mass per atom [g/mol] of the medium, sum over all (atomic fraction * atomic mass)
        M_bar = sum(a.f * a.A for a in self.atoms)

        # Calculate total number density N
        self.N_tot = (self.rho / M_bar) * scipy.constants.N_A * 1e6   # atoms/m^3

        # Store number density fraction for each atom in a dict.
        self.Nd = {a: a.f * self.N_tot for a in self.atoms} # atoms/m^3

    # FUNCTION: Multiply the stored total stopping power by some factor
    def conv_Se(self, factor):
        self.Se *= factor
        self.dEdx_interp = interpolate.PchipInterpolator(self.Es, self.Se, extrapolate = True)

    # FUNCTION: Multiply the stored energy by some factor
    def conv_E(self, factor):
        self.Es *= factor
        self.dEdx_interp = interpolate.PchipInterpolator(self.Es, self.Se, extrapolate = True)

    # FUNCTION: Compute the nuclear reactions (p, x) cross-sections weighted based on number density
    def compute_sig(self, E):
        return sum(self.Nd[a] * a.get_sig(E) for a in self.atoms) # 1/m

    # FUNCTION: Change the cross-sectional area of the medium ((y, z) plane)
    def set_A(self, A):
        self.A = A
        self.r = 2*self.A/self.P

    # FUNCTION: Change the cross-sectional perimeter of the medium ((y, z) plane)
    def set_P(self, P):
        self.P = P
        self.r = 2*self.A/self.P

    # FUNCTION: Return the total stopping power extrapolated from the table in eV/m*atom
    def get_Se_ev_m_atom(self, E):
        return self.dEdx_interp(E)  # eV/m*atom

    # FUNCTION: Return the total stopping power extrapolated from the table in eV/m
    def get_Se_ev_m(self, E):
        return self.dEdx_interp(E) * self.N_tot

    # FUNCTION: Compute the intensity gradient dI/dx (intensity attenuation only exists if (p,x) reactions occur)
    def get_dIdx(self, E, I):
        return - self.compute_sig(E) * I # 1/(m*s)

    # PRIVATE FUNCTION: Compute forward and lateral scattering from nuclear reactions
    def _Ed_fwd(self,
                cj,             # x coordinate of cell center
                cell_width,
                x_fwd,          # x position after cell center
                E,              # Beam energy at cj
                I               # Beam intensity at cj
                ):
        if x_fwd > self.Lx:
            x_fwd = self.Lx

        pre_fct = 1-np.exp(-self.r/self.LBD) # lateral pre-factor
        dIdx = self.get_dIdx(E, I)

        def f(x, dx, xfwd): # xfwd is a dummy variable
            fct1 = np.exp(-(xfwd-x)/self.LBD) - np.exp(-(xfwd-x+dx)/self.LBD)
            return self.RFWD * E * -dIdx * fct1

        return pre_fct * f(cj, cell_width, x_fwd)

    # PRIVATE FUNCTION: Compute backward and lateral scattering from nuclear reactions
    def _Ed_bwd(self,
                cj,             # x coordinate of cell center
                cell_width,
                x_bwd,          # x position before cell center
                E,              # Beam energy at cj
                I               # Beam intensity at cj
                ):
        if(x_bwd < self.x0):
            x_bwd = self.x0

        pre_fct = 1-np.exp(-self.r/self.LBD)    # lateral pre-factor
        dIdx = self.get_dIdx(E, I)

        def f(x, dx, xbwd): # xbwd is a dummy variable
            fct1 = np.exp(-(x-xbwd)/self.LBD) - np.exp(-(x-xbwd + dx)/self.LBD)
            return (1-self.RFWD) * E * -dIdx * fct1

        return pre_fct * f(cj, cell_width, x_bwd)

    # FUNCTION: Set the empirical fitting parameter lambda
    def set_LBD(self, E, conv_MeV=1e-6):
        self.LBD = 11.4 * ((E * conv_MeV) ** (1 / 3)) * 1e-3 # paper fit for optimal lambda, only stored once

    # FUNCTION: Return total dEdx (just the negative of total stopping power)
    def get_dEdx(self, E):
        return -self.get_Se_ev_m(E)  # eV/(m·s)



    # IGNORE THIS
    def _Ed_fwd_test(self, cj, x_fwd, dx, E, I):
        pre_fct = 1 - np.exp(-self.r / self.LBD)  # lateral pre-factor
        dIdx = self.get_dIdx(E, I)  # get intensity gradient

        def f(xfwd):
            fct1 = np.exp(-(xfwd - cj) / self.LBD)
            return self.RFWD / self.LBD * E * -dIdx * fct1

        return pre_fct * scipy.integrate.quad(f, cj, x_fwd + dx)[0]

    # IGNORE THIS
    def _Ed_bwd_test(self, cj, x_bwd, dx, E, I):
        pre_fct = 1 - np.exp(-self.r / self.LBD)  # lateral pre-factor
        dIdx = self.get_dIdx(E, I)  # get intensity gradient

        def f(xbwd):
            fct1 = np.exp(-(cj - xbwd) / self.LBD)
            return (1 - self.RFWD) / self.LBD * E * -dIdx * fct1

        return pre_fct * scipy.integrate.quad(f, x_bwd - dx, cj)[0]
