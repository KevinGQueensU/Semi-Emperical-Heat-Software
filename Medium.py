import numpy as np
import scipy.constants
import numpy as np
import scipy.integrate
import scipy.interpolate as interpolate

# Sort 2 arrays based on input array x values
def _sorted_unique_xy(x, y, agg="last"):
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    # keep only finite pairs
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]

    # sort by x
    idx = np.argsort(x, kind="mergesort")  # stable
    x = x[idx]; y = y[idx]

    # collapse duplicates
    ux, start_idx, counts = np.unique(x, return_index=True, return_counts=True)
    if agg == "first":
        uy = y[start_idx]
    elif agg == "last":
        # last index of each run = start + count - 1
        last_idx = start_idx + counts - 1
        uy = y[last_idx]
    else:  # mean
        uy = np.add.reduceat(y, start_idx) / counts

    return ux, uy\
# Read a file and return the first two cols
def read_file(filename):
    try:
        file = open(filename, 'r')
    except:
        print('File not found')
    col1 = np.empty(0)
    col2 = np.empty(0)
    for line in file:
        # Strip leading/trailing whitespace and check if it's a comment
        stripped_line = line.strip()
        if not stripped_line or stripped_line.startswith('#'):
            continue  # Skip empty lines and comment lines
        # Split the line by tab delimiter
        data_elements = stripped_line.split()
        col1 = np.append(col1, float(data_elements[0]))
        col2 = np.append(col2, float(data_elements[1]))

    return col1, col2

# ATOM CLASS: Used to describe atoms that comprise a medium of choice
class atom:
    # Constructor that takes in name, atomic number Z, atomic mass A (g/mol), atomic fraction f,
    # filepath with cross sections (p,x)
    def __init__(self, name, Z, A, f, sig_file, E_conv = 1, sig_conv = 1e-28):
        self.name = name
        self.Z = Z # > 0
        self.A = A # g/mol
        self.f = f # 0 < f < 1
        self.Es, self.sig_bs = read_file(sig_file)

        self.Es = np.r_[0, self.Es] * E_conv  # -> eV
        self.sig_bs =  np.r_[0, self.sig_bs] * sig_conv # -> m^2
        self.Es, self.sig_bs = _sorted_unique_xy(self.Es, self.sig_bs)

        self.sig_interp = interpolate.PchipInterpolator(self.Es, self.sig_bs, extrapolate=True)

    # Multiply cross sections by some factor
    def conv_sig(self, factor):
        self.sig_bs *= factor

    #Multiply energies by some factor
    def conv_E(self, factor):
        self.Es *= factor

    # Read in a new cross section file
    def set_sig(self, file_name):
        self.Es, self.sig_bs = read_file(file_name)

    # Input an energy, return the corresponding (p,x) extrapolated from the file
    def get_sig(self, E):
        return max(self.sig_interp(E), 0.0) # m^2


## MEDIUM CLASS used in heat equation for calculating energy gradient
class Medium:
    RFWD = 0.6 # optimum forward back scattering fraction from paper

    ## CONSTRUCTOR: Takes in n (electron density, 1/m^3), rho (bulk density, g/cm^3), atoms,
    # L (length of medium, m), A (cross sectional area along x, m^2),
    # P (perimeter of cross sectional area along x, m),
    # dEdx_filename (filename that has stopping power table [E, Se]
    def __init__(self, n, rho, atoms, L, A, P, dEdX_filename, beam, x0 = 0, Se_conv = 9.746268365e-18, E_conv = 1e6):
        if isinstance(atoms, (list, tuple, np.ndarray)):
            self.atoms = np.array(list(atoms), dtype=object).reshape(-1)
        else:
            # if only a single atom is passed
            self.atoms = np.array([atoms], dtype=object)
        self.n = n # electron density 1/m^3
        self.rho = rho # bulk density g/cm^3
        self.A = A # area m^2
        self.P = P # perimeter m
        self.L = L # length m
        self.x0 = x0 # start of medium, m
        self.r = 2*A/P # radius m
        self.beam = beam
        self.filename = dEdX_filename
        self.Es, self.Se = read_file(dEdX_filename)
        self.Es = np.r_[0, self.Es] * E_conv  # -> eV
        self.Se = np.r_[0, self.Se] * Se_conv # eV·m^2/atom
        self.Es, self.Se = _sorted_unique_xy(self.Es, self.Se) # sort array for interp
        self.LBD = 0 # fitting parameter, initially set to 0
        self._compute_number_densities()
        self.dEdx_interp = interpolate.PchipInterpolator(self.Es, self.Se, extrapolate = True)

    # PRIVATE: Compute Nd based on atom atomic fraction and mass, store as nict
    def _compute_number_densities(self):
        # average molar mass per atom (g/mol), atomic fraction * atomic mass [g/mol] summed
        M_bar = sum(a.f * a.A for a in self.atoms)

        # Calculate total number density
        self.N_tot = (self.rho / M_bar) * scipy.constants.N_A * 1e6   # atoms/m^3

        # Store number density for each atom in a dict.
        self.Nd = {a: a.f * self.N_tot for a in self.atoms} # atoms/m^3 for each atom type
        print(self.Nd)

    # Multiply the stopping power stored for dEdx by some factor
    def conv_Se(self, factor):
        self.Se *= factor

    # Multiply the energy stored for dEdx by some factor
    def conv_E(self, factor):
        self.Es *= factor

    # Compute all of the (p, x) cross sections (reactions not covered in dE/dx)
    def compute_sig(self, E):
        # sum(Nd*sig(p,x))
        return sum(self.Nd[a] * a.get_sig(E) for a in self.atoms) # 1/m

    # Change the cross section area along x
    def set_A(self, A):
        self.A = A
        self.r = 2*self.A/self.P

    # Change the cross section perimeter along x
    def set_P(self, P):
        self.P = P
        self.r = 2*self.A/self.P

    # Return the stopping power interpolated from the table in eV/m*atom
    def get_Se(self, E):
        return self.dEdx_interp(E)  # eV/m*atom

    # Return the stopping power interpolated from the table in eV/m
    def get_Se_ev_m(self, E):
        return self.dEdx_interp(E) * self.N_tot
    # Compute the intensity gradient dI/dx
    def get_dIdx(self, E, I, test = False):
        if test:
            return self.compute_sig(E)
        return - self.compute_sig(E) * I # 1/(m*s)

    def dEdx_Bethe(self, E, beam):
        E_rest = beam.A * 931.494 * 1e6  # rest energy of a proton
        gamma = 1 + E/E_rest
        beta_r = np.sqrt(1-1/gamma**2)

        k_0 = 1/(4*np.pi*scipy.constants.epsilon_0)
        prefct = (4*np.pi*k_0**2*beam.Z**2 * scipy.constants.e**4 * self.n)/(E_rest*beta_r**2)
        ln_fct = 2*E_rest*beta_r**2/(beam.Vm*(1-beta_r**2)) - beta_r**2
        return prefct * (np.log(ln_fct) - beta_r**2)

    # PRIVATE: Compute forward and lateral scattering from secondary cascades
    def _Ed_fwd(self, cj, cell_width, x_fwd, E, I):
        if x_fwd > self.L:
            x_fwd = self.L
        pre_fct = 1-np.exp(-self.r/self.LBD) # lateral pre-factor
        dIdx = self.get_dIdx(E, I) # get intensity gradient

        def f(x, dx, xfwd):
            fct1 = np.exp(-(xfwd-x)/self.LBD) - np.exp(-(xfwd-x+dx)/self.LBD)
            return self.RFWD * E * -dIdx * fct1

        return pre_fct * f(cj, cell_width, x_fwd)

    # PRIVATE: Compute backward and lateral scattering from secondary cascades
    def _Ed_bwd(self, cj, cell_width, x_bwd, E, I):
        if(x_bwd < self.x0):
            x_bwd = self.x0
        pre_fct = 1-np.exp(-self.r/self.LBD) # lateral pre-factor
        dIdx = self.get_dIdx(E, I) # get intensity gradient

        def f(x, dx, xbwd):
            fct1 = np.exp(-(x-xbwd)/self.LBD) - np.exp(-(x-xbwd + dx)/self.LBD)
            return (1-self.RFWD) * E * -dIdx * fct1

        return pre_fct * f(cj, cell_width, x_bwd)

    def _Ed_fwd_test(self, cj, x_fwd, dx, E, I):
        pre_fct = 1-np.exp(-self.r/self.LBD) # lateral pre-factor
        dIdx = self.get_dIdx(E, I) # get intensity gradient

        def f(xfwd):
            fct1 = np.exp(-(xfwd-cj)/self.LBD)
            return self.RFWD/self.LBD  * E * -dIdx * fct1

        return pre_fct * scipy.integrate.quad(f, cj, x_fwd + dx)[0]

    def _Ed_bwd_test(self, cj, x_bwd, dx, E, I):
        pre_fct = 1-np.exp(-self.r/self.LBD) # lateral pre-factor
        dIdx = self.get_dIdx(E, I) # get intensity gradient

        def f(xbwd):
            fct1 = np.exp(-(cj-xbwd)/self.LBD)
            return (1-self.RFWD)/self.LBD * E * -dIdx * fct1

        return pre_fct * scipy.integrate.quad(f, x_bwd - dx, cj)[0]

    # Compute the energy gradient at a specified x (m), given a step size dx (m),
    # instantaneous particle energy E (eV), and intensity I (1/s)
    def set_LBD(self, E, conv_MeV=1e-6):
        self.LBD = 11.4 * ((E * conv_MeV) ** (1 / 3)) * 1e-3 # paper fit for optimal lambda, only stored once

    def get_dEdx(self, E):
        return -self.get_Se_ev_m(E)  # eV/(m·s)




#%%
if True:
    filename = 'C://Users//k_gao//Desktop//PaleoBSMwithTRIM-main//minerals_def//Ni//Cross_Sections//Ni_px.txt'
