import numpy as np

# BEAM CLASS used to calculate energy gradient in heat equation
class Beam:
    # Beam can have a Gaussian, Gaussian Annular, or a combination shape
    TYPES = ('Gaussian', 'Annular', 'Both')

    # CONSTRUCTOR: Defines the particle beam properties and shape, with additional optional divergence parameters
    # Beam is taken to be propagating along the x-axis direction
    def __init__(self,
                 E_0: float,  # Initial Instantaneous Energy [eV]
                 I_0: float,  # Initial Intenisty [1/s]
                 Z: int,      # Beam particle atomic number
                 A: int = 1,  # OPTIONAL Beam particle mass number (for isotopes)
                 frac_gauss: float = 1,    # OPTIONAL fraction of beam that is Gaussian shaped (f_annular = 1 - f_gauss)
                 sig_ga_y0: float = 0,  sig_ga_z0: float = 0, # OPTIONAL Initial Gaussian beam STD along y and z directions
                 sig_an_y0: float = 0, sig_an_z0: float = 0,  # OPTIONAL Initial Gaussian Annular beam STD along y and z directions
                 mu_y0: float = 0,  mu_z0: float = 0,         # OPTIONAL Initial Gaussian Annular beam mean along y and z directions
                 type: str = 'Gaussian'    # OPTIONAL Type of beam, default is Gaussian
                 ):
        # TEST: Make sure the defined beam type is valid
        if type not in self.TYPES:
            raise ValueError(f"Invalid Type: Must be Gaussian or Annular or Both")

        # Mean excitation energy of medium depending on the beam's particle type
        if 2 <= Z and Z <= 13:
            self.Vm = 11.2 + 11.7*Z # eV
        elif Z > 13:
            self.Vm = 52.8 + 8.71*Z # eV
        elif Z == 1:
            self.Vm = 19.0 # eV
        else:
            raise ValueError(f"Z must be greater than 0") # invalid Z given

        # Default parameters
        self.type = type
        self.E_0 = E_0 # eV
        self.I_0 = I_0 # s^-1
        self.P_0 = E_0 * I_0 # initial power, eV/s
        self.A = A
        self.Z = Z

        self.frac_gauss = frac_gauss
        self.frac_ann = 1 - frac_gauss # f_gauss + f_ann SHOULD ALWAYS ADD UP TO 1
        self.sig_ga_y0 = sig_ga_y0 ; self.sig_ga_z0 = sig_ga_z0
        self.sig_an_y0 = sig_an_y0 ; self.sig_an_z0 = sig_an_z0
        self.mu_y0 = mu_y0 ; self.mu_z0 = mu_z0

    # FUNCTION: Returns power density [W/m^2] at a specified position (x, y, z) [m] with (y, z) beam divergence angles (alpha, beta)
    # EQUATIONS FROM https://doi.org/10.1016/j.nimb.2019.09.016
    def PD(self,
           x : float, y : float, z : float, # Coordinates in [m]
           alpha: float,    # Beam divergence angle along y direction
           beta: float      # Beam divergence angle along z direction
    ):
        # Initializing Gaussian and Annular power densities to 0
        PD_gauss = 0
        PD_ann = 0

        # Calculate new std's based on x position and divergence
        sig_ga_y = self.sig_ga_y0 + x * np.tan(alpha)
        sig_ga_z = self.sig_ga_z0 + x * np.tan(beta)

        # Calculate Gaussian power density
        if self.type == 'Gaussian' or self.type == 'Both':
            fp = self.P_0 /(2*np.pi*sig_ga_y*sig_ga_z) # prefactor
            fy = np.exp(-y**2/(2*sig_ga_y**2)) # y spread
            fz = np.exp(-z ** 2 / (2 * sig_ga_z ** 2)) # z spread

            PD_gauss = fp * fy * fz

        # Calculate Annular Gaussian power density
        if self.type == 'Annular' or self.type == 'Both':
            # Calculate new lateral means and std's at position
            mu_y = self.mu_y0 * (sig_ga_y/self.sig_ga_y0)
            mu_z = self.mu_z0 * (sig_ga_z/self.sig_ga_z0)
            sig_an_y = self.sig_an_y0*(mu_y/self.mu_y0)
            sig_an_z = self.sig_an_z0*(mu_z/self.mu_z0)

            fp_denom = 1 + np.sqrt(np.pi/2 * (mu_y*mu_z/(sig_an_y*sig_an_z))) # denom term
            fp = self.P_0 / (2 * np.pi * sig_an_y * sig_an_z * fp_denom) # full prefactor
            fy = np.exp(-y**2/(2*sig_an_y**2)) # y spread
            fz = np.exp(-z**2/(2 * sig_an_z ** 2)) # z spread

            PD_ann = fp * fy * fz

        # Return sum of power densities weighted by fraction
        return self.frac_ann * PD_ann + self.frac_gauss * PD_gauss

