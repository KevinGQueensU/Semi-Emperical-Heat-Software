import scipy as sp
import numpy as np

## BEAM CLASS used to calculate energy gradient in heat equation
class Beam:
    TYPES = ('Gaussian', 'Annular', 'Both') # Types of beams

    #CONSTRUCTOR: Takes initial instantaneous energy E_0 [eV], intial intensity I_0 [1/s], beam particle atomic number Z.
    # Optional parameters for setting Gaussian and Annular Gaussian beam parameters
    def __init__(self, E_0, I_0, Z, A = 1, f_gauss = 1, sig_ga_y0=0, sig_ga_z0=0, sig_an_y0=0, sig_an_z0=0, mu_y0=0, mu_z0=0, type ='Gaussian'):
        if type not in self.TYPES:
            raise ValueError(f"Invalid Type: Must be Gaussian or Annular or Both")
        self.type = type
        self.E_0 = E_0 # eV
        self.I_0 = I_0 # s^-1
        self.P_0 = E_0 * I_0 # initial power, eV/s

        self.A = A
        self.Z = Z
        self.f_gauss = f_gauss # f_gauss + f_ann SHOULD ALWAYS ADD UP TO 1
        self.f_ann = 1 - f_gauss

        if 2 <= Z and Z <= 13:
            self.Vm = 11.2 + 11.7*Z #eV
        elif Z > 13:
            self.Vm = 52.8 + 8.71*Z #eV
        elif Z == 1:
            self.Vm = 19.0 #eV
        else:
            raise ValueError(f"Z must be greater than 0")

        self.sig_ga_y0 = sig_ga_y0
        self.sig_ga_z0 = sig_ga_z0
        self.sig_an_y0 = sig_an_y0
        self.sig_an_z0 = sig_an_z0
        self.mu_y0 = mu_y0
        self.mu_z0 = mu_z0

    # Returns power density at a specificed (x, y, z) [m] with (y, z) beam divergence angles (alpha, beta)
    def PD(self, x, y, z, alpha, beta):
        PD_gauss = 0
        PD_ann = 0

        # Calculate new std's
        sig_ga_y = self.sig_ga_y0 + x * np.tan(alpha)
        sig_ga_z = self.sig_ga_z0 + x * np.tan(beta)

        # If beam type Gaussian
        if self.type == 'Gaussian' or self.type == 'Both':
            fp = self.f_gauss * self.P_0 /(2*np.pi*sig_ga_y*sig_ga_z) # prefactor
            fy = np.exp(-y**2/(2*sig_ga_y**2)) # y spread
            fz = np.exp(-z ** 2 / (2 * sig_ga_z ** 2)) # z spread
            PD_gauss = fp * fy * fz

        # If beam type Annular
        if self.type == 'Annular' or self.type == 'Both':
            # Calculate new std's
            mu_y = self.mu_y0 * (sig_ga_y/self.sig_ga_y0)
            mu_z = self.mu_z0 * (sig_ga_z/self.sig_ga_z0)
            sig_an_y = self.sig_an_y0*(mu_y/self.mu_y0)
            sig_an_z = self.sig_an_z0*(mu_z/self.mu_z0)

            fp_denom = 1 + np.sqrt(np.pi/2 * (mu_y*mu_z/(sig_an_y*sig_an_z))) # denom term
            fp = self.f_ann * self.P_0 / (2 * np.pi * sig_an_y * sig_an_z * fp_denom) # full prefactor
            fy = np.exp(-y**2/(2*sig_an_y**2)) # y spread
            fz = np.exp(-z**2/(2 * sig_an_z ** 2)) # z spread
            PD_ann = fp * fy * fz

        return PD_ann + PD_gauss


