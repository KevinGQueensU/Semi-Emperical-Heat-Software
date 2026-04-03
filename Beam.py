import numpy as np

# BEAM CLASS used to calculate energy gradient in heat equation
class Beam:
    # Beam can be Rectangular, Circular, Gaussian, Gaussian Annular, or Gaussian + Gaussian Annular
    TYPES = ('Rectangular', 'Circular', 'Gaussian', 'Annular', 'Both')

    # CONSTRUCTOR: Defines the particle beam properties and shape, with additional optional divergence parameters
    # Beam is taken to be propagating along the x-axis direction
    def __init__(self,
                 E_0: float,                    # Initial instantaneous energy [eV]
                 I_0: float,                    # Initial intensity [1/s]
                 Z: int,                        # Beam particle atomic number
                 dim: int,                      # 2D or 3D beam
                 A: int = 1,                    # Beam particle mass number
                 L: float = 0, W=0,            # Rectangular half-extents
                 r: float = 0,                  # Circular beam radius
                 frac_gauss: float = 1,         # Fraction that is Gaussian (Both only)
                 sig_ga_y0: float = 0, sig_ga_z0=0,   # Gaussian STDs
                 sig_an_y0: float = 0, sig_an_z0=0,   # Annular STDs
                 mu_y0: float = 0, mu_z0=0,    # Annular means
                 type: str = None,              # Override auto-detection
                 ):

        if not (dim == 2 or dim == 3):
            raise ValueError("Beam can only be 2 or 3 dimensional")

        # Detect which parameter groups are present
        has_rect = (L > 0) or (W > 0)
        has_circ = (r > 0)
        has_gauss = (sig_ga_y0 > 0) or (sig_ga_z0 > 0)
        has_ann = (sig_an_y0 > 0) or (sig_an_z0 > 0) or (mu_y0 != 0) or (mu_z0 != 0)
        groups = [has_rect, has_circ, has_gauss, has_ann]

        # Auto-detect type if not given
        if type is None:
            if sum([has_rect, has_circ]) > 1:
                raise ValueError(
                    "Ambiguous beam parameters: cannot mix Rectangular and Circular"
                )
            if has_rect and not has_gauss and not has_ann and not has_circ:
                type = 'Rectangular'
            elif has_circ and not has_gauss and not has_ann and not has_rect:
                type = 'Circular'
            elif has_gauss and has_ann and not has_rect and not has_circ:
                type = 'Both'
            elif has_gauss and not has_ann and not has_rect and not has_circ:
                type = 'Gaussian'
            elif has_ann and not has_gauss and not has_rect and not has_circ:
                type = 'Annular'
            elif not any(groups):
                raise ValueError("No beam shape parameters provided")
            else:
                raise ValueError(
                    "Ambiguous beam parameters: please define type explicitly"
                )
        elif type not in self.TYPES:
            raise ValueError(f"Invalid type '{type}': must be one of {self.TYPES}")

        # Validate parameters for the resolved type
        if type == 'Rectangular':
            if has_gauss or has_ann or has_circ:
                raise ValueError("Rectangular beam should not have Circular/Gaussian/Annular parameters")
            if L <= 0:
                raise ValueError("Rectangular beam requires L > 0")
            if dim == 3 and W <= 0:
                raise ValueError("3D Rectangular beam requires W > 0")

        if type == 'Circular':
            if has_gauss or has_ann or has_rect:
                raise ValueError("Circular beam should not have Rectangular/Gaussian/Annular parameters")
            if r <= 0:
                raise ValueError("Circular beam requires r > 0")
            if dim == 2:
                raise ValueError("Circular beam is only defined for 3D")

        if type in ('Gaussian', 'Both'):
            if sig_ga_y0 <= 0:
                raise ValueError(f"{type} beam requires sig_ga_y0 > 0")
            if dim == 3 and sig_ga_z0 <= 0:
                raise ValueError(f"3D {type} beam requires sig_ga_z0 > 0")

        if type in ('Annular', 'Both'):
            if sig_an_y0 <= 0:
                raise ValueError(f"{type} beam requires sig_an_y0 > 0")
            if mu_y0 == 0:
                raise ValueError(f"{type} beam requires mu_y0 != 0")
            if dim == 3:
                if sig_an_z0 <= 0:
                    raise ValueError(f"3D {type} beam requires sig_an_z0 > 0")
                if mu_z0 == 0:
                    raise ValueError(f"3D {type} beam requires mu_z0 != 0")

        if type == 'Both' and (frac_gauss <= 0 or frac_gauss >= 1):
            raise ValueError("'Both' beam requires 0 < frac_gauss < 1")

        # Mean excitation energy
        if Z < 1:
            raise ValueError("Z must be greater than 0")
        elif Z == 1:
            self.Vm = 19.0
        elif Z <= 13:
            self.Vm = 11.2 + 11.7 * Z
        else:
            self.Vm = 52.8 + 8.71 * Z

        self.type = type
        self.E_0 = E_0
        self.I_0 = I_0
        self.P_0 = E_0 * I_0
        self.A = A
        self.Z = Z
        self.dim = dim
        self.L = L
        self.W = W
        self.r = r
        self.frac_gauss = frac_gauss
        self.frac_ann = 1 - frac_gauss
        self.sig_ga_y0 = sig_ga_y0
        self.sig_ga_z0 = sig_ga_z0
        self.sig_an_y0 = sig_an_y0
        self.sig_an_z0 = sig_an_z0
        self.mu_y0 = mu_y0
        self.mu_z0 = mu_z0

    # FUNCTION: Returns power density [W/m^2] at a specified position (x, y, z) [m] with (y, z) beam divergence angles (alpha, beta)
    # EQUATIONS FROM https://doi.org/10.1016/j.nimb.2019.09.016
    def PD(self,
           x : float, y, z,  # Coordinates in [m]
           alpha: float,     # Beam divergence angle along y direction
           beta: float       # Beam divergence angle along z direction
           ) -> float:

        if self.type == 'Rectangular':
            masky = np.asarray((y <= (self.L / 2)) & (y >= (-self.L / 2)), dtype=np.float64)
            if self.dim == 2:
                return self.P_0 / self.L * masky
            else:
                maskz = np.asarray((z <= self.W / 2) & (z >= (-self.W / 2)), dtype=np.float64)
                return self.P_0 / (self.L * self.W) * masky * maskz

        if self.type == 'Circular':
            r2 = y ** 2 + z ** 2
            mask = np.asarray(r2 <= self.r ** 2, dtype=np.float64)
            return self.P_0 / (np.pi * self.r ** 2) * mask

        # Initializing Gaussian and Annular power densities to 0
        PD_gauss = 0
        PD_ann = 0

        # Calculate new std's based on x position and divergence
        sig_ga_y = self.sig_ga_y0 + x * np.tan(alpha)
        sig_ga_z = self.sig_ga_z0 + x * np.tan(beta)

        # Calculate Gaussian power density
        if self.type == 'Gaussian' or self.type == 'Both':
            fy = (1/(np.sqrt(2*np.pi)*sig_ga_y))*np.exp(-y ** 2 / (2 * sig_ga_y**2)) # y spread
            fz = (1/(np.sqrt(2*np.pi)*sig_ga_z))*np.exp(-z ** 2 / (2 * sig_ga_z ** 2)) # z spread

            PD_gauss = self.P_0 * fy if self.dim == 2 else self.P_0 * fy * fz

        # Calculate Annular Gaussian power density
        if self.type == 'Annular' or self.type == 'Both':
            # Calculate new lateral means and std's at position
            mu_y = self.mu_y0 * (sig_ga_y/self.sig_ga_y0)
            mu_z = self.mu_z0 * (sig_ga_z/self.sig_ga_z0)
            sig_an_y = self.sig_an_y0*(mu_y/self.mu_y0)
            sig_an_z = self.sig_an_z0*(mu_z/self.mu_z0)

            fp_denom = 1 + np.sqrt(np.pi/2 * (mu_y*mu_z/(sig_an_y*sig_an_z))) # denom term
            fp = self.P_0 / (2 * np.pi * fp_denom) # full prefactor
            fy = (1/sig_an_y)*np.exp(-y**2/(2*sig_an_y**2)) # y spread
            fz = (1/sig_an_z)*np.exp(-z**2/(2 * sig_an_z ** 2)) # z spread

            PD_ann = fp * fy if self.dim == 2 else fp * fy * fz

        # Return sum of power densities weighted by fraction
        return self.frac_ann * PD_ann + self.frac_gauss * PD_gauss