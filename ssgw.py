import numpy as np
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


def coth(x: np.ndarray) -> np.ndarray:
    res = np.zeros_like(x)
    small = np.abs(x) < 1e-8
    res[small] = 1 / x[small]
    ok = ~small
    exp_x = np.exp(2 * x[ok])
    res[ok] = (exp_x + 1) / (exp_x - 1)
    large = np.abs(x) > 100
    res[large] = np.sign(x[np.abs(x) > 100])
    return res


class SSGW:
    """
    SSGW: Steady Surface Gravity Waves.
    Computation of irrotational 2D periodic surface pure gravity waves
    of arbitrary length in arbitrary depth.

    Attributes:
        zs: np.ndarray
            Complex abscissas at the free surface
        ws: np.ndarray
            Complex velocity at the free surface
        depth: float
            Water depth
        wavenumber: float
            Wave number k
        wave_height: float
            Total wave height
        ce: float
            Celerity c_e
        cs: float
            Celerity c_s
        bernoulli: float
            Bernoulli constant
        crest_height: float
            Height of wave crest
        trough_height: float
            Height of wave trough
        impulse: float
            Wave impulse
        potential_energy: float
            Potential energy
        kinetic_energy: float
            Kinetic energy
        radiation_stress: float
            Radiation stress
        momentum_flux: float
            Momentum flux
        energy_flux: float
            Energy flux
        group_velocity: float
            Group velocity
    """

    def __init__(self, kd: float, kH2: float, N: int = 2048, tol: float = 1e-14):
        """
        Parameters:
            kd: float
                Relative depth (wavenumber "k" times mean water depth "d")
            kH2: float
                Steepness (half the total wave height "H" times the wavenumber "k")
            N: int, optional
                Number of positive Fourier modes (default: 2048)
            tol: float, optional
                Tolerance (default: 1e-14)
        """
        # Input validation
        if kd < 0 or np.imag(kd) != 0 or kH2 < 0 or np.imag(kH2) != 0:
            raise ValueError("Input parameters kd and kH2 must be real and positive.")

        # Determine depth and choose parameters
        if 1 - np.tanh(kd) < tol:  # Deep water case
            d = np.inf  # Depth
            k = 1  # Wavenumber
            g = 1  # Acceleration due to gravity
        else:  # Finite depth case
            d = 1  # Depth
            k = kd / d  # Wavenumber
            g = 1  # Acceleration due to gravity

        H = 2 * kH2 / k  # Total wave height
        L = np.pi / k  # Half-length of the computational domain
        dal = L / N  # Delta alpha
        dk = np.pi / L  # Delta k

        # Vectors
        va = np.arange(2 * N) * dal  # Vector of abscissas in the conformal space
        vk = (
            np.concatenate([np.arange(N), np.arange(-N, 0)]) * dk
        )  # Vector of wavenumbers

        # Initial guess for the solution
        Ups = H / 2 * (1 + np.cos(k * va))  # Airy solution for Upsilon
        sig = 1  # Parameter sigma

        # Commence Petviashvili's iterations
        err = np.inf
        while err > tol:
            # Compute sigma and delta
            mUps = np.mean(Ups)
            Ys = Ups - mUps

            if d == np.inf:  # Deep water
                sig = 1
                CYs = np.real(
                    np.fft.ifft(np.abs(vk) * np.fft.fft(Ys) / (2 * N)) * (2 * N)
                )
                mys = -np.sum(Ys * CYs) / N / 2
            else:  # Finite depth
                C_hat = vk * coth(sig * d * vk)
                C_hat[0] = 1 / (sig * d)
                S2_hat = (vk * 1 / np.sinh(sig * d * vk)) ** 2
                S2_hat[0] = 1 / (sig * d) ** 2
                Ys_hat = np.fft.fft(Ys) / (2 * N)

                E = (
                    np.mean(Ys * np.real(np.fft.ifft(C_hat * Ys_hat) * (2 * N)))
                    + (sig - 1) * d
                )
                dE = d - d * np.mean(
                    Ys * np.real(np.fft.ifft(S2_hat * Ys_hat) * (2 * N))
                )
                sig = sig - E / dE
                mys = (sig - 1) * d

            del_ = mys - mUps
            C_hat = vk * coth(sig * d * vk)
            C_hat[0] = 1 / (sig * d)

            # Compute Bernoulli constant B
            Ups2 = Ups * Ups
            mUps2 = np.mean(Ups2)
            CUps = np.real(np.fft.ifft(C_hat * np.fft.fft(Ups) / (2 * N)) * (2 * N))
            CUps2 = np.real(np.fft.ifft(C_hat * np.fft.fft(Ups2) / (2 * N)) * (2 * N))
            DCU = CUps[N] - CUps[0]
            DCU2 = CUps2[N] - CUps2[0]
            Bg = (
                2 * del_
                - H / sig * (1 + del_ / d + sig * CUps[0]) / DCU
                + DCU2 / DCU / 2
            )

            # Define linear operators in Fourier space
            Cinf_hat = np.abs(vk)
            Cinf_hat[0] = 0
            CIC_hat = np.tanh(sig * d * np.abs(vk))
            if d == np.inf:
                CIC_hat[0] = 1

            L_hat = (Bg - 2 * del_) * Cinf_hat - ((1 + del_ / d) / sig) * CIC_hat
            IL_hat = 1 / L_hat
            IL_hat[0] = 1

            # Petviashvili's iteration
            Ups_hat = np.fft.fft(Ups) / (2 * N)
            CUps_hat = C_hat * Ups_hat
            LUps = np.real(np.fft.ifft(L_hat * Ups_hat) * (2 * N))
            Ups2_hat = np.fft.fft(Ups * Ups) / (2 * N)
            NUps_hat = (
                CIC_hat
                * np.fft.fft(Ups * np.real(np.fft.ifft(CUps_hat) * (2 * N)))
                / (2 * N)
            )
            NUps_hat = NUps_hat + Cinf_hat * Ups2_hat / 2
            NUps = np.real(np.fft.ifft(NUps_hat) * (2 * N))
            S = np.sum(Ups * LUps) / np.sum(Ups * NUps)
            U = S * S * np.real(np.fft.ifft(NUps_hat * IL_hat) * (2 * N))
            U = H * (U - U[N]) / (U[0] - U[N])

            # Update values
            err = np.max(np.abs(U - Ups))
            Ups = U

        # Post processing
        IH_hat = -1j * coth(sig * d * vk)
        IH_hat[0] = 0
        Ys = Ups - np.mean(Ups)
        Ys_hat = np.fft.fft(Ys) / (2 * N)
        CYs = np.real(np.fft.ifft(C_hat * Ys_hat) * (2 * N))
        Xs = np.real(np.fft.ifft(IH_hat * Ys_hat) * (2 * N))
        mys = -np.sum(Ys * CYs) / N / 2
        Zs = Xs + 1j * Ys
        dZs = np.fft.ifft(1j * vk * np.fft.fft(Zs) / (2 * N)) * (2 * N)
        zs = va + 1j * mys + Zs
        dzs = 1 + dZs
        B = g * Bg
        ce = np.sum((1 + CYs) / np.abs(dzs) ** 2) / 2 / N
        ce = np.sqrt(B / ce)
        cs = sig * ce
        ws = -ce / dzs
        a = np.max(np.imag(zs))
        b = -np.min(np.imag(zs))

        if d == np.inf:
            Bce2d = 0
            IC = 1 / np.abs(vk)
            IC[0] = 0
        else:
            Bce2d = (B - ce**2) * d
            IC = np.tanh(vk * sig * d) / vk
            IC[0] = sig * d  # Inverse C-operator

        ydx = np.real(dzs) * np.imag(zs)
        intI = -ce * mys  # Impulse
        intV = np.mean(ydx * np.imag(zs)) * g / 2  # Potential energy
        intK = intI * ce / 2  # Kinetic energy
        intSxx = 2 * ce * intI - 2 * intV + Bce2d  # Radiation stress
        intS = intSxx - intV + g * d**2 / 2  # Momentum flux
        intF = (
            Bce2d * ce / 2 + (B + ce**2) * intI / 2 + (intK - 2 * intV) * ce
        )  # Energy flux
        cg = intF / (intK + intV)  # Group velocity

        # Assign attributes
        self.zs = zs
        self.ws = ws
        self.depth = d
        self.wavenumber = k
        self.wave_height = H
        self.ce = ce
        self.cs = cs
        self.bernoulli = B
        self.crest_height = a
        self.trough_height = b
        self.impulse = intI
        self.potential_energy = intV
        self.kinetic_energy = intK
        self.radiation_stress = intSxx
        self.momentum_flux = intS
        self.energy_flux = intF
        self.group_velocity = cg
