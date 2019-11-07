"""Module for beam filters which cause light attenuation. Filters are assumed
to be homogeneous, thus no phase change effects are introduced when
a wavefield passes through them.
"""
import numpy as np
import quantities as q
import pkg_resources
from scipy.ndimage import imread, shift
import scipy.interpolate as interp
import syris.config as cfg
from syris.opticalelements import OpticalElement
from syris.physics import energy_to_wavelength
from syris.util import get_gauss
from syris.materials import make_henke


class Filter(OpticalElement):

    """Beam frequency filter."""

    def get_next_time(self, t_0, distance):
        """A filter doesn't move, this function returns infinity."""
        return np.inf * q.s


class GaussianFilter(OpticalElement):

    """Gaussian beam filter."""

    def __init__(self, energies, center, sigma, peak_transmission=1):
        """Create a Gussian beam filter for *energies* [keV], center it at *center* [keV] and use
        std *sigma* [keV]. *peak_transmission* specifies the transmitted intensity for energy
        *center*, i.e. this is the highest transmitted intensity.
        """
        if len(energies) < 4:
            raise ValueError('Number of energy points too low for interpolation')
        energies = energies.rescale(q.keV).magnitude
        center = center.rescale(q.keV).magnitude
        sigma = sigma.rescale(q.keV).magnitude
        profile = get_gauss(energies, center, sigma) * peak_transmission * q.keV
        self._tck = interp.splrep(energies, profile)

    def get_next_time(self, t_0, distance):
        """A filter doesn't move, this function returns infinity."""
        return np.inf * q.s

    def _transfer(self, shape, pixel_size, energy, offset, exponent=False, t=None, queue=None,
                  out=None, check=True, block=False):
        """Transfer function implementation. Only *energy* is relevant because a filter has the same
        thickness everywhere.
        """
        coeff = interp.splev(energy.rescale(q.keV).magnitude, self._tck)
        eps = np.finfo(cfg.PRECISION.np_float).eps

        if exponent:
            result = np.log(coeff) / 2 if np.abs(coeff) > eps else -np.inf
        else:
            result = np.sqrt(coeff) if np.abs(coeff) > eps else 0.0

        return cfg.PRECISION.np_cplx(result)


class MaterialFilter(Filter):

    """Beam frequency filter."""

    def __init__(self, thickness, material):
        """Create a beam filter with projected *thickness* in beam direction
        and *material*.
        """
        self.thickness = thickness.simplified
        self.material = material

    def get_attenuation(self, energy):
        """Get attenuation at *energy*."""
        return (self.thickness *
                self.material.get_attenuation_coefficient(energy)).simplified.magnitude

    def get_next_time(self, t_0, distance):
        """A filter doesn't move, this function returns infinity."""
        return np.inf * q.s

    def _transfer(self, shape, pixel_size, energy, offset, exponent=False, t=None, queue=None,
                  out=None, check=True, block=False):
        """Transfer function implementation. Only *energy* is relevant because a filter has the same
        thickness everywhere.
        """
        lam = energy_to_wavelength(energy).simplified.magnitude
        thickness = self.thickness.simplified.magnitude
        ri = self.material.get_refractive_index(energy)
        result = -2 * np.pi / lam * thickness * (ri.imag + ri.real * 1j)

        if not exponent:
            result = np.exp(result)

        return result.astype(cfg.PRECISION.np_cplx)


class Scintillator(MaterialFilter):

    """Scintillator emits visible light when it is irradiated by X-rays."""

    def __init__(self, thickness, material, light_yields, energies, luminescence, wavelengths,
                 optical_ref_index):
        """Create a scintillator with *light_yields* [1 / keV] at *energies*, *luminescence* are the
        portions of total emmitted photons per some portion of wavelengths [1 / nm] (they are
        normalized so that their integral is 1) with respect to visible light *wavelengths*,
        *optical_ref_index* is the refractive index between the scintillator material and air.
        """
        super(Scintillator, self).__init__(thickness, material)
        self._lights_yields = light_yields
        self._energies = energies
        self._wavelengths = wavelengths
        self._luminescence = luminescence / luminescence.sum() / self.d_wavelength
        self.opt_ref_index = optical_ref_index

        self._ly_tck = interp.splrep(self._energies.rescale(q.keV).magnitude,
                                     self._lights_yields.rescale(1 / q.keV).magnitude)
        self._lum_tck = interp.splrep(self._wavelengths.rescale(q.nm).magnitude,
                                      self._luminescence.rescale(1 / q.nm).magnitude)

    @property
    def wavelengths(self):
        """Wavelengths for which the emission is defined."""
        return self._wavelengths

    @property
    def d_wavelength(self):
        """Wavelength spacing."""
        return (self.wavelengths[1] - self.wavelengths[0]).rescale(q.nm)

    def get_light_yield(self, energy):
        """Get light yield at *energy* [1 / keV]."""
        return interp.splev(energy.rescale(q.keV).magnitude, self._ly_tck) / q.keV

    def get_luminescence(self, wavelength):
        """Get luminescence at *wavelength* [1 / nm]."""
        return interp.splev(wavelength.rescale(q.nm).magnitude, self._lum_tck) / q.nm

    def get_conversion_factor(self, energy):
        """Get the conversion factor to convert X-ray photons to visible light photons
        [dimensionless].
        """
        absorbed = 1 - np.exp(-self.get_attenuation(energy))
        ly = self.get_light_yield(energy)

        return absorbed * ly * energy.rescale(q.keV)


class CdWO4(Scintillator):

    def __init__(self, thickness, energies, material = None):
        lam, em = np.load(pkg_resources.resource_filename(__name__,
                                                          'data/cdwo4_emission_spectrum.npy'))
        lam = lam * q.nm
        em = em * 1 / q.nm
        mat = make_henke('CdWo4', energies, density = 7.9 *q.g / (q.cm)**3, formula = 'CdWO4')
        ly = 14 * np.ones(len(energies)) / q.keV
        super(CdWO4, self).__init__(thickness, mat, ly, energies, em, lam, 1)


class CrystalSurfaceArtefacts(OpticalElement):
    def __init__(self, path, trajectory):
        self.path = path
        self.trajectory = trajectory

        x = (np.arange(0, 2048) * 2 * q.um - 1024 * 2 * q.um).rescale(q.m)
        y = (np.arange(0, 2048) * 2 * q.um - 1024 * 2 * q.um).rescale(q.m)
        mask = imread(self.path) / 255. * 2
        self._spline2d = interp.RectBivariateSpline(x,y,mask)

    def get_next_time(self, t_0, distance):
        """A filter doesn't move, this function returns infinity."""
        return np.inf * q.s

    def _transfer(self, shape, pixel_size, energy, offset, exponent=False, t=None, queue=None,
                  out=None, check=True, block=False):
        """Transfer function implementation.
        """
        if queue is None:
            queue = cfg.OPENCL.queue
        if out is None:
            out = cl_array.zeros(queue, shape, dtype=cfg.PRECISION.np_cplx)

        if t is None:
            x, y, z = self.trajectory.control_points.simplified.magnitude[0]
        else:
            x, y, z = self.trajectory.get_point(t).simplified.magnitude
        x += offset[1].simplified.magnitude
        y += offset[0].simplified.magnitude
        center = (x, y, z)

        x = np.arange(0, shape[0]) * pixel_size[0] - center[0] * q.m
        y = np.arange(0, shape[1]) * pixel_size[1] - center[1] * q.m
        mask = self._spline2d(x, y)

        mask_cl = cl_array.zeros(queue, shape, dtype=cfg.PRECISION.np_cplx)
        mask_cl.set(mask.astype(cfg.PRECISION.np_cplx)  * 1j)

        if not exponent:
            mask_cl = clmath.exp(mask_cl, queue)

        return mask_cl
