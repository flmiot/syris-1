"""Synchrotron radiation imaging experiments base module."""
import numpy as np
import pyopencl.array as cl_array
import quantities as q
import logging
import syris.config as cfg
import syris.math as smath
import syris.imageprocessing as ip
from syris.physics import propagate, compute_propagator, energy_to_wavelength
from syris.devices.filters import GaussianFilter


LOG = logging.getLogger(__name__)


class Experiment(object):

    """A virtual synchrotron experiment base class."""

    def __init__(self, samples, source, detector, propagation_distance, energies):
        self.source = source
        self.samples = samples
        self.detector = detector
        self.propagation_distance = propagation_distance
        self.energies = energies
        self._time = None

    @property
    def time(self):
        """Total time of all samples."""
        if self._time is None:
            self._time = max([obj.trajectory.time for obj in self.samples
                              if obj.trajectory is not None])

        return self._time

    def get_next_time(self, t, pixel_size):
        """Get next time from *t* for all the samples."""
        return min([obj.get_next_time(t, pixel_size) for obj in self.samples])

    def make_source_blur(self, shape, pixel_size, queue=None, block=False):
        """Make geometrical source blurring kernel with *shape* (y, x) size and *pixel_size*. Use
        OpenCL command *queue* and *block* if True.
        """
        l = self.source.sample_distance
        size = self.source.size
        width = (self.propagation_distance * size[1] / l).simplified.magnitude
        height = (self.propagation_distance * size[0] / l).simplified.magnitude
        sigma = (smath.fwnm_to_sigma(height, n=2), smath.fwnm_to_sigma(width, n=2)) * q.m

        return ip.get_gauss_2d(shape, sigma, pixel_size=pixel_size, fourier=True,
                               queue=queue, block=block)

    def compute_intensity(self, t_0, t_1, shape, pixel_size, queue=None, block=False):
        """Compute intensity between times *t_0* and *t_1*."""
        exp_time = (t_1 - t_0).simplified.magnitude
        image = propagate(self.samples, shape, self.energies, self.propagation_distance,
                          pixel_size, detector=self.detector, t=t_0) * exp_time

        return image

    def make_sequence(self, t_start, t_end, shape=None, shot_noise=True, amplifier_noise=True,
                      source_blur=True, queue=None):
        """Make images between times *t_start* and *t_end*."""
        if queue is None:
            queue = cfg.OPENCL.queue
        shape_0 = self.detector.camera.shape
        if shape is None:
            shape = shape_0
        ps_0 = self.detector.pixel_size
        ps = shape_0[0] / float(shape[0]) * ps_0
        fps = self.detector.camera.fps
        frame_time = 1 / fps
        times = np.arange(t_start.simplified.magnitude, t_end.simplified.magnitude,
                          frame_time.simplified.magnitude) * q.s
        image = cl_array.Array(queue, shape, dtype=cfg.PRECISION.np_float)
        source_blur_kernel = None
        if source_blur:
            source_blur_kernel = self.make_source_blur(shape, ps, queue=queue, block=False)

        fmt = 'Making sequence with shape {} and pixel size {} from {} to {}'
        LOG.debug(fmt.format(shape, ps, t_start, t_end))

        for t_0 in times:
            image.fill(0)
            t = t_0
            t_next = self.get_next_time(t, ps)
            while t_next < t_0 + frame_time:
                LOG.debug('Motion blur: {} -> {}'.format(t, t_next))
                image += self.compute_intensity(t, t_next, shape, ps)
                t = t_next
                t_next = self.get_next_time(t, ps)
            image += self.compute_intensity(t, t_0 + frame_time, shape, ps)
            if source_blur:
                image = ip.ifft_2(ip.fft_2(image) * source_blur_kernel).real
            camera_image = self.detector.camera.get_image(image, shot_noise=shot_noise,
                                                          amplifier_noise=amplifier_noise)
            LOG.debug('Image: {} -> {}'.format(t_0, t_0 + frame_time))
            yield camera_image


class Tomography(object):

    """A virtual discontinous tomography experiment class."""

    def __init__(self, oe, sample, source, detector, distances, energies, sample_name="unnamed"):
        """Make tomography experiment with list of optical elements *oe*, tomography sample *sample*,
        *source*, *detector*, positioned at *distances*. """

        self.source = source
        self.oe = oe
        self.sample = sample
        self.detector = detector
        self.distances = distances
        self.energies = energies
        self._time = None
        self._clock = 0*   q.s
        self.sample_name = sample_name

        if len(distances) != len(oe):
            raise ValueError("For every optical element, exactly one distance needs to be specified.")


    @property
    def time(self):
        """Total time of all samples."""
        if self._time is None:
            self._time = max([obj.trajectory.time for obj in self.oe
                              if obj.trajectory is not None])
        return self._time


    def get_next_time(self, t, pixel_size):
        """Get next time from *t* for all the samples."""
        return min([obj.get_next_time(t, pixel_size) for obj in self.oe])


    def tomo_rotate( self, angle ):
        """ Rotate *sample*. """

        x = np.sin(angle.rescale(q.rad))
        z = np.cos(angle.rescale(q.rad))
        for body in self.sample._bodies:
            body.trajectory._direction = np.array((x,0,z))
            body.update_projection_cache()


    def compute_intensity(self, t_0, t_1, shape, pixel_size, queue=None, block=False, flat = False):
        """Compute intensity between times *t_0* and *t_1*."""

        exp_time = (t_1 - t_0).simplified.magnitude

        if queue is None:
            queue = cfg.OPENCL.queue
        u = cl_array.Array(queue, shape, dtype=cfg.PRECISION.np_cplx)
        u_sample = cl_array.zeros(queue, shape, cfg.PRECISION.np_cplx)
        intensity = cl_array.zeros(queue, shape, cfg.PRECISION.np_float)

        for energy in self.energies:
            u.fill(1)
            for oeid, oe in enumerate(self.oe):

                if flat and oe == self.sample:
                    continue

                u *= oe.transfer(shape, pixel_size, energy, t=t_0, queue=queue,
                                 out = u_sample, check=False, block=block)

                # Propagate and blur optical element when not source
                if self.distances[oeid] != 0*q.m and oe != self.source:
                    lam = energy_to_wavelength(energy)
                    propagator = compute_propagator(u.shape[0], self.distances[oeid], lam, pixel_size,
                                                    queue=queue, block=block, mollified = True)

                    ip.fft_2(u, queue=queue, block=block)

                    sdistance = np.sum(self.distances[:oeid+1])
                    fwhm = (self.distances[oeid] * self.source.size / sdistance).simplified
                    sigma = smath.fwnm_to_sigma(fwhm, n=2)
                    psf = ip.get_gauss_2d(shape, sigma, pixel_size=pixel_size, fourier=True,
                                          queue=queue, block=block)
                    u *= psf
                    u *= propagator
                    ip.ifft_2(u, queue=queue, block=block)

            intensity += self.detector.convert(abs(u) ** 2, energy)

        return intensity * exp_time


    def write_scan_log( self, path, projections, num_ref_per_block, num_proj_per_block,
                        rotation, num_dark_img, shape = None):
        """Write scan log (DESY P05/P07 format) to *path*. """

        scan_str = (
        'energy={}\n'
        'distance={}\n'
        'ROI={}\n'
        'eff_pix={}\n'
        'projections={}\n'
        'num_ref_per_block={}\n'
        'ref_prefix=ref\n'
        'num_proj_per_block={}\n'
        'sample={}\n'
        'rotation={}\n'
        'pos_s_stage_z=0.0\n'
        'angle_order=continuous\n'
        'height_steps=1\n'
        'dark_prefix=dark\n'
        'num_dark_img={}\n'
        'exposure_time={}\n'
        'proj_prefix=proj\n'
        'off_axes=0\n'
        )

        shape_0 = self.detector.camera.shape
        if shape is None:
            shape = shape_0

        outstr = scan_str.format(
                        np.mean(self.energies.rescale(q.eV).magnitude),
                        self.distances[-1].rescale(q.mm).magnitude,
                        ',{},0,{},0'.format(shape[0], shape[1]),
                        self.detector.pixel_size.rescale(q.mm).magnitude,
                        projections, num_ref_per_block, num_proj_per_block,
                        self.sample_name, rotation, num_dark_img,
                        self.detector.camera._exp_time.rescale(q.s).magnitude)

        with open(path, "w") as text_file:
            text_file.write(outstr)


    def make_tomography(self, projections, rotation, pause, num_ref_per_block = 1,
                        num_proj_per_block = 1, num_dark_img = 0, start_frame = 0,
                        shape=None, shot_noise=True, amplifier_noise=True,
                        source_blur=True, queue=None):
        """Make sequence of *projections* projection images over 0 to *rotation* degrees. *pause*
        after each image. Proceed in image blocks, with *num_ref_per_block* flatfields and
        *num_proj_per_block* projections per block. Make *num_dark_img* dark images at the beginning.
        Start with *start_frame* (must be less or equal total number of images). """

        if queue is None:
            queue = cfg.OPENCL.queue
        shape_0 = self.detector.camera.shape
        if shape is None:
            shape = shape_0
        ps_0 = self.detector.pixel_size
        ps = shape_0[0] / float(shape[0]) * ps_0

        image = cl_array.Array(queue, shape, dtype=cfg.PRECISION.np_float)
        source_blur_kernel = None
        #if source_blur:
            #source_blur_kernel = self.make_source_blur(shape, ps, queue=queue, block=False)

        angles = np.linspace(0, rotation, num = projections) * q.deg
        angle_step_size = abs(angles[1] - angles[0])
        overall_no_images = num_dark_img + projections + \
            projections / num_proj_per_block * num_ref_per_block

        DARK_IMAGE = 0
        PROJECTION = 1
        FLATFIELD = 2

        darks = np.repeat(DARK_IMAGE, num_dark_img)
        blocks = np.repeat(PROJECTION, num_proj_per_block)
        blocks = np.append(blocks, np.repeat(FLATFIELD, num_ref_per_block))
        blocks = np.tile(blocks, projections / num_proj_per_block)
        image_type = np.append(darks, blocks)

        self.clock = 0*q.s
        exptime = self.detector.camera._exp_time

        counter_darkimages = 0
        counter_projections = 0
        counter_flatfields = 0

        for i in np.arange(0, overall_no_images):

            if start_frame > i:
                self.clock += exptime + pause
                if image_type[i] == DARK_IMAGE:
                    counter_darkimages += 1
                elif image_type[i] == PROJECTION:
                    counter_projections += 1
                elif image_type[i] == FLATFIELD:
                    counter_flatfields += 1
                yield None, None

            else:
                image.fill(0)

                t_0 = self.clock
                t_next = self.get_next_time(self.clock, ps)
                image_name = None

                # Dark images:
                if image_type[i] == DARK_IMAGE:
                    image_name = 'dark_{:>05}.tif'.format(counter_darkimages)
                    counter_darkimages +=1
                    self.clock += exptime + pause

                # Projections:
                elif image_type[i] == PROJECTION:
                    # Turn sample
                    self.tomo_rotate( angles[counter_projections] )
                    while t_next < t_0 + exptime:
                        LOG.debug('Motion blur: {} -> {}'.format(t_0, t_next))
                        image += self.compute_intensity(self.clock, t_next, shape, ps)
                        self.clock = t_next
                        t_next = self.get_next_time(self.clock, ps)
                    image += self.compute_intensity(self.clock, t_0 + exptime, shape, ps)
                    self.clock = t_0 + exptime + pause

                    #if source_blur:
                        #image = ip.ifft_2(ip.fft_2(image) * source_blur_kernel).real
                    image_name = 'proj_{:>05}.tif'.format(counter_projections)
                    counter_projections +=1
                    LOG.debug('Projection: {} -> {}'.format(t_0, t_0 + exptime))

                # Flatfields:
                elif image_type[i] == FLATFIELD:
                    while t_next < t_0 + exptime:
                        LOG.debug('Motion blur: {} -> {}'.format(t_0, t_next))
                        image += self.compute_intensity(self.clock, t_next, shape, ps, flat = True)
                        self.clock = t_next
                        t_next = self.get_next_time(self.clock, ps)
                    image += self.compute_intensity(self.clock, t_0 + exptime, shape, ps, flat = True)
                    self.clock = t_0 + exptime + pause
                    image_name = 'ref_{:>05}.tif'.format(counter_flatfields)
                    counter_flatfields +=1

                else:
                    raise ValueError("Unknow image type requested. "\
                        "Options are: Dark image, projection, flatfield.")

                camera_image = self.detector.camera.get_image(image, shot_noise=shot_noise,
                                                          amplifier_noise=amplifier_noise)

                yield camera_image, image_name
