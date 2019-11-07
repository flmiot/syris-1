"""Synchrotron radiation imaging experiments base module."""
import numpy as np
import pyopencl.array as cl_array
import quantities as q
import logging
import syris.config as cfg
import syris.math as smath
import syris.imageprocessing as ip
from syris.physics import propagate


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


class TomoExperiment( Experiment ):

    """tbd"""

    def __init__(self, sample, source, detector, propagation_distance, energies,
                 sample_name = "sample"):
        self.sample = sample
        samples = [source]
        samples.extend(sample)
        self.sample_name = sample_name
        super(TomoExperiment, self).__init__(samples, source, detector, propagation_distance, energies)

    def rotate_sample( self, angle ):

        self.sample.clear_transformation()
        nx = self.detector.camera.shape[0]
        ny = self.detector.camera.shape[1]
        ps = self.detector.pixel_size


        self.sample.rotate_all_mesh_triangles( angle , geom.Y_AX)

    def make_tomography(self, projections, rotation, pause, num_ref_per_block = 1,
                        num_proj_per_block = 1, num_dark_img = 1, shape=None,
                        shot_noise=True, amplifier_noise=True, source_blur=True,
                        queue=None):
        """Make images for *angles*."""


        angles = np.linspace(0, rotation, num = projections) * q.deg
        angle_step_size = abs(angles[1] - angles[0])

        """
        # write scan log to file
        scan_str = (
        "energy={}/n"
        "distance={}/n"
        "ROI={}/n"
        "eff_pix={}/n"
        "projections={}/n"
        "num_ref_per_block={}/n"
        "ref_prefix=ref/n"
        "num_proj_per_block{}/n"
        "sample={}/n"
        "rotation={}/n"
        "pos_s_stage_z=0.0/n"
        "angle_order=continuous/n"
        "height_steps=1/n
        "dark_prefix=dark/n"
        "num_dark_img={}/n"
        "exposure_time={}/n"
        "proj_prefix=proj/n"
        "off_axes=0/n"
        )
        scan_str.format(np.mean(self.energies),  self.propagation_distance,
                   ",{},0,{},0".format(shape[0], shape[1]), self.detector.pixel_size,
                   projections, num_ref_per_block, num_proj_per_block, self.sample_name,
                   rotation, num_dark_img, self.detector.camera._exp_time)
        with open("scan.log", "w") as text_file:
            text_file.write(scan_str)
        """

        if queue is None:
            queue = cfg.OPENCL.queue
        shape_0 = self.detector.camera.shape
        if shape is None:
            shape = shape_0
        ps_0 = self.detector.pixel_size
        ps = shape_0[0] / float(shape[0]) * ps_0
        fps = self.detector.camera.fps
        frame_time = 1 / fps
        step_time = pause + frame_time
        t_start = 0*q.s
        t_end = ( pause + frame_time ) * len(angles) * q.s
        times = np.arange(t_start.simplified.magnitude, t_end.simplified.magnitude,
                          step_time.simplified.magnitude) * q.s
        image = cl_array.Array(queue, shape, dtype=cfg.PRECISION.np_float)
        source_blur_kernel = None
        #if source_blur:
            #source_blur_kernel = self.make_source_blur(shape, ps, queue=queue, block=False)

        for i, t_0 in enumerate(times):
            image.fill(0)

            t = t_0
            t_next = self.get_next_time(t, ps)

            # Turn sample
            self.rotate_sample( angle_step_size )

            while t_next < t_0 + frame_time:
                LOG.debug('Motion blur: {} -> {}'.format(t, t_next))
                image += self.compute_intensity(t, t_next, shape, ps)
                t = t_next
                t_next = self.get_next_time(t, ps)
            image += self.compute_intensity(t, t_0 + frame_time, shape, ps)
            #if source_blur:
                #image = ip.ifft_2(ip.fft_2(image) * source_blur_kernel).real
            camera_image = self.detector.camera.get_image(image, shot_noise=shot_noise,
                                                          amplifier_noise=amplifier_noise)
            LOG.debug('Image: {} -> {}'.format(t_0, t_0 + frame_time))
            yield camera_image
