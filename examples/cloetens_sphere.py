# -*- coding: utf-8 -*-
"""
@author: otteflor

Simulate the double sphere phantom from Peter Cloetens thesis:

Contribution to Phase Contrast Imaging, Reconstruction and Tomography with Hard
Synchrotron Radiation - Principles, Implementation and Applications
"""

import os, sys
import time
import matplotlib.pyplot as plt
import numpy as np
import quantities as q
import scipy.misc as sc
import syris


import logging

from syris.physics import propagate
from syris.bodies.base import CompositeBody
from syris.bodies.mesh import Mesh
from syris.devices.cameras import make_kit_cmos
from syris.devices.detectors import Detector
from syris.devices.lenses import Lens
from syris.devices.filters import MaterialFilter, CdWO4, GaussianFilter
from syris.devices.sources import make_topotomo, SpectraSource
from syris.geometry import Trajectory
from syris.experiments import Tomography
from syris.gpu.util import get_host
from syris.math import fwnm_to_sigma
from syris.collada import read_collada
from util import get_material, show
import tifffile as tf
import syris.math as smath

PLOT_AND_PAUSE = True
OUTPUT = 'output'
OBJ_PATH = 'C:/Users/hambu/Git/syris-1/examples/data/double_sphere.dae'
SPECTRA_FILE = "C:/Users/hambu/Git/syris-1/examples/data/20keV@5th_86p5m_fov_-10_10mm.dta"
NO_OF_IMAGES = 10
THETA_MIN = 0
THETA_MAX = 180
START_I = 0
SUPERSAMPLING = 1
n = 2048 * SUPERSAMPLING
PAUSE = 0.01 *q.sec
EXPOSURE_TIME = 1000 * q.ms
DISTANCE = 86.5 * q.m
NUM_REF_PER_BLOCK = 0
NUM_PROJ_PER_BLOCK = 5
NUM_DARK_IMG = 0

LOG = logging.getLogger(__name__)
#syris.init( loglevel = logging.DEBUG)
syris.init()


shape = (n, n)
dE_mono = 1e-4
energy = 20000
energies = np.linspace(energy - 2*dE_mono* energy, energy + 2*dE_mono* energy, num = 4)*q.eV
dE = energies[1] - energies[0]

# === CAMERA ===
camera = make_kit_cmos(EXPOSURE_TIME)
camera.pixel_size = camera.pixel_size / SUPERSAMPLING
camera.shape = shape
# === LENS ===
lens = Lens(10, na = 0.1, transmission_eff=0.6, sigma=None)
# === SCINTILLATOR ===
scintillator = CdWO4(100 * q.um, energies)
# === DETECTOR ===
detector = Detector(scintillator, lens, camera)

# === BENDING MAGNET SOURCE ===
"""
Sector 4	sigma
BeamAngleDelta.X[5][0]	0,089555518	urad
BeamAngleDelta.Y[5][0]	0,033106066	urad
BeamPositionDelta.X[5][0]	0,16299738	um
BeamPositionDelta.Y[5][0]	0,062605455	um
"""
fak = 0.6
k = 1
ps = detector.pixel_size
vel = k * detector.pixel_size / EXPOSURE_TIME
points = k * NO_OF_IMAGES +  NO_OF_IMAGES
pos_rx = np.random.normal(loc = 0.0, scale =  fak * 0.089555518, size = points) * q.urad
pos_ry = np.random.normal(loc = 0.0, scale =  fak * 0.033106066, size = points) * q.urad
pos_x = n/2 * ps + np.random.normal(loc = 0.0, scale = fak * 0.16299738, size = points) * q.um + pos_rx * DISTANCE
pos_y = n/2 * ps + np.random.normal(loc = 0.0, scale = fak * 0.062605455, size = points) * q.um + pos_ry * DISTANCE
pos_z = np.zeros(points) * q.m
pp = zip(pos_x.rescale(q.m).simplified.magnitude, pos_y.rescale(q.m).simplified.magnitude, pos_z.rescale(q.m).simplified.magnitude) * q.m


#spectra_file = "/home/ubuntu/ownCloud/spectra/u1_2m_lb/08keV@1st_70m_fov_-10_10mm_gap_tbdmm.dta"


#source_trajectory = Trajectory([(n/2, n/2, 0)] * detector.pixel_size)
source_trajectory = Trajectory(pp, velocity = vel )
source_trajectory_stat = Trajectory([(n/2, n/2, 0)] * ps )
undu = SpectraSource(SPECTRA_FILE, DISTANCE, dE, (5, 140)*q.um, detector.pixel_size,
                     source_trajectory, phase_profile = 'sphere', fluctuation = 0.06)
undu.trajectory.bind(detector.pixel_size)
#bm = make_topotomo(dE=dE, trajectory=source_trajectory, pixel_size=detector.pixel_size)
#bm.trajectory.bind(detector.pixel_size)

## == GAUSIAN FILTER (MONOCHROMATOR APPROX) ==
fwhm = dE_mono * energy * q.eV
sigma = smath.fwnm_to_sigma(fwhm, n=2)
fltr = GaussianFilter(energies, energy * q.eV, sigma)

# == SAMPLE ===
meshes = read_collada(OBJ_PATH, [(n / 2, n / 2, 0)] * detector.pixel_size, iterations = 1)

tr = Trajectory([(0 / 2, 0 / 2, 0)] * detector.pixel_size)

cb = CompositeBody(tr, bodies = meshes)
cb.bind_trajectory(detector.pixel_size)

airm = get_material('air_dry.mat')
cu = get_material('cu.mat')
air_gap = MaterialFilter(3*q.m, airm)
abs_fltr = MaterialFilter(100*q.um, cu)
# === MAKE EXPERIMENT ===
ex = Tomography( [undu, fltr, abs_fltr, air_gap, cb], cb, undu, detector, [53.455,0,0,32.477,0.15] * q.m, energies)


# == CONDUCT EXPERIMENT
if OUTPUT is not None and not os.path.exists( OUTPUT ):
    os.makedirs( OUTPUT, mode=0o755)

t_0 = 0 * q.s

t_1 = NO_OF_IMAGES / detector.camera.fps
st = time.time()
mpl_im = None

# make projections
for i, [data, filename] in enumerate(ex.make_tomography(NO_OF_IMAGES, THETA_MAX,
    PAUSE, NUM_REF_PER_BLOCK, NUM_PROJ_PER_BLOCK, NUM_DARK_IMG, start_frame = START_I, shape = shape)):

    if START_I <= i:
        image = get_host(data)
        msg = '===== COMPUTED {}'
        LOG.debug(msg.format(filename))

        if PLOT_AND_PAUSE:
            show(image)
            plt.show()

        if OUTPUT:
            path_img = os.path.join( OUTPUT, filename)
            tf.imsave(path_img, image.astype(np.uint16))


path_log = os.path.join( OUTPUT, 'scan.log')
ex.write_scan_log(path_log, NO_OF_IMAGES, NUM_REF_PER_BLOCK,
                  NUM_PROJ_PER_BLOCK, THETA_MAX, NUM_DARK_IMG)
