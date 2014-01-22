"""Sample constitutes of graphical objects and to them assigned materials."""
import quantities as q
import numpy as np
import pyopencl as cl
from pyopencl.array import vec
from syris import config as cfg
from syris.gpu import util as g_util
import logging
from syris.opticalelements.graphicalobjects import CompositeObject,\
    get_moved_groups


LOGGER = logging.getLogger(__name__)


class Sample(object):

    """A sample consisting of moving graphical objects."""

    def __init__(self, parts, shape, pixel_size):
        """Create a moving sample composed of graphical objects, which are
        :py:class:`GraphicalObject` instances and materials assigned
        to them. Dictionary *parts* is in the form
        :py:class:`GraphicalObject`: :py:class:`Material`.
        """
        self.shape = shape
        self.pixel_size = pixel_size.simplified
        # Transform values to sets.
        self._parts = dict([(key, set(parts[key])) for key in parts])
        self._prg = g_util.get_program(g_util.get_metaobjects_source())

    @property
    def materials(self):
        """All sample materials."""
        return self._parts.keys()

    @property
    def objects(self):
        """Return all objects for all materials."""
        objects = []

        for mat in self._parts:
            for obj in self._parts[mat]:
                objects.append(obj)

        return objects

    def get_moved_groups(self, t_0):
        """
        Get the shortest time t_1 and fastest objects which move within
        the t_1 - *t_0*. There can be more of these objects
        because they might move with the same velocity.
        """
        shortest = None
        fastest_objects = []

        for obj in self.objects:
            t_1 = obj.get_next_time(t_0, self.pixel_size)
            if shortest is None or shortest > t_1:
                shortest = t_1
                fastest_objects = [obj]
            elif shortest == t_1:
                fastest_objects.append(obj)

        return shortest, get_moved_groups(fastest_objects, t_0, t_1,
                                          self.pixel_size)

    def move(self, abs_time):
        """
        Move from starting time *abs_time* and return the next time
        something within the sample moves.
        """
        next_t, groups = self.get_moved_groups(abs_time)

        for obj in groups:
            # Perform the actual calculation
            pass

        return next_t

    def get_material_thickness(self, material):
        """Return projected thickness of the sample made of *material*."""
        pass

    def get_objects(self, material):
        """Get graphical objects assigned with *material*."""
        return tuple(self._parts[material])

    def add(self, material, gr_obj):
        """
        Add graphical object *gr_obj* to the parts of the sample
        and assign *material* to it.
        """
        if material not in self._parts:
            self._parts[material] = set([gr_obj])
        else:
            self._parts[material].add(gr_obj)

    def get_moved_materials(self, t_0, t_1):
        """Return materials which objects moved between *t_0* and *t_1*."""
        materials = []

        for material in self._parts:
            for obj in self._parts[material]:
                if obj.moved(t_0, t_1, self.pixel_size / 2):
                    materials.append(material)
                    break

        return materials

    def _compute_thickness(self, th_mem, abs_time, offset, gr_obj,
                           clear=True):
        units = q.mm
        num_objects = 0

        gr_obj.clear_transformation()
        gr_obj.move(abs_time)
        b_box = gr_obj.bounding_box
        if gr_obj.__class__ == CompositeObject:
            num_objects = len(gr_obj.primitive_objects)
            size_coeff = 1.0 / _get_average_coefficient(gr_obj, "radius")
            z_middle = _get_average_coefficient(gr_obj, "position",
                                                lambda x: x[2])
        else:
            num_objects = 1
            size_coeff = 1 / gr_obj.radius
            z_middle = gr_obj.position[2]

        objects_string = gr_obj.pack(units, size_coeff)

        y_0, x_0, y_1, x_1 = [int(round(value / self.pixel_size))
                              for value in b_box.roi]

        objects_mem = cl.Buffer(cfg.CTX, cl.mem_flags.READ_ONLY |
                                cl.mem_flags.COPY_HOST_PTR,
                                hostbuf=objects_string)

        self._prg.thickness(cfg.QUEUE,
                            self.shape,
                            None,
                            th_mem,
                            objects_mem,
                            np.int32(num_objects),
                            vec.make_int2(*offset[::-1]),
                            vec.make_int4(x_0, y_0, x_1, y_1),
                            cfg.NP_FLOAT(size_coeff.rescale(1 / units)),
                            cfg.NP_FLOAT(z_middle.rescale(units)),
                            g_util.make_vfloat2(self.pixel_size.rescale(units),
                                                self.pixel_size.rescale(
                                                    units)),
                            np.int32(clear))

        objects_mem.release()

    def get_thickness(self, material, abs_time, offset):
        """
        Get sample's projected thickness at time *abs_time*. *offset*
        (y, x) is the spatial offset given by :py:class:`Tiler`.
        """

        LOGGER.debug("Creating objects at time {0}.".format(abs_time))

        th_mem = cl.Buffer(cfg.CTX, cl.mem_flags.READ_WRITE,
                           size=self.shape[0] * self.shape[1] * cfg.CL_FLOAT)

        i = 0
        for gr_obj in self._parts[material]:
            self._compute_thickness(th_mem, abs_time, offset, gr_obj, i == 0)
            i += 1

        return th_mem


class StaticSample(object):

    """A sample which does not move throughout an experiment."""

    def __init__(self, material, thickness):
        """Create a static sample with projected *thickness*."""
        self.material = material
        self.thickness = thickness


def _get_average_coefficient(composite, attr_name, func=lambda x: x):
    coeffs = []

    for g_o in composite:
        if g_o.__class__ == CompositeObject:
            for primitive in g_o.primitive_objects:
                coeffs.append(func(getattr(primitive, attr_name)))
        else:
            coeffs.append(func(getattr(g_o, attr_name)))

    return (min(coeffs) + max(coeffs)) / 2.0