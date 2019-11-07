"""Collada import functionality."""
import os
import re
import numpy as np
import quantities as q
import xml.etree.ElementTree
from syris.bodies.mesh import Mesh
import syris.geometry as geom
from syris.materials import make_fromfile

def read_collada(filename, scene_origin, orientation=geom.X_AX, mapping='xxyzz-y', iterations=1):

    """Read collada (*.dae) scene file *filename* and return list with objects of type Mesh. Place
    it at *scene_origin*. *orientation* specifies the direction of orientation vectors for all
    meshes. Scene objects must have materials assigned. For assigned material 'Material', a syris
    material file 'Material.mat' is expected *filename*'s path. *mapping* can be used to change the
    mapping of coordinate systems between syris and collada scenes. Standard is 'xxyzz-y' to make
    z (collada) point upwards in syris (-y). Options are 'xxyyzz' and 'xxyzz-y'. *iterations* is
    passed to all mesh objects.
    Only geometry is imported, i.e there is currently no support for animated objects."""

    e = xml.etree.ElementTree.parse(filename).getroot()
    ns = {'co': 'http://www.collada.org/2005/11/COLLADASchema'}
    scene_units = float(e.findall('co:asset/co:unit', ns)[0].get("meter")) * q.m
    geometries = e.findall('co:library_geometries/co:geometry', ns)
    visual_scene = e.find('co:library_visual_scenes/co:visual_scene', ns)
    #animations = e.find('cp:library_animations/co:animation')

    Meshes = []

    for g in geometries:
        geom_name = g.get("id")[:-5]
        m = g.find('co:mesh', ns)
        arrays = m.findall("co:source/co:float_array",ns)
        pattern = re.compile(r'(?P<x>[0-9e.-]*) (?P<y>[0-9e.-]*) (?P<z>[0-9e.-]*) ')
        vstr = arrays[0].text + " "
        vertices = np.array(re.findall(pattern, vstr)).astype(np.float32)
        polylistr = m.find('co:polylist/co:p', ns).text + " "
        faces = np.array(polylistr.split()[::2]).astype(np.int)
        triangles = vertices[faces]

        if(mapping == 'xxyzz-y'):
            # mapping corresponds to  rotation around x-axis by 90 deg.
            mat_mapping = geom.rotate(90* q.deg, geom.X_AX)
        elif(mapping == 'xxyyzz'):
            mat_mapping = np.identity(4)
        else:
            raise ValueError('Invalid mapping option for collada import.')

        # above vertices are stored with respect to their object origin,
        # transform them to obtain global vertex coordinates
        # there are two possible ways to specify transformations in collada, matrix and transloc:
        # find out which one was used and apply transformations
        arg = "co:node[@name='" + geom_name + "']"
        trafos = visual_scene.find(arg,ns)
        if(trafos.find("co:rotate[@sid='rotationZ']", ns) is not None):
            # TransLoc format
            rotZ = float(trafos.find("co:rotate[@sid='rotationZ']", ns).text.split()[-1])
            rotY = float(trafos.find("co:rotate[@sid='rotationY']", ns).text.split()[-1])
            rotX = float(trafos.find("co:rotate[@sid='rotationX']", ns).text.split()[-1])
            scale = map(float, trafos.find("co:scale[@sid='scale']", ns).text.split())
            str_origin = trafos.find("co:translate[@sid='location']", ns).text.split()
            origin = map(float, str_origin)  * q.dimensionless

            matr_x = geom.rotate(rotX * q.deg, geom.X_AX)
            matr_y = geom.rotate(rotY * q.deg, geom.Y_AX)
            matr_z = geom.rotate(rotZ * q.deg, geom.Z_AX)
            mat_trans = geom.translate(origin)
            mat_scale = geom.scale(scale)

            # build transformation matrix
            mat = np.dot(mat_mapping, mat_trans)
            mat = np.dot(mat, mat_scale)
            mat = np.dot(mat, matr_z)
            mat = np.dot(mat, matr_y)
            mat = np.dot(mat, matr_x)

        elif (trafos.find("co:matrix[@sid='transform']", ns) is not None):
            # Matrix format
            str_mat = trafos.find("co:matrix[@sid='transform']", ns).text.split()
            mat = np.dot(mat_mapping, np.array(str_mat, dtype = float).reshape(4,4))

        else:
            raise RuntimeError("Transformation type in collada file could not be determined.")

        # apply transformation to vertices
        for i, vert in enumerate(triangles):
            vert = (tuple(vert) + (1,))
            triangles[i] = np.dot(mat, vert)[:-1]

        triangles = triangles.transpose()

        # handle materials
        material = m.find('co:polylist', ns).get("material")[:-9] + ".mat"
        matname = os.path.join(os.path.dirname(filename), material)
        if(not os.path.isfile(matname)):
            raise RuntimeError("Materialfile {} not found. Files have to be in the same "\
            "directory as the corresponding collada scene file.".format(matname))
        m = make_fromfile(matname)

        # create and add Mesh object
        tr = geom.Trajectory(scene_origin)
        mesh = Mesh(triangles * scene_units , tr, material = m, center = None,
                    orientation = orientation, iterations = iterations)
        Meshes.extend([ mesh ])

    return Meshes
