import numpy as np
import pickle

import struct

ASCII_FACET = """facet normal 0 0 0
outer loop
vertex {face[0][0]:.4f} {face[0][1]:.4f} {face[0][2]:.4f}
vertex {face[1][0]:.4f} {face[1][1]:.4f} {face[1][2]:.4f}
vertex {face[2][0]:.4f} {face[2][1]:.4f} {face[2][2]:.4f}
endloop
endfacet
"""

BINARY_HEADER ="80sI"
BINARY_FACET = "12fH"

class ASCII_STL_Writer:
    """ Export 3D objects build of 3 or 4 vertices as ASCII STL file.
    """
    def __init__(self, stream):
        self.fp = stream
        self._write_header()

    def _write_header(self):
        self.fp.write("solid python\n")

    def close(self):
        self.fp.write("endsolid python\n")

    def _write(self, face):
        self.fp.write(ASCII_FACET.format(face=face))

    def _split(self, face):
        p1, p2, p3, p4 = face
        return (p1, p2, p3), (p3, p4, p1)

    def add_face(self, face):
        """ Add one face with 3 or 4 vertices. """
        if len(face) == 4:
            face1, face2 = self._split(face)
            self._write(face1)
            self._write(face2)
        elif len(face) == 3:
            self._write(face)
        else:
            raise ValueError('only 3 or 4 vertices for each face')

    def add_faces(self, faces):
        """ Add many faces. """
        for face in faces:
            self.add_face(face)

class Binary_STL_Writer(ASCII_STL_Writer):
    """ Export 3D objects build of 3 or 4 vertices as binary STL file.
    """
    def __init__(self, stream):
        self.counter = 0
        super(Binary_STL_Writer, self).__init__(stream)

    def close(self):
        self._write_header()

    def _write_header(self):
        self.fp.seek(0)
        self.fp.write(struct.pack(BINARY_HEADER, b'Python Binary STL Writer', self.counter))

    def _write(self, face):
        self.counter += 1
        data = [
            0., 0., 0.,
            face[0][0], face[0][1], face[0][2],
            face[1][0], face[1][1], face[1][2],
            face[2][0], face[2][1], face[2][2],
            0
        ]
        self.fp.write(struct.pack(BINARY_FACET, *data))


def get_quad(center, n, side=1.):
    x, y, z = np.array(center).astype('float64')
    n1, n2, n3 = np.array(n).astype('float64')
    l = side/2.

    nm = np.linalg.norm
    s = np.sign

    if any(np.isnan(v) for v in n):
        return

    #if n2 < 0.2:
    #    return

    # Build two vectors orthogonal between themselves and the normal
    if (np.abs(n2) > 0.2 or np.abs(n3) > 0.2):
        C = np.array([1, 0, 0])
    else:
        C = np.array([0, 1, 0])
    ortho1 = np.cross(n, C)
    ortho1 *= l / np.linalg.norm(ortho1)
    ortho2 = np.cross(n, ortho1)
    ortho2 *= l / np.linalg.norm(ortho2)

    ortho1[[2,1]] = ortho1[[1,2]]
    ortho2[[2,1]] = ortho2[[1,2]]
    ortho1[1] = -ortho1[1]
    ortho2[1] = -ortho2[1]

    return [[
        center + ortho1,
        center + ortho2,
        center - ortho1,
        center - ortho2,
    ]]


with open('data.pkl', 'rb') as fhdl:
    normals = pickle.load(fhdl)

with open('out.stl', 'wb') as fp:
    writer = Binary_STL_Writer(fp)
    for x in range(0, normals.shape[0], 20):
        for y in range(0, normals.shape[1], 20):
            quad = get_quad(
                (0, x, y),
                normals[x,y,:],
                16,
            )
            if quad:
                print(normals[x,y,:], [int(j) for i in quad[0] for j in i])
                writer.add_faces(quad)
    writer.close()

