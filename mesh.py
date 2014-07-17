from itertools import product
import struct
import pickle

import numpy as np
from scipy import sparse


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

class ASCII_STL_Writer(object):
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

    if np.allclose(n, np.zeros(n.shape)):
        return

    # Build two vectors orthogonal between themselves and the normal
    if (np.abs(n2) > 0.2 or np.abs(n3) > 0.2):
        C = np.array([1, 0, 0])
    else:
        C = np.array([0, 1, 0])
    ortho1 = np.cross(n, C)
    ortho1 *= l / np.linalg.norm(ortho1)
    ortho2 = np.cross(n, ortho1)
    ortho2 *= l / np.linalg.norm(ortho2)

    #ortho1[[2,1]] = ortho1[[1,2]]
    #ortho2[[2,1]] = ortho2[[1,2]]
    ortho1[1] = -ortho1[1]
    ortho2[1] = -ortho2[1]

    return [[
        center + ortho1,
        center + ortho2,
        center - ortho1,
        center - ortho2,
    ]]


def surfaceFromNormals(normals):
    valid_indices = ~np.isnan(normals)
    w, h, d = normals.shape
    nz = np.transpose(np.hstack((
        normals[:,:,2].ravel(),
        normals[:,:,2].ravel(),
    )))
    vectorsize = nz.shape
    valid_idx = ~np.isnan(nz)

    M = sparse.dia_matrix((2*w*h, w*h), dtype=np.float64)
    # n_z z(x + 1, y) - n_z z(x,y) = n_x
    M.setdiag(-nz, 0)
    M.setdiag(nz, 1)
    # n_z z(x, y + 1) - n_z z(x,y) = n_y
    M.setdiag(-nz, -w*h)
    M.setdiag(np.hstack(([0] * w, nz)), -w*h + w)

    
    # boundary values
    import pdb; pdb.set_trace()

    M = M.tocsr()[valid_idx]

    # build [ n_x n_y ]'
    m = np.hstack((
        normals[:,:,0].ravel(),
        normals[:,:,1].ravel(),
    )).reshape(-1, 1)
    m = m[valid_idx]

    #import pdb; pdb.set_trace()
    # Solve
    #x, istop, itn, r1norm, r2norm, anorm, acond, arnorm, xnorm, var = sparse.linalg.lsqr(M, m)
    x, istop, itn, normr, normar, norma, conda, normx = sparse.linalg.lsmr(M, m)

    # Build the surface (x, y, z)
    surface = np.dstack((
        np.indices((w, h))[0],
        np.indices((w, h))[1],
        x.reshape((w, h))
    ))
    return surface

def writeMesh(surface, normals, filename):
    s = surface
    with open(filename, 'wb') as fp:
        writer = Binary_STL_Writer(fp)
        for x in range(0, s.shape[0], 5):
            for y in range(0, s.shape[1], 5):
        #for x, y in product(range(s.shape[0]), range(s.shape[1])):
                quad = get_quad(
                    s[x,y,:],
                    normals[x,y,:],
                    4,
                )
                if quad:
                    writer.add_faces(quad)
        writer.close()


def write3dNormals(normals, filename):
    with open(filename, 'wb') as fp:
        writer = Binary_STL_Writer(fp)
        for x in range(0, normals.shape[0], 5):
            for y in range(0, normals.shape[1], 5):
                quad = get_quad(
                    (0, x, y),
                    normals[x,y,:],
                    4,
                )
                if quad:
                    writer.add_faces(quad)
        writer.close()


if __name__ == '__main__':
    with open('data.pkl', 'rb') as fhdl:
        normals = pickle.load(fhdl)
    writeMesh(normals)