#!/usr/bin/env python
import argparse
import json

import numpy as np
from scipy.misc import imread

import matplotlib
matplotlib.use('Agg')


def getImage(filename):
    return imread(filename, flatten=True)


def getLightning(filename):
    with open(filename, 'r') as fhdl:
        retVal = json.load(fhdl)
    return retVal


def photometricStereo(lightning_filename, images_filenames):
    lightning = getLightning(lightning_filename)
    images = list(map(getImage, images_filenames))

    I = np.vstack(x.ravel() for x in images)
    N = np.vstack(lightning[x] for x in images_filenames)
    N_i = np.linalg.pinv(N)
    # TODO: Check impact of rho
    rho = np.linalg.norm(N_i.dot( I ), axis=0)
    normals, residual, rank, s = np.linalg.lstsq(N, I)
    w, h = images[0].shape
    normals = normals.reshape(3, w, h).swapaxes(0, 2)
    # TODO: Raise an error on misbehavior of lstsq.

    return normals


def colorizeNormals(normals):
    """Saves the normals as an image"""
    # Normalize the normals
    nf = np.linalg.norm(normals, axis=normals.ndim - 1)
    normals_n = normals / np.dstack((nf, nf, nf))

    color = (normals_n + 1) / 2

    return color

def generateNormalMap(dims=600):
    """Generate a mapping of the normals to understand the colorizeNormals
    output."""
    x, y = np.meshgrid(np.linspace(-1, 1, dims), np.linspace(-1, 1, dims))
    zsq = 1 - np.power(x, 2) - np.power(y, 2)

    valid = zsq >= 0

    z = np.zeros(x.shape)
    z[valid] = np.sqrt(zsq[valid])

    img = np.transpose(colorizeNormals(np.array([x.ravel(), -y.ravel(), z.ravel()])), [2, 3, 1])
    img = img.reshape((dims[0], dims[1], 3))

    img[~valid[:,:,[1, 1, 1]]] = 0

    return img


def main():
    parser = argparse.ArgumentParser(
        description="Photometric Stereo",
    )
    parser.add_argument(
        "lightning",
        help="Filename of JSON file containing lightning information",
    )
    parser.add_argument(
        "image",
        nargs="+",
        help="Images filenames",
    )
    args = parser.parse_args()

    import pickle
    try:
        with open('data.pkl', 'rb') as fhdl:
            normals = pickle.load(fhdl)
    except:
        normals = photometricStereo(args.lightning, args.image)
        with open('data.pkl', 'wb') as fhdl:
            pickle.dump(normals, fhdl)

    color = colorizeNormals(normals)
    import matplotlib.pyplot as plt
    plt.imsave('out.png', color)

    normals = generateNormalMap()
    plt.imsave('map.png', normals)


if __name__ == "__main__":
    main()
