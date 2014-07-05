#!/usr/bin/env python
import argparse
import json
import pickle

import numpy as np
from scipy.misc import imread

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


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

    this_array = np.dstack([x, -y, z]).swapaxes(0, 1)
    color = colorizeNormals(this_array)
    img = color

    img[~valid] = 0

    return img


def main():
    parser = argparse.ArgumentParser(
        description="Photometric Stereo",
    )
    parser.add_argument(
        "lightning",
        nargs="?",
        help="Filename of JSON file containing lightning information",
    )
    parser.add_argument(
        "image",
        nargs="*",
        help="Images filenames",
    )
    parser.add_argument(
        "--generate-map",
        action='store_true',
        help="Generate a map.png file which represends the colors of the "
             "normal mapping.",
    )
    args = parser.parse_args()
        
    if args.generate_map:
        normals = generateNormalMap()
        plt.imsave('map.png', normals)
        return

    if not (args.lightning and len(args.image) >= 3):
        print("Please specify a lightning file and 3+ image files.")
        return

    try:
        with open('data.pkl', 'rb') as fhdl:
            normals = pickle.load(fhdl)
    except:
        normals = photometricStereo(args.lightning, args.image)
        with open('data.pkl', 'wb') as fhdl:
            pickle.dump(normals, fhdl)

    color = colorizeNormals(normals)
    plt.imsave('out.png', color)


if __name__ == "__main__":
    main()