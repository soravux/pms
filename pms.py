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
    import matplotlib.pyplot as plt
    #plt.quiver(normals[:,:,0], normals[:,:,1])
    #plt.savefig('normals.png')
    #plt.show()
    plt.imsave('normals1.png', normals[:,:,0])
    plt.imsave('normals2.png', normals[:,:,1])
    plt.imsave('normals3.png', normals[:,:,2])


if __name__ == "__main__":
    main()
