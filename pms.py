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
    images = map(getImage, images_filenames)

    I = np.dstack(images)
    N = np.vstack(lightning[x] for x in images_filenames)
    N_i =np.linalg.inv(N)
    normals = np.empty(I.shape)
    for x in range(I.shape[0]):
        for y in range(I.shape[1]):
            rho = np.linalg.norm(N_i.dot( I[x,y] ))
            normals[x,y] = N_i.dot( I[x,y] ) / rho
    import pdb; pdb.set_trace()
    
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
    #plt.savefig('blu.png')
    #plt.show()
    plt.imsave('blu1.png', normals[:,:,0])
    plt.imsave('blu2.png', normals[:,:,1])
    plt.imsave('blu3.png', normals[:,:,2])


if __name__ == "__main__":
    main()
