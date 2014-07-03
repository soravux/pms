import sys
import os
from copy import copy
from itertools import product

from generate import generateImages

# Add parent directory into sys.path
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.sys.path.insert(0, parentdir)

import pms


def doTestAndCompare(template, lights):
    images, lightning_file = generateImages(template, lights)
    normals = pms.photometricStereo(lightning_file, images)
    import pickle
    with open('data.pkl', 'wb') as fhdl:
        pickle.dump(normals, fhdl)

    #for image in images:
    #    os.remove(image)
    os.remove(lightning_file)


lights = (
    (30, ),
    (-20, 0, 20),
    (-20, 0, 20),
)
light_positions = list(product(*lights))
#light_positions = [light_positions[0], light_positions[1], light_positions[6]]


def test_sphere():
    doTestAndCompare("sphere.pov.tmpl", light_positions)

#def test_cube_front():
#    doTestAndCompare("cube_front.pov.tmpl", light_positions)

#def test_cube_angled():
#    doTestAndCompare("cube_angled.pov.tmpl", light_positions)

#import pdb; pdb.set_trace()
if __name__ == '__main__':
    funcs = list(globals().keys())
    for func in funcs:
        try:
            func_name = globals()[func].__name__
        except AttributeError:
            continue
        if not func_name.startswith('test'):
            continue
        print("Executing", func_name)
        globals()[func]()