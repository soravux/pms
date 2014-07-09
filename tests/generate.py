import shutil
import subprocess
import itertools
import os
import json

import numpy as np


light_command = """
light_source
{{
  <{x},{y},{z}>
  color <1,1,1>
}}
"""


def generatePOVFile(filename, template, light_position):
    shutil.copy(template, filename)
    with open(filename, 'a') as fhdl:
        fhdl.write(light_command.format(
            x=light_position[0],
            y=light_position[1],
            z=light_position[2],
        ))


def generateImages(template, light_positions, lightning_file="lightning.json"):
    """Generates images using POV-Ray and lightning file"""
    devnull = open(os.devnull, 'w')
    base_name = template.split(".")[0]
    filenames = []
    lightning = {}
    for light_position in light_positions:
        filename = "{0}.{1}.pov".format(base_name, ".".join(map(str, light_position)))
        filenames.append(filename.rsplit(".", 1)[0] + ".png")
        if os.path.isfile(filenames[-1]):
            continue
        if np.linalg.norm(light_position) < 10:
            print("Light seems too close of the object.")

        generatePOVFile(filename, template, light_position)
        output = subprocess.call([
            "povray",
            "+FN", # PNG
            "+W640", #"+W1024",
            "+H480", #"+H768",
            filename,
        ], stderr=devnull)
        if output:
            raise Exception('Could not execute povray.')
        os.remove(filename)

        lightning[filenames[-1]] = np.negative(np.array(light_position, dtype=np.float64))
        lightning[filenames[-1]] /= np.linalg.norm(lightning[filenames[-1]])
        lightning[filenames[-1]] = lightning[filenames[-1]].tolist()

    with open(lightning_file, 'w') as fhdl:
        json.dump(lightning, fhdl)

    return filenames, lightning_file


if __name__ == '__main__':
    posx = [20]
    posy = [-20, 0, 20]
    posz = [-20, 0, 20]
    light_positions = itertools.product(posx, posy, posz)
    files = generateImages("cube_angled.pov.tmpl", light_positions)
    for file_ in files:
        os.remove(file_)