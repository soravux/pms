import shutil
import subprocess
import itertools
import os


light_command = """
light_source
{{
  <{x},{y},{z}>
  color White
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


def generateImages(template, light_positions):
    filenames = []
    for light_position in light_positions:
        filename = "{0}-{1}.pov".format(template.split(".")[0], ".".join(map(str, light_position)))
        generatePOVFile(filename, template, light_position)
        subprocess.call(["povray", filename])
        os.remove(filename)
        filenames.append(filename.rsplit(".", maxsplit=1)[0] + ".png")
    return filenames


if __name__ == '__main__':
    posx = [20]
    posy = [-20, 0, 20]
    posz = [-20, 0, 20]
    light_positions = itertools.product(posx, posy, posz)
    files = generateImages("sphere.pov.tmpl", light_positions)
    for file_ in files:
        os.remove(file_)