import numpy as np
import pdb


def get_bb8_of_model(obj_m):
    xmin, ymin, zmin = obj_m[:, 0].min(), obj_m[:, 1].min(), obj_m[:, 2].min()
    xmax, ymax, zmax = obj_m[:, 0].max(), obj_m[:, 1].max(), obj_m[:, 2].max()
    bb8 = np.array([
        [xmin, ymin, zmin],
        [xmin, ymax, zmin],
        [xmax, ymax, zmin],
        [xmax, ymin, zmin],
        [xmin, ymin, zmax],
        [xmin, ymax, zmax],
        [xmax, ymax, zmax],
        [xmax, ymin, zmax],
    ])
    return bb8
