import numpy as np


diameters = {
    'cat': 15.2633,
    'ape': 9.74298,
    'benchvise': 28.6908,
    'bowl': 17.1185,
    'cam': 17.1593,
    'camera': 17.1593,
    'can': 19.3416,
    'cup': 12.5961,
    'driller': 25.9425,
    'duck': 10.7131,
    'eggbox': 17.6364,
    'glue': 16.4857,
    'holepuncher': 14.8204,
    'iron': 30.3153,
    'lamp': 28.5155,
    'phone': 20.8394
}

linemod_cls_names = ['ape', 'cam', 'cat', 'duck', 'glue', 'iron', 'phone', 'benchvise', 'can', 'driller', 'eggbox', 'holepuncher', 'lamp']

linemod_K = np.array([[572.4114, 0., 325.2611],
                  [0., 573.57043, 242.04899],
                  [0., 0., 1.]])