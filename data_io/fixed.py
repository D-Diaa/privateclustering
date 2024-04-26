import numpy as np

MOD = np.iinfo(np.int32).max + 1
PREC = 16
SCALE = 2 ** PREC


def to_int(values):
    return values.astype(np.int32)


def to_fixed(values):
    return to_int(values * SCALE)


def rescale(values):
    return to_int(unscale(values))


def unscale(values):
    return values.astype(np.float32) / SCALE
