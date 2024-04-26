import numpy as np

from data_io.fixed import to_fixed


def load_txt(path: str):
    values_list = []
    with open(path, "r") as f:
        lines = f.readlines()
        for line in lines:
            if "x" in line:
                continue
            values_list.append([float(x) for x in line.split()])
    values_arr = np.array(values_list)
    return values_arr


def shuffle_and_split(values, clients, proportions=None):
    if proportions is None:
        size = len(values) // clients
        sizes = [size for _ in range(clients - 1)]
    else:
        prop_sum = sum(proportions)
        total = values.shape[0]
        sizes = [int(proportions[i] / prop_sum * total) for i in range(clients - 1)]
    np.random.shuffle(values)
    st = 0
    value_lists = []
    for client in range(clients - 1):
        size = sizes[client]
        value_lists.append(values[st: st + size, :])
        st += size
    value_lists.append(values[st:, :])
    return value_lists


def normalize(values, fixed=False, bounds=None):
    if bounds is None:
        bounds = [-1, 1]
    mx = values.max(axis=0)
    mn = values.min(axis=0)
    normalized = (values - mn) / (mx - mn)
    normalized = normalized * 2 - 1
    normalized[normalized < 0] *= (-bounds[0])
    normalized[normalized >= 0] *= bounds[1]
    if fixed:
        return to_fixed(normalized)
    else:
        return normalized
