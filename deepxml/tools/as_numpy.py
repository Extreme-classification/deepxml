import numpy as np 
import sys


n_rows, n_cols = int(sys.argv[2]), int(sys.argv[3])
print("Dims: ", n_rows, n_cols)
embed = np.zeros((n_rows, n_cols), dtype=np.float32)

def get_values(item):
    return np.array([float(_item.split(":")[1]) for _item in item.split(";")])

with open(sys.argv[1], 'r') as fp:
    for idx, line in enumerate(fp):
        _idx, vals = line.split("\t", maxsplit=1)
        embed[int(_idx), :]  = get_values(vals)

np.save(sys.argv[4], embed)
