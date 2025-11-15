import numpy as np

data_array = np.load("data/boss_sptypes.npy", allow_pickle=True)
print("dtype names:", data_array.dtype.names)
print("example record:", data_array[0])