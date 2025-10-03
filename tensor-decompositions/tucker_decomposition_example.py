"""
Tucker decomposition of a 3D tensor
"""
import numpy as np
from tensorly.decomposition import tucker
import tensorly as tl

tensor = tl.tensor(np.random.rand(4, 4, 4))
core, factors = tucker(tensor, ranks=[2, 2, 2])
print("Tucker core shape:", core.shape)
for i, f in enumerate(factors):
    print(f"Mode {i+1} factor shape: {f.shape}")
