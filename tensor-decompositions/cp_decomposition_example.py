"""
CP (CANDECOMP/PARAFAC) decomposition of a 3D tensor
"""
import numpy as np
from tensorly.decomposition import parafac
import tensorly as tl

tensor = tl.tensor(np.random.rand(4, 4, 4))
factors = parafac(tensor, rank=2)
print("CP decomposition factors:")
for i, f in enumerate(factors[1]):
    print(f"Mode {i+1} factor shape: {f.shape}")
