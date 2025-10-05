import math

def dot(u, v):
    return sum(ui * vi for ui, vi in zip(u, v))

def norm(v):
    return math.sqrt(dot(v, v))

def scalar_multiply(scalar, v):
    return [scalar * vi for vi in v]

def subtract(u, v):
    return [ui - vi for ui, vi in zip(u, v)]

def gram_schmidt(vectors):
    orthonormal = []
    for v in vectors:
        w = v[:]
        for u in orthonormal:
            proj = scalar_multiply(dot(v, u), u)
            w = subtract(w, proj)
        w_norm = norm(w)
        if w_norm > 1e-10:
            orthonormal.append(scalar_multiply(1 / w_norm, w))
    return orthonormal

# Example: Non-orthonormal vectors
V = [
    [1, 1, 0],
    [1, 0, 1],
    [0, 1, 1]
]

print("Original (non-orthonormal) vectors:")
for v in V:
    print(v)

# Apply Gram-Schmidt
Q = gram_schmidt(V)

print("\nOrthonormal basis vectors:")
for q in Q:
    print(q)

# Check dot products
print("\nDot products between orthonormal vectors:")
for i in range(len(Q)):
    for j in range(len(Q)):
        print(f"dot(Q[{i}], Q[{j}]) = {dot(Q[i], Q[j]):.4f}")
