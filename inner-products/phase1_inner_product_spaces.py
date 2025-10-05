"""
Inner Product Implementation - Pure Python (No Libraries)
"""

def inner_product(x, y):
    """Standard Euclidean inner product"""
    if len(x) != len(y):
        raise ValueError("Vectors must have same length")
    
    result = 0
    for i in range(len(x)):
        result += x[i] * y[i]
    return result

def norm(v):
    """Vector norm from inner product"""
    return inner_product(v, v) ** 0.5

def angle_between_vectors(x, y):
    """Angle in radians between two vectors"""
    dot_product = inner_product(x, y)
    norm_x = norm(x)
    norm_y = norm(y)
    
    if norm_x == 0 or norm_y == 0:
        return 0
    
    cos_theta = dot_product / (norm_x * norm_y)
    
    # Clamp to [-1, 1] to avoid numerical errors
    if cos_theta > 1:
        cos_theta = 1
    elif cos_theta < -1:
        cos_theta = -1
    
    return acos(cos_theta)

def acos(x):
    """Approximate arccos using Taylor series"""
    if x == 1:
        return 0
    if x == -1:
        return 3.14159265359
    if x == 0:
        return 1.5707963268
    
    # Use identity: arccos(x) = π/2 - arcsin(x)
    return 1.5707963268 - asin(x)

def asin(x):
    """Approximate arcsin using Taylor series"""
    if abs(x) > 1:
        raise ValueError("Input must be between -1 and 1")
    
    # Taylor series: arcsin(x) = x + x³/6 + 3x⁵/40 + 5x⁷/112 + ...
    result = x
    term = x
    
    for n in range(1, 20):
        term = term * x * x * (2*n - 1) / (2*n)
        result += term / (2*n + 1)
    
    return result

def is_orthogonal(x, y, tolerance=1e-10):
    """Check if vectors are orthogonal"""
    return abs(inner_product(x, y)) < tolerance

def project_onto(v, u):
    """Project vector v onto vector u"""
    u_dot_u = inner_product(u, u)
    if u_dot_u == 0:
        return [0] * len(v)
    
    scalar = inner_product(v, u) / u_dot_u
    return [scalar * u[i] for i in range(len(u))]

def gram_schmidt_step(vectors, orthogonal_vectors, index):
    """Single step of Gram-Schmidt orthogonalization"""
    v = vectors[index][:]  # Copy the vector
    
    # Subtract projections onto all previous orthogonal vectors
    for j in range(len(orthogonal_vectors)):
        proj = project_onto(v, orthogonal_vectors[j])
        for i in range(len(v)):
            v[i] -= proj[i]
    
    return v

def normalize_vector(v):
    """Make vector unit length"""
    v_norm = norm(v)
    if v_norm == 0:
        return v[:]
    return [v[i] / v_norm for i in range(len(v))]

def gram_schmidt(vectors):
    """Complete Gram-Schmidt orthogonalization"""
    if not vectors:
        return []
    
    orthogonal = []
    
    for i in range(len(vectors)):
        # Get orthogonal vector
        v_orth = gram_schmidt_step(vectors, orthogonal, i)
        
        # Skip if vector becomes zero (linearly dependent)
        if norm(v_orth) > 1e-10:
            orthogonal.append(v_orth)
    
    return orthogonal

def gram_schmidt_orthonormal(vectors):
    """Gram-Schmidt with normalization"""
    orthogonal = gram_schmidt(vectors)
    return [normalize_vector(v) for v in orthogonal]

def matrix_vector_multiply(matrix, vector):
    """Multiply matrix by vector"""
    result = []
    for i in range(len(matrix)):
        sum_val = 0
        for j in range(len(vector)):
            sum_val += matrix[i][j] * vector[j]
        result.append(sum_val)
    return result

def transpose_matrix(matrix):
    """Matrix transpose"""
    if not matrix or not matrix[0]:
        return []
    
    rows = len(matrix)
    cols = len(matrix[0])
    
    transposed = []
    for j in range(cols):
        row = []
        for i in range(rows):
            row.append(matrix[i][j])
        transposed.append(row)
    
    return transposed

def matrix_multiply(A, B):
    """Matrix multiplication"""
    if not A or not A[0] or not B or not B[0]:
        return []
    
    rows_A = len(A)
    cols_A = len(A[0])
    rows_B = len(B)
    cols_B = len(B[0])
    
    if cols_A != rows_B:
        raise ValueError("Matrix dimensions don't match for multiplication")
    
    result = []
    for i in range(rows_A):
        row = []
        for j in range(cols_B):
            sum_val = 0
            for k in range(cols_A):
                sum_val += A[i][k] * B[k][j]
            row.append(sum_val)
        result.append(row)
    
    return result

def create_projection_matrix(orthonormal_vectors):
    """Create projection matrix P = QQ^T"""
    if not orthonormal_vectors:
        return []
    
    n = len(orthonormal_vectors[0])
    Q = transpose_matrix(orthonormal_vectors)  # Q has orthonormal columns
    Q_T = orthonormal_vectors  # Q transpose
    
    return matrix_multiply(Q, Q_T)

def apply_projection(projection_matrix, vector):
    """Apply projection matrix to vector"""
    return matrix_vector_multiply(projection_matrix, vector)

# Test functions
def test_basic_operations():
    """Test basic inner product operations"""
    print("=== Basic Operations ===")
    
    v1 = [3, 4]
    v2 = [1, 2]
    
    print(f"v1 = {v1}")
    print(f"v2 = {v2}")
    print(f"⟨v1,v2⟩ = {inner_product(v1, v2)}")
    print(f"||v1|| = {norm(v1):.3f}")
    print(f"||v2|| = {norm(v2):.3f}")
    print(f"angle = {angle_between_vectors(v1, v2):.3f} rad")
    print()

def test_orthogonality():
    """Test orthogonal vectors"""
    print("=== Orthogonality ===")
    
    # Orthogonal vectors
    u1 = [1, 0]
    u2 = [0, 1]
    u3 = [3, 4]
    u4 = [-4, 3]
    
    print(f"u1 = {u1}, u2 = {u2}")
    print(f"⟨u1,u2⟩ = {inner_product(u1, u2)} (orthogonal: {is_orthogonal(u1, u2)})")
    
    print(f"u3 = {u3}, u4 = {u4}")
    print(f"⟨u3,u4⟩ = {inner_product(u3, u4)} (orthogonal: {is_orthogonal(u3, u4)})")
    print()

def test_projections():
    """Test vector projections"""
    print("=== Projections ===")
    
    v = [2, 3]
    u = [1, 0]
    
    proj = project_onto(v, u)
    print(f"Project {v} onto {u}: {proj}")
    
    # Verify orthogonality of residual
    residual = [v[i] - proj[i] for i in range(len(v))]
    print(f"Residual: {residual}")
    print(f"Residual ⊥ u: {is_orthogonal(residual, u)}")
    print()

def test_gram_schmidt():
    """Test Gram-Schmidt process"""
    print("=== Gram-Schmidt ===")
    
    vectors = [
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1]
    ]
    
    print("Original vectors:")
    for i, v in enumerate(vectors):
        print(f"  v{i+1} = {v}")
    
    orthogonal = gram_schmidt(vectors)
    print("Orthogonal vectors:")
    for i, v in enumerate(orthogonal):
        print(f"  u{i+1} = [{v[0]:.3f}, {v[1]:.3f}, {v[2]:.3f}]")
    
    orthonormal = gram_schmidt_orthonormal(vectors)
    print("Orthonormal vectors:")
    for i, v in enumerate(orthonormal):
        print(f"  q{i+1} = [{v[0]:.3f}, {v[1]:.3f}, {v[2]:.3f}]")
    
    # Verify orthogonality
    print("Verification:")
    for i in range(len(orthonormal)):
        for j in range(i+1, len(orthonormal)):
            dot = inner_product(orthonormal[i], orthonormal[j])
            print(f"  q{i+1} · q{j+1} = {dot:.6f}")
    print()

def test_projection_matrix():
    """Test projection matrices"""
    print("=== Projection Matrix ===")
    
    # Create orthonormal basis for a plane
    q1 = [1, 0, 0]
    q2 = [0, 1, 0]
    
    P = create_projection_matrix([q1, q2])
    print("Projection matrix onto xy-plane:")
    for row in P:
        print(f"  {[f'{x:.1f}' for x in row]}")
    
    # Test projection
    v = [3, 4, 5]
    proj_v = apply_projection(P, v)
    print(f"Project {v}: {[f'{x:.1f}' for x in proj_v]}")
    print()

def run_all_tests():
    """Run all test cases"""
    test_basic_operations()
    test_orthogonality()
    test_projections()
    test_gram_schmidt()
    test_projection_matrix()

if __name__ == "__main__":
    run_all_tests()