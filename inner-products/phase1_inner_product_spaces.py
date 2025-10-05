"""
Inner Product Implementation - Pure Python (No Libraries)
"""
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive plots
import matplotlib.pyplot as plt

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

def plot_vectors_2d(vectors, labels=None, colors=None, title="Vectors"):
    """Plot 2D vectors"""
    plt.figure(figsize=(8, 8))
    
    if colors is None:
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    
    if labels is None:
        labels = [f'v{i+1}' for i in range(len(vectors))]
    
    # Plot vectors as arrows from origin
    for i, v in enumerate(vectors):
        if len(v) >= 2:
            plt.arrow(0, 0, v[0], v[1], 
                     head_width=0.1, head_length=0.1, 
                     fc=colors[i % len(colors)], 
                     ec=colors[i % len(colors)],
                     linewidth=2, label=labels[i])
    
    # Set equal aspect and grid
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linewidth=0.5)
    plt.axvline(x=0, color='k', linewidth=0.5)
    
    # Set reasonable limits
    all_coords = []
    for v in vectors:
        if len(v) >= 2:
            all_coords.extend([v[0], v[1]])
    
    if all_coords:
        margin = max(abs(min(all_coords)), abs(max(all_coords))) * 0.1
        plt.xlim(min(all_coords) - margin, max(all_coords) + margin)
        plt.ylim(min(all_coords) - margin, max(all_coords) + margin)
    
    plt.legend()
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def plot_projection_2d(original, target, projection, title="Vector Projection"):
    """Plot vector projection in 2D"""
    plt.figure(figsize=(8, 8))
    
    # Original vector
    plt.arrow(0, 0, original[0], original[1], 
             head_width=0.1, head_length=0.1, 
             fc='blue', ec='blue', linewidth=2, label='Original vector v')
    
    # Target vector (what we project onto)
    plt.arrow(0, 0, target[0], target[1], 
             head_width=0.1, head_length=0.1, 
             fc='red', ec='red', linewidth=2, label='Target vector u')
    
    # Projection
    plt.arrow(0, 0, projection[0], projection[1], 
             head_width=0.08, head_length=0.08, 
             fc='green', ec='green', linewidth=2, label='Projection proj_u(v)')
    
    # Residual (perpendicular component)
    residual = [original[i] - projection[i] for i in range(2)]
    plt.arrow(projection[0], projection[1], residual[0], residual[1], 
             head_width=0.05, head_length=0.05, 
             fc='orange', ec='orange', linewidth=1.5, 
             linestyle='--', label='Residual (⊥ component)')
    
    # Dashed line showing projection construction
    plt.plot([original[0], projection[0]], [original[1], projection[1]], 
             'k--', alpha=0.5, linewidth=1)
    
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linewidth=0.5)
    plt.axvline(x=0, color='k', linewidth=0.5)
    
    # Set limits
    all_coords = [original[0], original[1], target[0], target[1], 
                  projection[0], projection[1]]
    margin = max(abs(min(all_coords)), abs(max(all_coords))) * 0.2
    plt.xlim(min(all_coords) - margin, max(all_coords) + margin)
    plt.ylim(min(all_coords) - margin, max(all_coords) + margin)
    
    plt.legend()
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def plot_gram_schmidt_2d(original_vectors, orthogonal_vectors, orthonormal_vectors):
    """Plot Gram-Schmidt process for 2D vectors"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    colors = ['red', 'blue', 'green', 'orange']
    
    # Plot 1: Original vectors
    ax = axes[0]
    for i, v in enumerate(original_vectors):
        if len(v) >= 2:
            ax.arrow(0, 0, v[0], v[1], 
                    head_width=0.1, head_length=0.1, 
                    fc=colors[i], ec=colors[i], linewidth=2, 
                    label=f'v{i+1} = [{v[0]}, {v[1]}]')
    
    ax.set_title('Original Vectors')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    # Plot 2: Orthogonal vectors
    ax = axes[1]
    for i, v in enumerate(orthogonal_vectors):
        if len(v) >= 2:
            ax.arrow(0, 0, v[0], v[1], 
                    head_width=0.1, head_length=0.1, 
                    fc=colors[i], ec=colors[i], linewidth=2, 
                    label=f'u{i+1} = [{v[0]:.2f}, {v[1]:.2f}]')
    
    ax.set_title('Orthogonal Vectors')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    # Plot 3: Orthonormal vectors
    ax = axes[2]
    for i, v in enumerate(orthonormal_vectors):
        if len(v) >= 2:
            ax.arrow(0, 0, v[0], v[1], 
                    head_width=0.1, head_length=0.1, 
                    fc=colors[i], ec=colors[i], linewidth=2, 
                    label=f'q{i+1} = [{v[0]:.2f}, {v[1]:.2f}]')
    
    # Draw unit circle to show normalization
    theta = [i * 0.01 for i in range(628)]  # 0 to 2π
    unit_x = [1 * (theta_val ** 2 - (theta_val ** 4)/24 + (theta_val ** 6)/720) if abs(theta_val) < 1.57 
              else 1 * (-1 if theta_val > 3.14 else 1) * ((3.14159 - theta_val) ** 2 - ((3.14159 - theta_val) ** 4)/24) ** 0.5 
              for theta_val in theta]
    unit_y = [1 * (theta_val - (theta_val ** 3)/6 + (theta_val ** 5)/120) if abs(theta_val) < 1.57 
              else 1 * (1 if theta_val < 3.14 else -1) * (1 - ((3.14159 - abs(theta_val - 3.14159)) ** 2)/2) ** 0.5 
              for theta_val in theta]
    
    # Simple unit circle approximation
    circle_points = 64
    unit_circle_x = []
    unit_circle_y = []
    for i in range(circle_points + 1):
        angle = 2 * 3.14159 * i / circle_points
        # Use polynomial approximation for cos and sin
        cos_val = 1 - angle*angle/2 + angle**4/24 - angle**6/720
        sin_val = angle - angle**3/6 + angle**5/120 - angle**7/5040
        if angle > 3.14159/2 and angle <= 3.14159:
            cos_val = -(1 - (3.14159 - angle)**2/2 + (3.14159 - angle)**4/24)
        elif angle > 3.14159 and angle <= 3*3.14159/2:
            cos_val = -(1 - (angle - 3.14159)**2/2 + (angle - 3.14159)**4/24)
            sin_val = -((angle - 3.14159) - (angle - 3.14159)**3/6 + (angle - 3.14159)**5/120)
        elif angle > 3*3.14159/2:
            cos_val = 1 - (2*3.14159 - angle)**2/2 + (2*3.14159 - angle)**4/24
            sin_val = -((2*3.14159 - angle) - (2*3.14159 - angle)**3/6)
        
        unit_circle_x.append(cos_val)
        unit_circle_y.append(sin_val)
    
    ax.plot(unit_circle_x, unit_circle_y, 'k--', alpha=0.3, linewidth=1, label='Unit circle')
    
    ax.set_title('Orthonormal Vectors')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    plt.tight_layout()
    plt.show()

def test_with_plots():
    """Run tests with visualizations"""
    print("=== VISUAL DEMONSTRATIONS ===")
    
    # Test 1: Basic projection
    print("\n1. Vector Projection:")
    v = [3, 2]
    u = [2, 0]
    proj = project_onto(v, u)
    
    print(f"Project {v} onto {u}")
    print(f"Projection: {proj}")
    print(f"Residual: {[v[i] - proj[i] for i in range(len(v))]}")
    
    plot_projection_2d(v, u, proj, "Vector Projection Example")
    
    # Test 2: Gram-Schmidt in 2D
    print("\n2. Gram-Schmidt Process:")
    original = [
        [2, 1],
        [1, 2]
    ]
    
    orthogonal = gram_schmidt(original)
    orthonormal = gram_schmidt_orthonormal(original)
    
    print("Original vectors:", original)
    print("Orthogonal:", [[round(x, 3) for x in v] for v in orthogonal])
    print("Orthonormal:", [[round(x, 3) for x in v] for v in orthonormal])
    
    plot_gram_schmidt_2d(original, orthogonal, orthonormal)
    
    # Test 3: Another projection example
    print("\n3. Another Projection:")
    v2 = [4, 3]
    u2 = [1, 1]
    proj2 = project_onto(v2, u2)
    
    print(f"Project {v2} onto {u2}")
    print(f"Projection: {[round(x, 3) for x in proj2]}")
    
    plot_projection_2d(v2, u2, proj2, "Projection onto Diagonal Line")
    
    # Test 4: Three vectors Gram-Schmidt (show first 2 in 2D)
    print("\n4. Gram-Schmidt with 3 vectors (showing first 2):")
    vectors_3d = [
        [1, 0],
        [1, 1]
    ]
    
    orth_3d = gram_schmidt(vectors_3d)
    orthonorm_3d = gram_schmidt_orthonormal(vectors_3d)
    
    print("Original:", vectors_3d)
    print("Orthogonal:", [[round(x, 3) for x in v] for v in orth_3d])
    print("Orthonormal:", [[round(x, 3) for x in v] for v in orthonorm_3d])
    
    plot_gram_schmidt_2d(vectors_3d, orth_3d, orthonorm_3d)

def run_all_tests():
    """Run all test cases"""
    test_basic_operations()
    test_orthogonality()
    test_projections()
    test_gram_schmidt()
    test_projection_matrix()

def main():
    """Main function with user choice"""
    print("Choose demonstration:")
    print("1. Run all tests (no plots)")
    print("2. Run visual demonstrations (with plots)")
    print("3. Both")
    
    try:
        choice = input("Enter choice (1-3): ").strip()
        
        if choice == '1':
            run_all_tests()
        elif choice == '2':
            test_with_plots()
        elif choice == '3':
            run_all_tests()
            print("\n" + "="*50)
            print("Now showing visual demonstrations...")
            print("="*50)
            test_with_plots()
        else:
            print("Invalid choice, running all tests...")
            run_all_tests()
            
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {e}")
        print("Running default tests...")
        run_all_tests()

if __name__ == "__main__":
    main()