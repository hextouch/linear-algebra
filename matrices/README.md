# Matrices

Matrices are rectangular arrays of numbers that serve as fundamental tools for representing linear transformations and solving systems of equations.

## Key Concepts

### Matrix Basics
- **Definition**: m×n array of scalars
- **Elements**: Individual entries aᵢⱼ (row i, column j)
- **Square matrix**: m = n (same number of rows and columns)
- **Vector**: Special case (n×1 or 1×n matrix)

### Matrix Operations
- **Addition**: Element-wise addition (same dimensions)
- **Scalar multiplication**: Multiply each element by scalar
- **Matrix multiplication**: Row-column dot products
- **Transpose**: Aᵀ (flip rows and columns)
- **Conjugate transpose**: A* (transpose + complex conjugate)

### Special Matrices
- **Zero matrix**: All elements are zero
- **Identity matrix**: Iᵢⱼ = 1 if i=j, 0 otherwise
- **Diagonal matrix**: Non-zero elements only on main diagonal
- **Upper/Lower triangular**: Zero below/above main diagonal
- **Symmetric**: A = Aᵀ
- **Skew-symmetric**: A = -Aᵀ
- **Orthogonal**: AᵀA = I
- **Unitary**: A*A = I

### Matrix Inverse
- **Definition**: A⁻¹ such that AA⁻¹ = A⁻¹A = I
- **Existence**: Matrix must be square and non-singular
- **Properties**: (AB)⁻¹ = B⁻¹A⁻¹, (Aᵀ)⁻¹ = (A⁻¹)ᵀ
- **Computing**: Gaussian elimination, adjugate method

### Determinant
- **Definition**: Scalar value associated with square matrices
- **2×2 case**: det(A) = ad - bc
- **Properties**: det(AB) = det(A)det(B)
- **Geometric meaning**: Scaling factor for area/volume
- **Singular matrix**: det(A) = 0 (no inverse exists)

### Rank and Nullity
- **Column rank**: Dimension of column space
- **Row rank**: Dimension of row space
- **Rank-nullity theorem**: rank(A) + nullity(A) = n
- **Full rank**: rank(A) = min(m,n)

## Matrix Factorizations Preview

### Elementary Operations
- **Row operations**: Scaling, swapping, adding rows
- **Elementary matrices**: Represent single row operations
- **Row echelon form**: Upper triangular with pivots
- **Reduced row echelon form**: Simplified further

### Important Properties
- **Associativity**: (AB)C = A(BC)
- **Distributivity**: A(B + C) = AB + AC
- **Non-commutativity**: Generally AB ≠ BA
- **Transpose rules**: (AB)ᵀ = BᵀAᵀ

## Applications

### Linear Systems
- **Coefficient matrix**: Ax = b representation
- **Augmented matrix**: [A|b] for solving systems
- **Matrix equation**: Compact representation

### Transformations
- **Rotation matrices**: Rotate vectors in space
- **Scaling matrices**: Change vector magnitudes
- **Reflection matrices**: Mirror vectors across planes
- **Projection matrices**: Project onto subspaces

### Data Representation
- **Data matrices**: Rows as observations, columns as features
- **Adjacency matrices**: Represent graph connections
- **Covariance matrices**: Statistical relationships
- **Confusion matrices**: Classification results

## Computational Aspects

### Algorithms
- **Matrix multiplication**: O(n³) naive, faster with Strassen
- **LU decomposition**: Solve systems efficiently
- **QR decomposition**: Orthogonal factorization
- **SVD**: Singular value decomposition

### Numerical Considerations
- **Condition number**: Sensitivity to perturbations
- **Numerical stability**: Avoiding amplification of errors
- **Sparse matrices**: Efficient storage for mostly zero matrices
- **Iterative methods**: For large systems