# Matrix Decompositions

Matrix decompositions break down matrices into products of simpler matrices, revealing structure and enabling efficient computation.

## LU Decomposition

### Definition
**A = LU** where:
- **L**: Lower triangular matrix with 1's on diagonal
- **U**: Upper triangular matrix
- **Interpretation**: Gaussian elimination in matrix form

### Variants
- **LU with partial pivoting**: PA = LU (P is permutation matrix)
- **LUP decomposition**: Most numerically stable
- **Crout method**: L has arbitrary diagonal
- **Doolittle method**: L has unit diagonal

### Applications
- **Solving systems**: Ax = b becomes LUx = b
- **Forward substitution**: Solve Ly = b
- **Back substitution**: Solve Ux = y
- **Determinant**: det(A) = det(L)det(U) = det(U)
- **Matrix inverse**: Solve AX = I column by column

### Computational Complexity
- **Decomposition**: O(n³/3) operations
- **Solving**: O(n²) operations per right-hand side
- **Multiple systems**: Efficient when A is fixed
- **Storage**: Can overwrite original matrix

## QR Decomposition

### Definition
**A = QR** where:
- **Q**: Orthogonal matrix (QᵀQ = I)
- **R**: Upper triangular matrix
- **Existence**: Any m×n matrix has QR decomposition

### Construction Methods

#### Gram-Schmidt Process
1. **Classical**: Orthogonalize columns sequentially
2. **Modified**: Numerically more stable version
3. **Column interpretation**: Q columns are orthonormal basis
4. **R entries**: Inner products during orthogonalization

#### Householder Reflections
- **Reflector matrices**: H = I - 2uuᵀ/‖u‖²
- **Zero elimination**: Create zeros below diagonal
- **Numerical stability**: Better than Gram-Schmidt
- **Compact storage**: Store reflector vectors

#### Givens Rotations
- **Plane rotations**: Zero individual elements
- **Sparse matrices**: Preserve sparsity better
- **Parallel computation**: More parallelizable
- **Applications**: Real-time processing

### Applications
- **Least squares**: Solve Ax ≈ b when A is overdetermined
- **QR algorithm**: Eigenvalue computation
- **Orthogonal basis**: Q provides orthonormal columns
- **Numerical stability**: Better conditioned than normal equations

## Singular Value Decomposition (SVD)

### Definition
**A = UΣVᵀ** where:
- **U**: m×m orthogonal matrix (left singular vectors)
- **Σ**: m×n diagonal matrix (singular values σ₁ ≥ σ₂ ≥ ... ≥ 0)
- **V**: n×n orthogonal matrix (right singular vectors)
- **Universality**: Every matrix has an SVD

### Key Properties
- **Singular values**: σᵢ = √λᵢ where λᵢ are eigenvalues of AᵀA
- **Rank**: Number of non-zero singular values
- **Condition number**: σ₁/σₙ (ratio of largest to smallest)
- **Frobenius norm**: ‖A‖_F = √(σ₁² + σ₂² + ... + σₙ²)

### Geometric Interpretation
- **Linear transformation**: A maps unit sphere to ellipsoid
- **Principal axes**: Singular vectors give axes directions
- **Axis lengths**: Singular values give semi-axis lengths
- **Optimal approximation**: Best rank-k approximation

### Applications

#### Data Analysis
- **Principal Component Analysis**: SVD of centered data matrix
- **Dimensionality reduction**: Keep largest singular values
- **Noise reduction**: Remove small singular values
- **Data compression**: Store only significant components

#### Numerical Linear Algebra
- **Pseudoinverse**: A⁺ = VΣ⁺Uᵀ
- **Least squares**: Minimum norm solution
- **Rank determination**: Count non-zero singular values
- **Matrix norms**: Various norms from singular values

#### Machine Learning
- **Collaborative filtering**: Recommendation systems
- **Latent semantic analysis**: Text processing
- **Image processing**: Compression and denoising
- **Feature extraction**: Dimensionality reduction

## Cholesky Decomposition

### Definition
For positive definite matrix A:
**A = LLᵀ** where L is lower triangular

### Properties
- **Existence**: Only for positive definite matrices
- **Uniqueness**: L is unique with positive diagonal
- **Efficiency**: Half the work of LU decomposition
- **Numerical stability**: No pivoting needed

### Applications
- **Covariance matrices**: Natural positive definiteness
- **Normal equations**: AᵀAx = Aᵀb in least squares
- **Simulation**: Generate correlated random variables
- **Optimization**: Quadratic programming

## Spectral Decomposition

### Eigendecomposition
For diagonalizable matrix A:
**A = PDP⁻¹** where:
- **P**: Matrix of eigenvectors
- **D**: Diagonal matrix of eigenvalues

### Spectral Theorem
For symmetric matrix A:
**A = QΛQᵀ** where:
- **Q**: Orthogonal matrix of eigenvectors
- **Λ**: Diagonal matrix of real eigenvalues
- **Orthogonality**: Eigenvectors are orthogonal

### Matrix Functions
- **Powers**: Aⁿ = QΛⁿQᵀ
- **Square root**: A^(1/2) = QΛ^(1/2)Qᵀ
- **Exponential**: e^A = Qe^ΛQᵀ
- **General functions**: f(A) = Qf(Λ)Qᵀ

## Schur Decomposition

### Definition
**A = QTQᵀ** where:
- **Q**: Orthogonal matrix
- **T**: Upper triangular (or block triangular for real matrices)
- **Diagonal elements**: Eigenvalues of A

### Properties
- **Existence**: Every square matrix has Schur decomposition
- **Real Schur form**: 2×2 blocks for complex eigenvalues
- **Numerical computation**: Basis for QR algorithm
- **Applications**: Eigenvalue algorithms

## Jordan Decomposition

### Definition
For any square matrix A:
**A = PJP⁻¹** where J is Jordan normal form

### Jordan Normal Form
- **Jordan blocks**: Nearly diagonal structure
- **Defective matrices**: When geometric < algebraic multiplicity
- **Generalized eigenvectors**: Complete the basis
- **Applications**: Differential equations, matrix functions

## Polar Decomposition

### Definition
**A = UP** where:
- **U**: Orthogonal matrix
- **P**: Positive semidefinite matrix
- **Interpretation**: Rotation followed by scaling

### Applications
- **Computer graphics**: Decompose transformations
- **Mechanics**: Deformation analysis
- **Signal processing**: Phase and magnitude separation

## Computational Considerations

### Numerical Stability
- **Condition numbers**: Sensitivity to perturbations
- **Pivoting strategies**: Improve numerical accuracy
- **Orthogonal methods**: Generally more stable
- **Error analysis**: Backward and forward error

### Efficiency
- **Operation counts**: Compare different methods
- **Memory requirements**: In-place vs. additional storage
- **Parallelization**: Exploit matrix structure
- **Sparse matrices**: Specialized algorithms

### Software Implementation
- **LAPACK**: Standard numerical linear algebra library
- **BLAS**: Basic Linear Algebra Subprograms
- **ScaLAPACK**: Parallel and distributed computing
- **GPU acceleration**: CUDA, cuBLAS, cuSOLVER