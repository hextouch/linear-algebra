# Eigenvalues and Eigenvectors

Eigenvalues and eigenvectors reveal the fundamental directions and scaling factors of linear transformations, providing insight into the geometric and algebraic structure of matrices.

## Fundamental Definitions

### Eigenvalues and Eigenvectors
For a square matrix A and non-zero vector v:
- **Eigenvalue equation**: Av = λv
- **λ**: eigenvalue (scalar)
- **v**: eigenvector (direction preserved by transformation)
- **Eigenspace**: E_λ = {v : Av = λv} (all eigenvectors for λ)

### Characteristic Equation
- **Characteristic polynomial**: p(λ) = det(A - λI)
- **Eigenvalues**: Roots of characteristic polynomial
- **Degree**: n for n×n matrix (counting multiplicities)
- **Fundamental theorem**: n complex eigenvalues (with multiplicity)

## Key Properties

### Multiplicities
- **Algebraic multiplicity**: Multiplicity as root of characteristic polynomial
- **Geometric multiplicity**: dim(eigenspace) = dim(E_λ)
- **Important inequality**: geometric ≤ algebraic multiplicity
- **Simple eigenvalue**: Algebraic multiplicity = 1

### Eigenspace Properties
- **Subspace**: Each eigenspace E_λ is a subspace
- **Linear independence**: Eigenvectors for distinct eigenvalues are linearly independent
- **Direct sum**: V = E_λ₁ ⊕ E_λ₂ ⊕ ... ⊕ E_λₖ (when diagonalizable)

## Special Cases and Types

### Real vs Complex Eigenvalues
- **Real matrices**: May have complex eigenvalues (in conjugate pairs)
- **Complex conjugates**: If λ = a + bi is eigenvalue, so is ā = a - bi
- **Real eigenspaces**: For real eigenvalues of real matrices
- **Complex eigenspaces**: Require complex arithmetic

### Special Matrix Types

#### Symmetric Matrices
- **Real eigenvalues**: All eigenvalues are real
- **Orthogonal eigenvectors**: For distinct eigenvalues
- **Spectral theorem**: Orthogonally diagonalizable
- **Applications**: Quadratic forms, optimization

#### Skew-Symmetric Matrices
- **Pure imaginary eigenvalues**: λ = ±iα
- **Zero eigenvalue**: Always present for odd dimension
- **Orthogonal eigenvectors**: Natural orthogonality

#### Positive Definite Matrices
- **Positive eigenvalues**: All λ > 0
- **Applications**: Covariance matrices, optimization
- **Cholesky decomposition**: A = LLᵀ

## Diagonalization

### Diagonalizable Matrices
- **Definition**: A = PDP⁻¹ where D is diagonal
- **Condition**: Geometric multiplicity = algebraic multiplicity for all eigenvalues
- **Eigenvalue matrix D**: Diagonal entries are eigenvalues
- **Eigenvector matrix P**: Columns are corresponding eigenvectors

### Diagonalization Process
1. Find characteristic polynomial: det(A - λI)
2. Solve for eigenvalues: roots of characteristic polynomial
3. Find eigenvectors: solve (A - λI)v = 0 for each λ
4. Check diagonalizability: sufficient eigenvectors?
5. Form matrices P and D

### Benefits of Diagonalization
- **Powers**: Aⁿ = PDⁿP⁻¹ (easy to compute)
- **Matrix functions**: f(A) = Pf(D)P⁻¹
- **System solving**: Easier in diagonal form
- **Understanding behavior**: Eigenvalues determine long-term behavior

## Advanced Topics

### Jordan Normal Form
- **Non-diagonalizable matrices**: Alternative canonical form
- **Jordan blocks**: Nearly diagonal structure
- **Generalized eigenvectors**: Fill in missing eigenvectors
- **Applications**: Differential equations, matrix functions

### Spectral Theory
- **Spectral theorem**: For normal matrices (AAᵀ = AᵀA)
- **Unitary diagonalization**: A = UDU* with U unitary
- **Spectral decomposition**: A = λ₁P₁ + λ₂P₂ + ... + λₖPₖ
- **Functional calculus**: Define functions of matrices

### Perturbation Theory
- **Eigenvalue sensitivity**: How eigenvalues change with matrix perturbations
- **Condition numbers**: Measure sensitivity to perturbations
- **Pseudospectra**: Sets of approximate eigenvalues
- **Applications**: Numerical stability, robust control

## Computational Methods

### Direct Methods
- **Characteristic polynomial**: Exact for small matrices
- **Symbolic computation**: Using computer algebra systems
- **Limitations**: Numerical instability for large matrices

### Iterative Methods
- **Power method**: Find dominant eigenvalue
- **QR algorithm**: Find all eigenvalues
- **Arnoldi iteration**: For large sparse matrices
- **Jacobi method**: For symmetric matrices

### Software Tools
- **NumPy**: `numpy.linalg.eig()`, `numpy.linalg.eigvals()`
- **SciPy**: Specialized algorithms for different matrix types
- **MATLAB**: `eig()`, `eigs()` functions
- **Mathematica**: `Eigenvalues[]`, `Eigenvectors[]`

## Applications

### Principal Component Analysis (PCA)
- **Covariance matrix**: Eigenvalues give variance in principal directions
- **Dimensionality reduction**: Keep largest eigenvalue directions
- **Data compression**: Reduce data dimensions while preserving information
- **Pattern recognition**: Feature extraction and visualization

### Dynamical Systems
- **Stability analysis**: Eigenvalues determine stability of equilibria
- **Phase portraits**: Eigenvectors give natural coordinate directions
- **Linear ODEs**: Solution structure determined by eigenvalues
- **Discrete systems**: xₙ₊₁ = Axₙ behavior from eigenvalues

### Quantum Mechanics
- **Observable operators**: Hermitian matrices with real eigenvalues
- **Energy levels**: Eigenvalues of Hamiltonian operator
- **State vectors**: Eigenvectors represent possible states
- **Measurement**: Collapse to eigenvector states

### Vibration Analysis
- **Natural frequencies**: Square roots of eigenvalues
- **Mode shapes**: Eigenvectors show vibration patterns
- **Structural analysis**: Building and bridge dynamics
- **Mechanical systems**: Springs, masses, and oscillations

### Economics and Finance
- **Portfolio theory**: Covariance matrix eigenvalues for risk
- **Economic models**: Stability of equilibria
- **Market dynamics**: Principal components of price movements
- **Risk management**: Eigenvalue-based measures

### Machine Learning
- **Spectral clustering**: Use eigenvalues of similarity matrices
- **Kernel methods**: Eigendecomposition of kernel matrices
- **Neural networks**: Weight matrix analysis
- **Recommendation systems**: Latent factor models