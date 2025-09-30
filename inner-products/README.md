# Inner Products and Orthogonality

Inner products provide a way to measure lengths, angles, and orthogonality in vector spaces, forming the foundation for geometric interpretations of linear algebra.

## Inner Product Spaces

### Definition
An **inner product** on a vector space V is a function ⟨·,·⟩: V × V → ℝ (or ℂ) satisfying:

1. **Linearity in first argument**: ⟨au + bv, w⟩ = a⟨u,w⟩ + b⟨v,w⟩
2. **Symmetry** (real) or **Conjugate symmetry** (complex): ⟨u,v⟩ = ⟨v,u⟩ or ⟨u,v⟩ = ⟨v,u⟩*
3. **Positive definiteness**: ⟨v,v⟩ > 0 for v ≠ 0

### Standard Examples

#### Euclidean Inner Product (ℝⁿ)
**⟨x,y⟩ = x·y = x₁y₁ + x₂y₂ + ... + xₙyₙ**
- **Dot product**: Most common inner product
- **Geometric interpretation**: ⟨x,y⟩ = ‖x‖‖y‖cos(θ)
- **Matrix form**: ⟨x,y⟩ = xᵀy

#### Complex Inner Product (ℂⁿ)
**⟨x,y⟩ = x₁ȳ₁ + x₂ȳ₂ + ... + xₙȳₙ**
- **Hermitian form**: ⟨x,y⟩ = x*y
- **Complex conjugate**: Ensures positive definiteness
- **Matrix form**: ⟨x,y⟩ = x*y

#### Function Spaces
**⟨f,g⟩ = ∫ₐᵇ f(x)g(x)dx** (L² inner product)
- **Continuous functions**: C[a,b]
- **Square integrable**: L²[a,b]
- **Weighted inner products**: ∫ f(x)g(x)w(x)dx

### Induced Norm
**‖v‖ = √⟨v,v⟩**
- **Length**: Geometric measure of vector magnitude
- **Positive**: ‖v‖ ≥ 0, with equality iff v = 0
- **Homogeneous**: ‖cv‖ = |c|‖v‖
- **Triangle inequality**: ‖u + v‖ ≤ ‖u‖ + ‖v‖

## Orthogonality

### Orthogonal Vectors
**⟨u,v⟩ = 0** means u and v are **orthogonal** (u ⊥ v)

#### Properties
- **Pythagorean theorem**: ‖u + v‖² = ‖u‖² + ‖v‖² when u ⊥ v
- **Linear independence**: Orthogonal non-zero vectors are linearly independent
- **Orthogonal complement**: V⊥ = {v : ⟨v,w⟩ = 0 for all w ∈ V}

### Orthogonal Sets
A set {v₁, v₂, ..., vₖ} is **orthogonal** if vᵢ ⊥ vⱼ for i ≠ j

#### Orthonormal Sets
**Orthogonal** + **unit vectors**: ‖vᵢ‖ = 1 for all i
- **Orthonormal basis**: Simplifies computations
- **Coordinate formula**: x = ⟨x,v₁⟩v₁ + ⟨x,v₂⟩v₂ + ... + ⟨x,vₙ⟩vₙ
- **Parseval's identity**: ‖x‖² = |⟨x,v₁⟩|² + |⟨x,v₂⟩|² + ... + |⟨x,vₙ⟩|²

## Gram-Schmidt Process

### Algorithm
Transform linearly independent set {u₁, u₂, ..., uₖ} into orthonormal set {q₁, q₂, ..., qₖ}:

1. **v₁ = u₁**, **q₁ = v₁/‖v₁‖**
2. **v₂ = u₂ - ⟨u₂,q₁⟩q₁**, **q₂ = v₂/‖v₂‖**
3. **v₃ = u₃ - ⟨u₃,q₁⟩q₁ - ⟨u₃,q₂⟩q₂**, **q₃ = v₃/‖v₃‖**
4. Continue for all vectors...

### Properties
- **Span preservation**: span{q₁, ..., qₖ} = span{u₁, ..., uₖ}
- **Incremental**: Each qₖ is orthogonal to previous q's
- **Matrix form**: A = QR (QR decomposition)
- **Numerical stability**: Modified Gram-Schmidt is more stable

### Applications
- **QR decomposition**: Fundamental matrix factorization
- **Orthonormal bases**: Simplify many computations
- **Least squares**: Solve overdetermined systems
- **Function approximation**: Orthogonal polynomials

## Orthogonal Projections

### Projection onto a Line
Project vector **b** onto line spanned by **a**:
**proj_a(b) = (⟨b,a⟩/⟨a,a⟩)a = (⟨b,a⟩/‖a‖²)a**

#### Properties
- **Closest point**: Minimizes distance from b to line
- **Orthogonal component**: b - proj_a(b) ⊥ a
- **Idempotent**: proj_a(proj_a(b)) = proj_a(b)

### Projection onto a Subspace
Project **b** onto subspace W spanned by {u₁, u₂, ..., uₖ}:

#### Orthonormal Basis Case
If {q₁, q₂, ..., qₖ} is orthonormal basis for W:
**proj_W(b) = ⟨b,q₁⟩q₁ + ⟨b,q₂⟩q₂ + ... + ⟨b,qₖ⟩qₖ**

#### Matrix Form
**P = QQᵀ** where Q has orthonormal columns
- **Projection matrix**: Pb gives projection of b onto col(Q)
- **Properties**: P² = P, Pᵀ = P (idempotent and symmetric)
- **Complementary projection**: I - P projects onto orthogonal complement

### Best Approximation Theorem
**proj_W(b)** is the **closest point** in W to b:
‖b - proj_W(b)‖ ≤ ‖b - w‖ for any w ∈ W

## Orthogonal Matrices

### Definition
Square matrix Q is **orthogonal** if **QᵀQ = I**
- **Equivalent conditions**: Q⁻¹ = Qᵀ, columns form orthonormal basis
- **Unitary matrices**: Complex analog where Q*Q = I

### Properties
- **Norm preservation**: ‖Qx‖ = ‖x‖ (isometry)
- **Angle preservation**: ⟨Qx,Qy⟩ = ⟨x,y⟩
- **Determinant**: det(Q) = ±1
- **Eigenvalues**: All eigenvalues have modulus 1

### Examples
#### Rotation Matrices (2D)
```
R(θ) = [cos θ  -sin θ]
       [sin θ   cos θ]
```

#### Reflection Matrices
- **Householder reflectors**: H = I - 2uuᵀ/‖u‖²
- **Reflect across hyperplane**: Orthogonal to vector u
- **Applications**: QR decomposition, eigenvalue algorithms

#### Permutation Matrices
- **Row/column reordering**: Orthogonal by construction
- **Sparse orthogonal**: Most entries are zero
- **Applications**: Pivoting in numerical algorithms

## Advanced Topics

### Spectral Theorem for Symmetric Matrices
Any symmetric matrix A can be written as:
**A = QΛQᵀ**
where Q is orthogonal and Λ is diagonal

#### Consequences
- **Real eigenvalues**: All eigenvalues of symmetric matrices are real
- **Orthogonal eigenvectors**: For distinct eigenvalues
- **Spectral decomposition**: A = λ₁q₁q₁ᵀ + λ₂q₂q₂ᵀ + ... + λₙqₙqₙᵀ

### Singular Value Decomposition (SVD)
**A = UΣVᵀ** where U and V are orthogonal
- **Geometric interpretation**: Composition of rotations and scaling
- **Principal components**: V columns give principal directions
- **Applications**: Data compression, noise reduction, pseudoinverse

### Least Squares and Normal Equations

#### Overdetermined Systems
For Ax = b where A is m×n with m > n:
- **Normal equations**: AᵀAx = Aᵀb
- **Orthogonal projection**: Solution projects b onto col(A)
- **QR method**: More numerically stable than normal equations

#### Geometric Interpretation
- **Closest solution**: Minimizes ‖Ax - b‖²
- **Orthogonality condition**: Residual r = b - Ax ⊥ col(A)
- **Projection**: Ax = proj_{col(A)}(b)

## Applications

### Data Analysis
- **Principal Component Analysis**: Find orthogonal directions of maximum variance
- **Factor analysis**: Orthogonal rotation of factors
- **Whitening transformation**: Decorrelate data variables
- **Dimensionality reduction**: Project onto lower-dimensional subspaces

### Signal Processing
- **Fourier analysis**: Orthogonal frequency components
- **Wavelet transforms**: Orthogonal time-frequency analysis
- **Filter design**: Orthogonal filter banks
- **Noise reduction**: Project onto signal subspace

### Computer Graphics
- **Coordinate transformations**: Orthogonal matrices preserve shape
- **Camera orientation**: Orthonormal coordinate frames
- **Animation**: Interpolation between orientations
- **Lighting calculations**: Normal vectors and reflection

### Quantum Mechanics
- **State vectors**: Unit vectors in Hilbert space
- **Observables**: Self-adjoint operators
- **Unitary evolution**: Orthogonal transformations preserve probabilities
- **Measurement**: Projection onto eigenspaces

### Optimization
- **Quadratic forms**: Orthogonal diagonalization
- **Constrained optimization**: Lagrange multipliers and projections
- **Gradient methods**: Orthogonal search directions
- **Trust region methods**: Orthogonal model reductions

### Statistics
- **Regression analysis**: Orthogonal decomposition of variance
- **ANOVA**: Orthogonal contrasts
- **Experimental design**: Orthogonal factor combinations
- **Multivariate analysis**: Canonical correlations