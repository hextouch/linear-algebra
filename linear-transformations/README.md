# Linear Transformations

Linear transformations are functions between vector spaces that preserve the operations of vector addition and scalar multiplication.

## Key Concepts

### Definition
A function T: V → W is linear if:
1. **Additivity**: T(u + v) = T(u) + T(v)
2. **Homogeneity**: T(cu) = cT(u)

Equivalently: **T(cu + dv) = cT(u) + dT(v)**

### Matrix Representation
- **Standard matrix**: A such that T(x) = Ax
- **Coordinate representation**: Depends on chosen bases
- **Change of basis**: Similarity transformations
- **Dimension compatibility**: m×n matrix for T: ℝⁿ → ℝᵐ

### Fundamental Subspaces

#### Kernel (Null Space)
- **Definition**: ker(T) = {v ∈ V : T(v) = 0}
- **Properties**: Always a subspace of V
- **Injectivity**: T is one-to-one iff ker(T) = {0}
- **Computation**: Solve Ax = 0

#### Image (Range)
- **Definition**: im(T) = {T(v) : v ∈ V}
- **Properties**: Subspace of W
- **Surjectivity**: T is onto iff im(T) = W
- **Column space**: im(T) = Col(A)

### Rank-Nullity Theorem
**dim(V) = rank(T) + nullity(T)**
- **Rank**: dim(im(T))
- **Nullity**: dim(ker(T))
- **Fundamental relationship**: Connects domain and range

## Types of Linear Transformations

### Geometric Transformations (ℝ² → ℝ²)

#### Rotation
```
R(θ) = [cos θ  -sin θ]
       [sin θ   cos θ]
```

#### Scaling
```
S(a,b) = [a  0]
         [0  b]
```

#### Reflection
- **Across x-axis**: [1  0; 0 -1]
- **Across y-axis**: [-1 0; 0  1]
- **Across line y=x**: [0 1; 1  0]

#### Shear
```
Shear_x(k) = [1  k]
             [0  1]
```

### Projection Transformations
- **Orthogonal projection**: Onto subspaces
- **Projection matrix**: P² = P, Pᵀ = P
- **Complementary projection**: I - P

### Important Properties

#### Invertibility
- **Invertible**: T is bijective (one-to-one and onto)
- **Square matrices**: n×n for T: ℝⁿ → ℝⁿ
- **Inverse transformation**: T⁻¹ exists iff det(A) ≠ 0

#### Composition
- **Function composition**: (S ∘ T)(v) = S(T(v))
- **Matrix multiplication**: [S ∘ T] = [S][T]
- **Non-commutativity**: Generally S ∘ T ≠ T ∘ S

## Change of Basis

### Coordinate Systems
- **Standard basis**: Natural coordinates
- **Alternative bases**: Different perspectives
- **Transition matrix**: P converts between bases
- **Similarity**: B = P⁻¹AP (same transformation, different basis)

### Applications of Change of Basis
- **Diagonalization**: Find basis where matrix is diagonal
- **Principal axes**: Natural coordinate system for quadratic forms
- **Normal modes**: Physics applications in oscillations

## Advanced Topics

### Linear Operators
- **Endomorphism**: T: V → V (same domain and codomain)
- **Eigenvalues and eigenvectors**: Special directions preserved
- **Characteristic polynomial**: det(A - λI) = 0
- **Spectral theory**: Decomposition using eigenvalues

### Functional Analysis Extensions
- **Bounded operators**: On normed spaces
- **Compact operators**: Finite-dimensional-like behavior
- **Spectral theorem**: For self-adjoint operators

## Applications

### Computer Graphics
- **3D transformations**: Rotation, scaling, translation
- **Homogeneous coordinates**: Unified representation
- **Viewing transformations**: Camera and projection matrices
- **Animation**: Interpolation between transformations

### Data Science
- **Feature transformations**: Preprocessing data
- **Dimensionality reduction**: PCA, factor analysis
- **Linear regression**: Fitting linear models
- **Classification**: Linear discriminants

### Physics and Engineering
- **Quantum mechanics**: Operators on state spaces
- **Signal processing**: Filters as linear transformations
- **Control theory**: System dynamics and stability
- **Mechanics**: Coordinate transformations

### Economics and Optimization
- **Linear programming**: Constraint matrices
- **Input-output models**: Economic relationships
- **Portfolio optimization**: Risk and return transformations