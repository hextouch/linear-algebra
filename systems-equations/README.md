# Systems of Linear Equations

Systems of linear equations form the foundation of linear algebra, with applications spanning science, engineering, economics, and data analysis.

## Basic Concepts

### General Form
**Ax = b** where:
- **A**: m×n coefficient matrix
- **x**: n×1 unknown vector
- **b**: m×1 constant vector
- **System**: m equations in n unknowns

### Expanded Form
```
a₁₁x₁ + a₁₂x₂ + ... + a₁ₙxₙ = b₁
a₂₁x₁ + a₂₂x₂ + ... + a₂ₙxₙ = b₂
...
aₘ₁x₁ + aₘ₂x₂ + ... + aₘₙxₙ = bₘ
```

### Augmented Matrix
**[A|b]**: Coefficient matrix with constant vector appended
- **Row operations**: Preserve solution set
- **Systematic solution**: Using matrix methods

## Types of Systems

### By Dimensions
- **Square system**: m = n (same equations and unknowns)
- **Overdetermined**: m > n (more equations than unknowns)
- **Underdetermined**: m < n (fewer equations than unknowns)

### By Solutions
- **Consistent**: Has at least one solution
- **Inconsistent**: No solution exists
- **Unique solution**: Exactly one solution
- **Infinitely many solutions**: Parameter-dependent family

### Homogeneous vs. Non-homogeneous
- **Homogeneous**: Ax = 0 (b = 0)
- **Non-homogeneous**: Ax = b (b ≠ 0)
- **Trivial solution**: x = 0 (always solves homogeneous system)
- **Solution structure**: x = x_p + x_h (particular + homogeneous)

## Solution Methods

### Gaussian Elimination

#### Forward Elimination
1. **Pivot selection**: Choose non-zero element
2. **Row operations**: Create zeros below pivot
3. **Upper triangular**: Transform to row echelon form
4. **Pivot positions**: Leading 1's in each row

#### Back Substitution
1. **Start from bottom**: Last equation has one unknown
2. **Substitute upward**: Use known values in previous equations
3. **Systematic solution**: Work row by row upward

#### Gauss-Jordan Elimination
- **Reduced form**: Continue elimination above pivots
- **Diagonal form**: Leading 1's with zeros above and below
- **Direct solution**: Variables read directly from final form

### Matrix Methods

#### Matrix Inverse Method
- **Condition**: A must be square and invertible
- **Solution**: x = A⁻¹b
- **Computational cost**: O(n³) for inverse, O(n²) for multiplication
- **Multiple systems**: Efficient when A is fixed, b varies

#### LU Decomposition
1. **Factorization**: A = LU
2. **Forward substitution**: Solve Ly = b
3. **Back substitution**: Solve Ux = y
4. **Efficiency**: O(n³) for decomposition, O(n²) per solve

### Iterative Methods

#### Jacobi Method
- **Diagonal dominance**: Convergence condition
- **Fixed point**: x = D⁻¹(L + U)x + D⁻¹b
- **Parallel computation**: Updates can be simultaneous
- **Convergence**: Slow but simple

#### Gauss-Seidel Method
- **Sequential updates**: Use newest values immediately
- **Faster convergence**: Than Jacobi for many problems
- **Implementation**: Overwrite values in place
- **SOR variant**: Successive Over-Relaxation for acceleration

#### Conjugate Gradient
- **Symmetric positive definite**: Specialized for SPD matrices
- **Optimal directions**: Minimizes quadratic function
- **Finite termination**: Exact solution in n steps (theory)
- **Large sparse systems**: Practical for huge problems

## Special Cases

### Homogeneous Systems (Ax = 0)

#### Solution Structure
- **Null space**: Set of all solutions forms subspace
- **Basis**: Fundamental set of solutions
- **General solution**: Linear combination of basis vectors
- **Dimension**: Nullity = n - rank(A)

#### Trivial vs. Non-trivial Solutions
- **Trivial solution**: x = 0 (always exists)
- **Non-trivial solutions**: Exist iff rank(A) < n
- **Square matrices**: Non-trivial solutions iff det(A) = 0

### Overdetermined Systems (m > n)

#### Inconsistent Systems
- **No exact solution**: More constraints than variables
- **Least squares**: Find best approximate solution
- **Normal equations**: AᵀAx = Aᵀb
- **Pseudoinverse**: x = A⁺b where A⁺ = (AᵀA)⁻¹Aᵀ

#### Consistent Systems
- **Compatible constraints**: Equations don't contradict
- **Unique solution**: If rank(A) = n
- **Multiple solutions**: If rank(A) < n

### Underdetermined Systems (m < n)

#### Solution Structure
- **Free variables**: n - rank(A) parameters
- **Particular solution**: One specific solution
- **General solution**: Particular + null space
- **Parametric form**: Express in terms of parameters

#### Minimum Norm Solution
- **Infinite solutions**: Choose one with smallest norm
- **Pseudoinverse**: x = A⁺b = Aᵀ(AAᵀ)⁻¹b
- **Optimization**: Minimize ‖x‖² subject to Ax = b

## Advanced Topics

### Least Squares Problems

#### Normal Equations
- **Overdetermined systems**: AᵀAx = Aᵀb
- **Symmetric matrix**: AᵀA is always symmetric
- **Positive definite**: When A has full column rank
- **Numerical issues**: Condition number squared

#### QR Method
- **Factorization**: A = QR
- **Triangular system**: Rx = Qᵀb
- **Numerical stability**: Better than normal equations
- **Orthogonal basis**: Q provides orthonormal columns

#### SVD Method
- **Pseudoinverse**: A⁺ = VΣ⁺Uᵀ
- **Rank deficiency**: Handles any rank matrix
- **Minimum norm**: Automatic when underdetermined
- **Numerical stability**: Best for ill-conditioned problems

### Sensitivity Analysis

#### Condition Numbers
- **Matrix condition**: κ(A) = ‖A‖‖A⁻¹‖
- **Solution sensitivity**: How errors in A, b affect x
- **Ill-conditioned**: Large condition number
- **Numerical precision**: Required digits for accuracy

#### Perturbation Theory
- **Error bounds**: Relate input and output errors
- **Backward stability**: Algorithm produces exact solution to nearby problem
- **Forward stability**: Algorithm produces approximate solution to exact problem

### Regularization

#### Ridge Regression
- **Modified normal equations**: (AᵀA + λI)x = Aᵀb
- **Regularization parameter**: λ > 0 improves conditioning
- **Bias-variance tradeoff**: Reduces overfitting
- **Applications**: Machine learning, statistics

#### Tikhonov Regularization
- **General form**: min ‖Ax - b‖² + λ‖Lx‖²
- **Smoothing operator**: L enforces solution properties
- **Inverse problems**: Stabilize ill-posed problems
- **Parameter selection**: Cross-validation, L-curve method

## Applications

### Engineering
- **Circuit analysis**: Kirchhoff's laws
- **Structural analysis**: Force equilibrium
- **Control systems**: State-space models
- **Signal processing**: Filter design

### Economics
- **Input-output models**: Economic interactions
- **Linear programming**: Resource allocation
- **Econometric models**: Regression analysis
- **Portfolio optimization**: Asset allocation

### Computer Graphics
- **3D transformations**: Coordinate changes
- **Interpolation**: Curve and surface fitting
- **Rendering**: Lighting and shading calculations
- **Animation**: Keyframe interpolation

### Data Science
- **Linear regression**: Predictive modeling
- **Principal component analysis**: Dimensionality reduction
- **Classification**: Linear discriminants
- **Recommender systems**: Matrix factorization

### Physics
- **Network analysis**: Circuit theory
- **Quantum mechanics**: State vector evolution
- **Computational physics**: Discretized differential equations
- **Optimization**: Energy minimization

## Computational Considerations

### Numerical Stability
- **Pivoting**: Partial or complete pivoting
- **Scaling**: Row and column equilibration
- **Iterative refinement**: Improve solution accuracy
- **Mixed precision**: Use different precisions strategically

### Sparse Systems
- **Storage formats**: CSR, CSC, COO formats
- **Direct methods**: Sparse LU, Cholesky
- **Iterative methods**: CG, GMRES, BiCGSTAB
- **Preconditioning**: Improve convergence

### Parallel Computing
- **Domain decomposition**: Split problem spatially
- **Block methods**: Exploit block structure
- **GPU acceleration**: Massively parallel solvers
- **Distributed computing**: Large-scale problems