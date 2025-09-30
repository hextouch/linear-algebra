# Vector Spaces

A vector space is a fundamental algebraic structure in linear algebra, consisting of a collection of objects called vectors that can be added together and multiplied by scalars.

## Key Concepts

### Vector Operations
- **Addition**: Combining two vectors to produce a third vector
- **Scalar multiplication**: Scaling a vector by a real number
- **Zero vector**: The additive identity element
- **Additive inverse**: For every vector v, there exists -v

### Vector Space Axioms
1. **Closure under addition**: u + v is in V
2. **Commutativity**: u + v = v + u
3. **Associativity**: (u + v) + w = u + (v + w)
4. **Zero vector**: v + 0 = v
5. **Additive inverse**: v + (-v) = 0
6. **Closure under scalar multiplication**: cv is in V
7. **Distributivity**: c(u + v) = cu + cv
8. **Distributivity**: (c + d)v = cv + dv
9. **Associativity**: c(dv) = (cd)v
10. **Identity**: 1v = v

### Linear Independence and Dependence
- **Linear combination**: c₁v₁ + c₂v₂ + ... + cₙvₙ
- **Linear independence**: Only trivial combination equals zero
- **Linear dependence**: Non-trivial combination equals zero
- **Span**: Set of all linear combinations

### Basis and Dimension
- **Basis**: Linearly independent spanning set
- **Dimension**: Number of vectors in a basis
- **Coordinate vector**: Representation with respect to a basis
- **Change of basis**: Converting between coordinate systems

### Subspaces
- **Definition**: Subset closed under vector operations
- **Column space**: Span of matrix columns
- **Null space**: Solutions to Ax = 0
- **Row space**: Span of matrix rows

## Important Examples

### Standard Vector Spaces
- **ℝⁿ**: n-dimensional real coordinate space
- **ℂⁿ**: n-dimensional complex coordinate space
- **Pₙ**: Polynomials of degree at most n
- **C[a,b]**: Continuous functions on interval [a,b]
- **M_{m×n}**: m×n matrices

### Common Bases
- **Standard basis for ℝⁿ**: {e₁, e₂, ..., eₙ}
- **Polynomial basis**: {1, x, x², ..., xⁿ}
- **Fourier basis**: {1, cos(x), sin(x), cos(2x), sin(2x), ...}

## Applications

- **Computer graphics**: 3D transformations and modeling
- **Physics**: State spaces in quantum mechanics
- **Economics**: Linear programming and optimization
- **Data science**: Feature spaces and dimensionality reduction
- **Engineering**: Signal processing and control systems

## Computational Tools

- **NumPy**: Vector operations and linear algebra
- **SciPy**: Advanced linear algebra functions
- **SymPy**: Symbolic computation for exact solutions
- **Matplotlib**: Visualization of vectors and spaces