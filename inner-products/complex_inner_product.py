"""
complex_inner_product.py

=============================================================================
                        COMPLEX INNER PRODUCT THEORY
=============================================================================

DEFINITION:
For vectors x, y ∈ ℂⁿ (complex n-dimensional space), the complex inner product is:

    ⟨x,y⟩ = x₁ȳ₁ + x₂ȳ₂ + ... + xₙȳₙ = Σᵢ₌₁ⁿ xᵢȳᵢ

where ȳᵢ denotes the complex conjugate of yᵢ.

MATRIX FORM:
In matrix notation: ⟨x,y⟩ = x*y where x* is the conjugate transpose of x.

KEY PROPERTIES:
1. Conjugate Symmetry (Hermitian Property):
   ⟨x,y⟩ = ⟨y,x⟩*  (conjugate of ⟨y,x⟩)
   
2. Linearity in First Argument:
   ⟨αx + βz, y⟩ = α⟨x,y⟩ + β⟨z,y⟩
   
3. Positive Definiteness:
   ⟨x,x⟩ ≥ 0 for all x, and ⟨x,x⟩ = 0 iff x = 0
   Note: ⟨x,x⟩ is always real and non-negative

INDUCED NORM:
The complex inner product induces a norm (length measure):
    ‖x‖ = √⟨x,x⟩ = √(|x₁|² + |x₂|² + ... + |xₙ|²)

ORTHOGONALITY:
Two vectors x, y are orthogonal if ⟨x,y⟩ = 0, written as x ⟂ y.

GEOMETRIC INTERPRETATION:
- The real part Re(⟨x,y⟩) relates to the "alignment" of x and y
- The imaginary part Im(⟨x,y⟩) captures phase relationships
- For unit vectors: |⟨x,y⟩| measures how "close" x and y are

PROJECTION FORMULA:
The orthogonal projection of vector b onto the line spanned by vector a is:
    proj_a(b) = (⟨b,a⟩/⟨a,a⟩) · a

This minimizes the distance ‖b - proj_a(b)‖.

APPLICATIONS:
- Quantum mechanics: Inner products between quantum states
- Signal processing: Correlation between complex signals
- Fourier analysis: Coefficients in complex Fourier series
- Control theory: Complex frequency domain analysis

=============================================================================

IMPLEMENTATION DETAILS:

Demonstrates the complex inner product on ℂⁿ with:
- Pure-Python implementation
- Optional NumPy matrix form (if NumPy is available)  
- Examples showing Hermitian property, positive-definiteness (norm), and projection

Usage:
    python3 complex_inner_product.py

"""
from math import isclose, sqrt
from typing import Iterable, List

try:
    import numpy as np
    _HAS_NUMPY = True
except Exception:
    _HAS_NUMPY = False


def complex_inner_product(x: Iterable[complex], y: Iterable[complex]) -> complex:
    """Compute the complex inner product ⟨x,y⟩ = sum_i x_i * conj(y_i).

    This follows the convention used in the repository README: conjugate on
    the second vector y.
    """
    return sum(a * complex(b).conjugate() for a, b in zip(x, y))


def norm(x: Iterable[complex]) -> float:
    """Return the induced norm ‖x‖ = sqrt(⟨x,x⟩).

    Guaranteed to return a non-negative real number.
    """
    val = complex_inner_product(x, x)
    # val should be real and non-negative; guard numerical noise
    return sqrt(max(0.0, val.real))


def is_orthogonal(x: Iterable[complex], y: Iterable[complex], tol: float = 1e-10) -> bool:
    ip = complex_inner_product(x, y)
    # use magnitude because the inner product may be complex; orthogonality
    # means the inner product is (numerically) zero
    return abs(ip) <= tol


def projection_onto(a: Iterable[complex], b: Iterable[complex]):
    """Project b onto the line spanned by a using the complex inner product.

    proj_a(b) = (⟨b,a⟩ / ⟨a,a⟩) * a
    """
    denom = complex_inner_product(a, a)
    if denom == 0:
        raise ValueError("Cannot project onto the zero vector")
    scalar = complex_inner_product(b, a) / denom
    return [scalar * ai for ai in a]


def matrix_form_np(x: Iterable[complex], y: Iterable[complex]):
    """If NumPy is available, compute the same inner product by array ops.

    Using the repository convention ⟨x,y⟩ = sum(x * conj(y)).
    """
    if not _HAS_NUMPY:
        raise RuntimeError("NumPy is not available")
    xa = np.asarray(x, dtype=np.complex128)
    ya = np.asarray(y, dtype=np.complex128)
    return float((xa * np.conjugate(ya)).sum()) if xa.size == 1 else (xa * np.conjugate(ya)).sum()


def _format_vec(v: Iterable[complex]) -> str:
    return "[" + ", ".join(f"{z}" for z in v) + "]"


if __name__ == "__main__":
    # Example vectors in C^2
    x = [1 + 2j, 3 - 1j]
    y = [2 - 1j, -1 + 4j]

    print("Complex Inner Product Demo")
    print("x =", _format_vec(x))
    print("y =", _format_vec(y))

    ip_xy = complex_inner_product(x, y)
    ip_yx = complex_inner_product(y, x)

    print(f"⟨x,y⟩ = {ip_xy}")
    print(f"⟨y,x⟩ = {ip_yx}")
    print(f"Conjugate symmetry check: conj(⟨x,y⟩) == ⟨y,x⟩? -> {complex(ip_xy).conjugate() == ip_yx}")

    # Positive-definiteness / norm
    norm_x = norm(x)
    norm_y = norm(y)
    print(f"‖x‖ = {norm_x:.6f}")
    print(f"‖y‖ = {norm_y:.6f}")
    print(f"⟨x,x⟩ = {complex_inner_product(x, x)}  (should be real and = sum(|xi|^2))")

    # Orthogonality test
    orth = is_orthogonal(x, y)
    print(f"x ⟂ y? -> {orth}")

    # Projection of y onto x
    try:
        proj = projection_onto(x, y)
        print("Projection of y onto x:", _format_vec(proj))
        # compute orthogonal component
        ortho_comp = [yi - pi for yi, pi in zip(y, proj)]
        print("Orthogonal component (y - proj_x(y)):", _format_vec(ortho_comp))
        print("Check orthogonality of orthogonal component with x:", is_orthogonal(ortho_comp, x))
    except ValueError as e:
        print("Projection error:", e)

    # Optional NumPy matrix form
    if _HAS_NUMPY:
        try:
            np_form = matrix_form_np(x, y)
            print(f"(NumPy) matrix-form ⟨x,y⟩ = {np_form}")
        except Exception as e:
            print("NumPy form failed:", e)
    else:
        print("NumPy not available; skipped matrix-form demo.")
