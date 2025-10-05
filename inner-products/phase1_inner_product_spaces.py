"""
Phase 1: Inner Product Spaces - Understanding Why the Axioms Matter

This script demonstrates:
1. The three inner product axioms and their importance
2. What happens when axioms are violated
3. Euclidean inner product examples
4. Geometric interpretation of inner products
5. How inner products create norms (lengths)

Learning objectives:
- Understand why each axiom is necessary
- See the connection between algebra and geometry
- Build intuition for inner product spaces
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Union, List, Tuple

class InnerProductDemo:
    """Demonstrates inner product concepts and axioms"""
    
    def __init__(self):
        self.examples_run = 0
    
    def euclidean_inner_product(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Standard Euclidean inner product (dot product)
        ⟨x,y⟩ = x₁y₁ + x₂y₂ + ... + xₙyₙ = x^T y
        """
        return np.dot(x, y)
    
    def demonstrate_axioms(self):
        """Demonstrate the three inner product axioms and why they matter"""
        print("=" * 60)
        print("INNER PRODUCT AXIOMS - Why They Matter")
        print("=" * 60)
        
        # Example vectors
        u = np.array([1, 2])
        v = np.array([3, 1]) 
        w = np.array([-1, 4])
        a, b = 2, -3
        
        print(f"Example vectors:")
        print(f"u = {u}")
        print(f"v = {v}")
        print(f"w = {w}")
        print(f"Scalars: a = {a}, b = {b}")
        print()
        
        # AXIOM 1: LINEARITY (Bilinearity)
        print("AXIOM 1: LINEARITY IN FIRST ARGUMENT")
        print("⟨au + bv, w⟩ = a⟨u,w⟩ + b⟨v,w⟩")
        print("-" * 40)
        
        left_side = self.euclidean_inner_product(a*u + b*v, w)
        right_side = a * self.euclidean_inner_product(u, w) + b * self.euclidean_inner_product(v, w)
        
        print(f"Left side:  ⟨{a}u + {b}v, w⟩ = ⟨{a*u + b*v}, {w}⟩ = {left_side}")
        print(f"Right side: {a}⟨u,w⟩ + {b}⟨v,w⟩ = {a}({self.euclidean_inner_product(u, w)}) + {b}({self.euclidean_inner_product(v, w)}) = {right_side}")
        print(f"Equal? {np.isclose(left_side, right_side)}")
        print()
        print("WHY THIS MATTERS:")
        print("- Linearity allows us to 'distribute' the inner product")
        print("- Essential for projections and decompositions")
        print("- Makes inner products compatible with linear combinations")
        print()
        
        # AXIOM 2: SYMMETRY 
        print("AXIOM 2: SYMMETRY")
        print("⟨u,v⟩ = ⟨v,u⟩ (for real inner products)")
        print("-" * 40)
        
        uv = self.euclidean_inner_product(u, v)
        vu = self.euclidean_inner_product(v, u)
        
        print(f"⟨u,v⟩ = ⟨{u}, {v}⟩ = {uv}")
        print(f"⟨v,u⟩ = ⟨{v}, {u}⟩ = {vu}")
        print(f"Equal? {np.isclose(uv, vu)}")
        print()
        print("WHY THIS MATTERS:")
        print("- Ensures the inner product behaves like familiar multiplication")
        print("- Angle between u and v equals angle between v and u")
        print("- Makes inner product matrices symmetric")
        print()
        
        # AXIOM 3: POSITIVE DEFINITENESS
        print("AXIOM 3: POSITIVE DEFINITENESS")
        print("⟨v,v⟩ > 0 for v ≠ 0, and ⟨0,0⟩ = 0")
        print("-" * 40)
        
        zero_vec = np.array([0, 0])
        nonzero_vecs = [u, v, w, np.array([1, 0]), np.array([0, 1])]
        
        print(f"⟨0,0⟩ = {self.euclidean_inner_product(zero_vec, zero_vec)}")
        
        for vec in nonzero_vecs:
            inner_prod = self.euclidean_inner_product(vec, vec)
            print(f"⟨{vec},{vec}⟩ = {inner_prod} > 0? {inner_prod > 0}")
        
        print()
        print("WHY THIS MATTERS:")
        print("- Gives us a notion of 'length': ||v|| = √⟨v,v⟩")
        print("- Ensures distances are always positive")
        print("- Zero length only for the zero vector")
        print("- Foundation for norms and metrics")
        print()
    
    def demonstrate_broken_axioms(self):
        """Show what happens when we violate the axioms"""
        print("=" * 60)
        print("WHAT HAPPENS WHEN AXIOMS ARE VIOLATED?")
        print("=" * 60)
        
        u = np.array([1, 2])
        v = np.array([3, 1])
        
        print("EXAMPLE 1: Violating Positive Definiteness")
        print("Let's try: ⟨x,y⟩ = x₁y₁ - x₂y₂ (note the minus sign)")
        print("-" * 50)
        
        def bad_inner_product(x, y):
            return x[0]*y[0] - x[1]*y[1]  # Violates positive definiteness!
        
        test_vec = np.array([1, 2])
        result = bad_inner_product(test_vec, test_vec)
        print(f"⟨{test_vec},{test_vec}⟩ = 1×1 - 2×2 = {result}")
        print(f"This is NEGATIVE! This breaks our notion of 'length'")
        print(f"We'd get ||{test_vec}|| = √{result} = imaginary number!")
        print()
        
        print("EXAMPLE 2: Violating Symmetry")
        print("Let's try: ⟨x,y⟩ = x₁y₂ + x₂y₁ (mixed up indices)")
        print("-" * 50)
        
        def asymmetric_product(x, y):
            return x[0]*y[1] + x[1]*y[0] if len(x) == len(y) == 2 else 0
        
        xy = asymmetric_product(u, v)
        yx = asymmetric_product(v, u)
        print(f"⟨{u},{v}⟩ = {u[0]}×{v[1]} + {u[1]}×{v[0]} = {xy}")
        print(f"⟨{v},{u}⟩ = {v[0]}×{u[1]} + {v[1]}×{u[0]} = {yx}")
        print(f"These are equal: {xy == yx}, so this one actually IS symmetric!")
        print("(This particular example doesn't break symmetry, but shows the idea)")
        print()
    
    def geometric_interpretation(self):
        """Demonstrate the geometric meaning of inner products"""
        print("=" * 60)
        print("GEOMETRIC INTERPRETATION: ⟨x,y⟩ = ||x|| ||y|| cos(θ)")
        print("=" * 60)
        
        # Define some vectors
        vectors = [
            (np.array([3, 0]), np.array([0, 4]), "perpendicular"),
            (np.array([1, 1]), np.array([1, 1]), "parallel (same direction)"),
            (np.array([2, 1]), np.array([-1, 2]), "perpendicular"),
            (np.array([3, 4]), np.array([-3, -4]), "parallel (opposite direction)"),
        ]
        
        for i, (x, y, description) in enumerate(vectors, 1):
            print(f"Example {i}: {description}")
            print(f"x = {x}, y = {y}")
            
            # Calculate components
            inner_prod = self.euclidean_inner_product(x, y)
            norm_x = np.linalg.norm(x)
            norm_y = np.linalg.norm(y)
            
            # Calculate angle
            cos_theta = inner_prod / (norm_x * norm_y) if norm_x * norm_y != 0 else 0
            theta_rad = np.arccos(np.clip(cos_theta, -1, 1))
            theta_deg = np.degrees(theta_rad)
            
            print(f"⟨x,y⟩ = {inner_prod}")
            print(f"||x|| = {norm_x:.3f}, ||y|| = {norm_y:.3f}")
            print(f"cos(θ) = {inner_prod}/{norm_x:.3f} × {norm_y:.3f} = {cos_theta:.3f}")
            print(f"θ = {theta_deg:.1f}°")
            
            # Interpret the result
            if abs(inner_prod) < 1e-10:
                print("→ Vectors are ORTHOGONAL (perpendicular)")
            elif cos_theta > 0:
                print("→ Vectors point in SIMILAR directions")
            else:
                print("→ Vectors point in OPPOSITE directions")
            print()
    
    def visualize_inner_product(self):
        """Create visualizations to show inner product geometry"""
        print("Creating visualization...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Orthogonal vectors
        x1, y1 = np.array([3, 0]), np.array([0, 4])
        ax1.quiver(0, 0, x1[0], x1[1], angles='xy', scale_units='xy', scale=1, color='blue', width=0.005, label='x')
        ax1.quiver(0, 0, y1[0], y1[1], angles='xy', scale_units='xy', scale=1, color='red', width=0.005, label='y')
        ax1.set_xlim(-1, 4)
        ax1.set_ylim(-1, 5)
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_title(f'Orthogonal: ⟨x,y⟩ = {self.euclidean_inner_product(x1, y1)}')
        
        # Plot 2: Parallel vectors (same direction)
        x2, y2 = np.array([2, 1]), np.array([4, 2])
        ax2.quiver(0, 0, x2[0], x2[1], angles='xy', scale_units='xy', scale=1, color='blue', width=0.005, label='x')
        ax2.quiver(0, 0, y2[0], y2[1], angles='xy', scale_units='xy', scale=1, color='red', width=0.005, label='y')
        ax2.set_xlim(-1, 5)
        ax2.set_ylim(-1, 3)
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_title(f'Parallel: ⟨x,y⟩ = {self.euclidean_inner_product(x2, y2)}')
        
        # Plot 3: Opposite direction
        x3, y3 = np.array([3, 2]), np.array([-1.5, -1])
        ax3.quiver(0, 0, x3[0], x3[1], angles='xy', scale_units='xy', scale=1, color='blue', width=0.005, label='x')
        ax3.quiver(0, 0, y3[0], y3[1], angles='xy', scale_units='xy', scale=1, color='red', width=0.005, label='y')
        ax3.set_xlim(-2, 4)
        ax3.set_ylim(-2, 3)
        ax3.set_aspect('equal')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        ax3.set_title(f'Opposite: ⟨x,y⟩ = {self.euclidean_inner_product(x3, y3)}')
        
        # Plot 4: General case with angle
        x4, y4 = np.array([3, 1]), np.array([1, 3])
        inner_prod = self.euclidean_inner_product(x4, y4)
        angle = np.degrees(np.arccos(inner_prod / (np.linalg.norm(x4) * np.linalg.norm(y4))))
        
        ax4.quiver(0, 0, x4[0], x4[1], angles='xy', scale_units='xy', scale=1, color='blue', width=0.005, label='x')
        ax4.quiver(0, 0, y4[0], y4[1], angles='xy', scale_units='xy', scale=1, color='red', width=0.005, label='y')
        
        # Draw angle arc
        theta = np.linspace(0, np.radians(angle), 20)
        r = 0.5
        arc_x = r * np.cos(theta)
        arc_y = r * np.sin(theta)
        ax4.plot(arc_x, arc_y, 'green', linewidth=2)
        ax4.text(0.3, 0.2, f'{angle:.1f}°', color='green', fontweight='bold')
        
        ax4.set_xlim(-1, 4)
        ax4.set_ylim(-1, 4)
        ax4.set_aspect('equal')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        ax4.set_title(f'General: ⟨x,y⟩ = {inner_prod}, θ = {angle:.1f}°')
        
        plt.tight_layout()
        plt.suptitle('Inner Product Geometry: ⟨x,y⟩ = ||x|| ||y|| cos(θ)', y=1.02, fontsize=14, fontweight='bold')
        plt.show()
    
    def demonstrate_induced_norm(self):
        """Show how inner products create norms (lengths)"""
        print("=" * 60)
        print("INDUCED NORM: ||v|| = √⟨v,v⟩")
        print("=" * 60)
        
        vectors = [
            np.array([3, 4]),      # Classic 3-4-5 triangle
            np.array([1, 0]),      # Unit vector
            np.array([0, 1]),      # Unit vector
            np.array([1, 1]),      # 45° vector
            np.array([-2, -2]),    # Negative components
        ]
        
        print("Vector\t\t⟨v,v⟩\t\t||v|| = √⟨v,v⟩\tGeometric Length")
        print("-" * 70)
        
        for v in vectors:
            inner_prod_self = self.euclidean_inner_product(v, v)
            norm_from_inner = np.sqrt(inner_prod_self)
            geometric_length = np.linalg.norm(v)
            
            print(f"{str(v):15}\t{inner_prod_self:8.1f}\t{norm_from_inner:8.3f}\t\t{geometric_length:8.3f}")
        
        print()
        print("KEY INSIGHTS:")
        print("1. The inner product with itself gives the squared length")
        print("2. Taking the square root gives the geometric length")
        print("3. This works for ANY inner product, not just the dot product")
        print("4. The norm satisfies: ||v|| ≥ 0, with equality iff v = 0")
        print()
    
    def interactive_demo(self):
        """Interactive demonstration where user can input vectors"""
        print("=" * 60)
        print("INTERACTIVE INNER PRODUCT CALCULATOR")
        print("=" * 60)
        
        try:
            print("Enter two 2D vectors to compute their inner product:")
            
            # Get first vector
            x1 = float(input("Vector 1 - x₁ component: "))
            x2 = float(input("Vector 1 - x₂ component: "))
            v1 = np.array([x1, x2])
            
            # Get second vector
            y1 = float(input("Vector 2 - y₁ component: "))
            y2 = float(input("Vector 2 - y₂ component: "))
            v2 = np.array([y1, y2])
            
            # Calculations
            inner_prod = self.euclidean_inner_product(v1, v2)
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            print(f"\nResults:")
            print(f"v₁ = {v1}")
            print(f"v₂ = {v2}")
            print(f"⟨v₁,v₂⟩ = {inner_prod}")
            print(f"||v₁|| = {norm1:.3f}")
            print(f"||v₂|| = {norm2:.3f}")
            
            if norm1 * norm2 != 0:
                cos_theta = inner_prod / (norm1 * norm2)
                theta_deg = np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))
                print(f"cos(θ) = {cos_theta:.3f}")
                print(f"θ = {theta_deg:.1f}°")
                
                if abs(inner_prod) < 1e-10:
                    print("→ The vectors are ORTHOGONAL!")
                elif cos_theta > 0.9:
                    print("→ The vectors are nearly PARALLEL!")
                elif cos_theta < -0.9:
                    print("→ The vectors are nearly OPPOSITE!")
            
        except ValueError:
            print("Invalid input. Please enter numeric values.")
        except Exception as e:
            print(f"Error: {e}")

def main():
    """Run the complete Phase 1 demonstration"""
    demo = InnerProductDemo()
    
    print("PHASE 1: INNER PRODUCT SPACES - Understanding the Foundations")
    print("=" * 80)
    print()
    print("This demonstration will help you understand:")
    print("• Why the three inner product axioms are essential")
    print("• The geometric meaning of inner products") 
    print("• How inner products create norms (lengths)")
    print("• What happens when axioms are violated")
    print()
    
    while True:
        print("Choose a demonstration:")
        print("1. Inner Product Axioms (Why They Matter)")
        print("2. What Happens When Axioms Are Violated") 
        print("3. Geometric Interpretation") 
        print("4. Visualize Inner Product Geometry")
        print("5. Induced Norm (Length from Inner Product)")
        print("6. Interactive Calculator")
        print("7. Run All Demonstrations")
        print("0. Exit")
        
        try:
            choice = input("\nEnter your choice (0-7): ").strip()
            
            if choice == '0':
                print("Happy learning! 📚")
                break
            elif choice == '1':
                demo.demonstrate_axioms()
            elif choice == '2':
                demo.demonstrate_broken_axioms()
            elif choice == '3':
                demo.geometric_interpretation()
            elif choice == '4':
                demo.visualize_inner_product()
            elif choice == '5':
                demo.demonstrate_induced_norm()
            elif choice == '6':
                demo.interactive_demo()
            elif choice == '7':
                demo.demonstrate_axioms()
                demo.demonstrate_broken_axioms()
                demo.geometric_interpretation()
                demo.demonstrate_induced_norm()
                demo.visualize_inner_product()
                print("\n" + "="*60)
                print("ALL DEMONSTRATIONS COMPLETE!")
                print("="*60)
            else:
                print("Invalid choice. Please enter 0-7.")
                
        except KeyboardInterrupt:
            print("\n\nExiting... Happy learning! 📚")
            break
        except Exception as e:
            print(f"Error: {e}")
        
        print("\n" + "-"*80 + "\n")

if __name__ == "__main__":
    main()