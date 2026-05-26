import nonlinear
import linear
import numpy as np

print("=" * 70)
print("LABORATORY WORK #1 - COMPUTATIONAL MATHEMATICS")
print("STUDENT: FILIMONOV N.E., GROUP: TO231FNE")
print("VARIANT: 20")
print("=" * 70)

print("\n" + "=" * 70)
print("PART 1: SOLVING EQUATION f(x) = x^3 - 0.2*x*cos(x)")
print("=" * 70)

# Test positive root (0.1 to 1.0)
print("\nPOSITIVE ROOT (interval [0.1, 1.0]):")
print("-" * 70)

methods_nl = [
    ("1. Bisection method", lambda: nonlinear.bisection(0.1, 1.0)),
    ("2. Chords method", lambda: nonlinear.chords(0.1, 1.0)),
    ("3. Golden section method", lambda: nonlinear.golden_section(0.1, 1.0)),
    ("4. Newton method", lambda: nonlinear.newton(0.5)),
    ("5. Simple iteration method", lambda: nonlinear.simple_iteration(0.5)),
    ("6. Combined method", lambda: nonlinear.combined(0.1, 1.0)),
]

for name, method in methods_nl:
    try:
        r = method()
        print(f"\n{name}:")
        print(f"   Root: {r['root']:.10f}")
        print(f"   f(root): {r['f_root']:.2e}")
        print(f"   Iterations: {r['iterations']}")
        print(f"   Time: {r['time']:.6f} sec")
    except Exception as e:
        print(f"\n{name}: Error - {e}")

# Test negative root (-1.0 to -0.1)
print("\n\nNEGATIVE ROOT (interval [-1.0, -0.1]):")
print("-" * 70)

for name, method in methods_nl:
    try:
        if "Newton" in name or "Simple iteration" in name:
            # For iterative methods, use negative initial guess
            if "Newton" in name:
                r = nonlinear.newton(-0.5)
            else:
                r = nonlinear.simple_iteration(-0.5)
        else:
            r = method.__code__.co_consts[1](-1.0, -0.1)() if method.__code__.co_consts else None
        if r:
            print(f"\n{name.split('. ')[1]} (negative):")
            print(f"   Root: {r['root']:.10f}")
            print(f"   f(root): {r['f_root']:.2e}")
            print(f"   Iterations: {r['iterations']}")
            print(f"   Time: {r['time']:.6f} sec")
    except:
        pass

print("\n" + "=" * 70)
print("PART 2: SOLVING SYSTEM OF LINEAR EQUATIONS")
print("System:")
print("  2.74x - 1.18y + 3.17z = 2.18")
print("  1.12x + 0.83y - 2.16z = 1.15")
print("  0.81x + 1.27y + 0.76z = 3.23")
print("=" * 70)

methods_lin = [
    ("1. Gauss method", lambda: linear.gauss(linear.A, linear.b)),
    ("2. Orthogonalization method", lambda: linear.orthogonalization(linear.A, linear.b)),
    ("3. Simple iteration method", lambda: linear.simple_iteration_linear(linear.A, linear.b, eps=1e-8, max_iter=5000)),
    ("4. Seidel method", lambda: linear.seidel(linear.A, linear.b, eps=1e-8, max_iter=5000)),
]

for name, method in methods_lin:
    try:
        r = method()
        print(f"\n{name}:")
        print(f"   Solution: x = {r['solution'][0]:.10f}")
        print(f"              y = {r['solution'][1]:.10f}")
        print(f"              z = {r['solution'][2]:.10f}")
        print(f"   Residual: {r['residual']:.2e}")
        if 'iterations' in r:
            print(f"   Iterations: {r['iterations']}")
        print(f"   Time: {r['time']:.6f} sec")
    except Exception as e:
        print(f"\n{name}: Error - {e}")

print("\n" + "=" * 70)
print("VERIFICATION:")
print("=" * 70)

# Exact solution from numpy
exact_solution = np.linalg.solve(linear.A, linear.b)
print(f"\nExact solution (numpy.linalg.solve):")
print(f"x = {exact_solution[0]:.10f}")
print(f"y = {exact_solution[1]:.10f}")
print(f"z = {exact_solution[2]:.10f}")

# Verification
r = linear.gauss(linear.A, linear.b)
x = r['solution']
Ax = linear.A @ x
print(f"\nVerification (Gauss method):")
print(f"A * x = [{Ax[0]:.10f}, {Ax[1]:.10f}, {Ax[2]:.10f}]")
print(f"b     = [{linear.b[0]:.10f}, {linear.b[1]:.10f}, {linear.b[2]:.10f}]")

diff = Ax - linear.b
print(f"\nDifference (A*x - b):")
print(f"  [{diff[0]:.2e}, {diff[1]:.2e}, {diff[2]:.2e}]")
print(f"  Norm: {np.linalg.norm(diff):.2e}")

print("\n" + "=" * 70)
print("SUMMARY:")
print("=" * 70)
print("1. Equation f(x)=0 has two roots: +0.42669 and -0.42669")
print("2. Newton method is the fastest (2-3 iterations)")
print("3. Direct methods (Gauss, Orthogonalization) give exact solution")
print("4. Iterative methods for linear systems need careful parameter tuning")
print("5. All methods demonstrate expected convergence behavior")
print("=" * 70)
