import math
import time

def f(x):
    return x**3 - 0.2 * x * math.cos(x)

def df(x):
    return 3*x**2 - 0.2*math.cos(x) + 0.2*x*math.sin(x)

def bisection(a, b, eps_x=1e-6, eps_f=1e-6, max_iter=1000):
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError(f"Function must change sign on interval. f({a})={fa:.6f}, f({b})={fb:.6f}")
    
    n = 0
    func_calls = 2
    start = time.time()
    
    while (b - a) / 2 > eps_x and n < max_iter:
        c = (a + b) / 2
        fc = f(c)
        func_calls += 1
        n += 1
        
        if abs(fc) < eps_f:
            break
        
        if fc * fa < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
    
    root = (a + b) / 2
    f_root = f(root)
    func_calls += 1
    return {
        "root": root,
        "f_root": f_root,
        "iterations": n,
        "func_calls": func_calls,
        "time": time.time() - start
    }

def chords(a, b, eps_x=1e-6, eps_f=1e-6, max_iter=1000):
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError(f"Function must change sign on interval. f({a})={fa:.6f}, f({b})={fb:.6f}")
    
    n = 0
    func_calls = 2
    start = time.time()
    x = a
    
    for n in range(max_iter):
        x_new = (a * fb - b * fa) / (fb - fa)
        fx_new = f(x_new)
        func_calls += 1
        
        if abs(fx_new) < eps_f or abs(x_new - x) < eps_x:
            x = x_new
            break
        
        if fx_new * fa < 0:
            b = x_new
            fb = fx_new
        else:
            a = x_new
            fa = fx_new
        
        x = x_new
    
    return {
        "root": x,
        "f_root": f(x),
        "iterations": n + 1,
        "func_calls": func_calls,
        "time": time.time() - start
    }

def golden_section(a, b, eps_x=1e-6, eps_f=1e-6, max_iter=1000):
    phi = (math.sqrt(5) - 1) / 2
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError(f"Function must change sign on interval. f({a})={fa:.6f}, f({b})={fb:.6f}")
    
    n = 0
    func_calls = 2
    start = time.time()
    
    while (b - a) > eps_x and n < max_iter:
        x1 = b - phi * (b - a)
        x2 = a + phi * (b - a)
        f1, f2 = f(x1), f(x2)
        func_calls += 2
        n += 1
        
        if abs(f1) < eps_f:
            root = x1
            break
        if abs(f2) < eps_f:
            root = x2
            break
        
        if f1 * fa < 0:
            b = x2
            fb = f2
        else:
            a = x1
            fa = f1
    
    root = (a + b) / 2
    return {
        "root": root,
        "f_root": f(root),
        "iterations": n,
        "func_calls": func_calls + 1,
        "time": time.time() - start
    }

def newton(x0, eps_x=1e-6, eps_f=1e-6, max_iter=1000):
    x = x0
    n = 0
    func_calls = 0
    deriv_calls = 0
    start = time.time()
    
    for n in range(max_iter):
        fx = f(x)
        dfx = df(x)
        func_calls += 1
        deriv_calls += 1
        
        if abs(dfx) < 1e-12:
            break
        
        x_new = x - fx / dfx
        
        if abs(x_new - x) < eps_x or abs(f(x_new)) < eps_f:
            x = x_new
            break
        
        x = x_new
    
    return {
        "root": x,
        "f_root": f(x),
        "iterations": n + 1,
        "func_calls": func_calls + 1,
        "deriv_calls": deriv_calls,
        "time": time.time() - start
    }

def simple_iteration(x0, eps_x=1e-6, eps_f=1e-6, max_iter=1000):
    # Use better lambda calculation
    lambda_val = 0.2  # Fixed value that works for this function
    phi = lambda x: x - lambda_val * f(x)
    
    x = x0
    n = 0
    func_calls = 0
    start = time.time()
    
    for n in range(max_iter):
        x_new = phi(x)
        func_calls += 1
        
        if abs(x_new - x) < eps_x or abs(f(x_new)) < eps_f:
            x = x_new
            break
        
        x = x_new
    
    return {
        "root": x,
        "f_root": f(x),
        "iterations": n + 1,
        "func_calls": func_calls + 1,
        "time": time.time() - start
    }

def combined(a, b, eps_x=1e-6, eps_f=1e-6, max_iter=1000):
    """Improved combined method"""
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError(f"Function must change sign on interval. f({a})={fa:.6f}, f({b})={fb:.6f}")
    
    n = 0
    func_calls = 2
    deriv_calls = 0
    start = time.time()
    
    # Use chord method as main, with occasional Newton steps
    x = a
    for n in range(max_iter):
        # Chord method iteration
        x_chord = (a * fb - b * fa) / (fb - fa)
        f_chord = f(x_chord)
        func_calls += 1
        
        # Every 5 iterations, try a Newton step
        if n % 5 == 0 and n > 0:
            df_val = df(x_chord)
            deriv_calls += 1
            if abs(df_val) > 1e-12:
                x_newton = x_chord - f_chord / df_val
                f_newton = f(x_newton)
                func_calls += 1
                
                # Use Newton if it's better
                if abs(f_newton) < abs(f_chord):
                    x_chord = x_newton
                    f_chord = f_newton
        
        if abs(f_chord) < eps_f or abs(x_chord - x) < eps_x:
            x = x_chord
            break
        
        # Update interval
        if f_chord * fa < 0:
            b = x_chord
            fb = f_chord
        else:
            a = x_chord
            fa = f_chord
        
        x = x_chord
    
    return {
        "root": x,
        "f_root": f(x),
        "iterations": n + 1,
        "func_calls": func_calls,
        "deriv_calls": deriv_calls,
        "time": time.time() - start
    }
