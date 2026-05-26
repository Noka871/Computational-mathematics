import math

def f(x):
    return x**3 - 0.2 * x * math.cos(x)

# ??? ????????, ??? ??????? ?????? ????
print("Searching for interval where f(x) changes sign:")
for i in range(-5, 6):
    x = i * 0.5
    fx = f(x)
    print(f"f({x:5.1f}) = {fx:10.6f}")
    
print("\nTrying specific values:")
test_points = [-1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5]
for x in test_points:
    fx = f(x)
    print(f"f({x:5.1f}) = {fx:10.6f}")
