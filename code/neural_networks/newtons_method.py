from hackthederivative import complex_step_finite_diff as deriv
import random
import sys

def dx(f,x):
    return abs(0-f(x))

def newtons_method(f,x0,e,max_iterations=10000):
    delta = dx(f,x0)
    count = 0
    
    while delta > e:
        x0 = x0 - (f(x0)/deriv(f,x0))
        delta = dx(f,x0)
        count += 1
        if count > max_iterations:
            break
    if count < max_iterations:
        print("Root is at: ", x0)
        print("f(x) at root is: ", f(x0))
        sys.exit(0)

[newtons_method(lambda x: 6*x**5 - 5*x**4 - 4*x**3 + 3*x**2,random.randint(0,100),1e-5) for _ in range(1000)]
