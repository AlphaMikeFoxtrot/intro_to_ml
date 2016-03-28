
def generate_functions(n):
    polynomials = ["x**"+str(i) for i in xrange(n)]
    odds = ["x**"+str(i) for i in xrange(n) if i%2!=0]
    evens = ["x**"+str(i) for i in xrange(n) if i%2==0]
    with open("functions.py","w") as f:
        for i in xrange(1,n):
            f.write("\ndef inorder_polynomial"+str(i)+"(x):\n\treturn "+" + ".join(polynomials[:i]))
            f.write("\ndef odds_polynomial"+str(i)+"(x):\n\treturn "+" + ".join(odds[:i]))
            f.write("\ndef evens_polynomial"+str(i)+"(x):\n\treturn "+" + ".join(polynomials[:i]))

generate_functions(10)
