pseudo_code = """
def grow_tree(S):
    if (y = 0 for all (x,y) contained in S) return new leaf(0)
    else if (y = 1 for all (x,y) contained in S) return new leaf(1)
    else
        choose best attribute x[j]
        S_0 = all (x,y) contained in S with x[j] = 0
        S_1 = all (x,y) contained in S with x[j] = 1
        return new node(x[j],grow_tree(S_0),grow_tree(S_1))
"""

choose_best ="""
def choose_best_attribute(S):
choose j to minimize J[j], computed as follows:
    S_0 = all (x,y) contained in S with x[j] == 0
    S_1 = all (x,y) contained in S with x[j] == 1
    y[0] = the most common value of y in S_0
    y[1] = the most common value of y in S_1
    J[0] = the number of examples (x,y) contained in S_0 with y != y[0]
    J[1] = the number of examples (x,y) contained in S_1 with y != y[1]
    J[j] = J[0] + J[1] (total errors if we split on this feature)
    return j
"""
