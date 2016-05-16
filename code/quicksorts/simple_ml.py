import math
import random
from time import time

def quicksort(alist,choose_pivot,measure_of_central_tendency,measure_of_variance):
    if len(alist) <=1:
        return alist
    else:
        pivot = choose_pivot(alist,measure_of_central_tendency,measure_of_variance)
        less = []
        equal = []
        greater = []
        for i in alist:
            if i < pivot: less.append(i)
            elif i == pivot: equal.append(i)
            else: greater.append(i)
        return quicksort(less,choose_pivot,measure_of_central_tendency,measure_of_variance)+equal+quicksort(greater,choose_pivot,measure_of_central_tendency,measure_of_variance)


def choose_pivot(alist,measure_of_central_tendency,measure_of_variance):
    central_tendency = measure_of_central_tendency(alist)
    variance = measure_of_variance(alist,central_tendency)
    for elem in alist:
        if elem > (central_tendency-variance) and elem < (central_tendency+variance):
            return elem
    return alist[0]

measure_of_central_tendency = lambda alist: sum(alist)/float(len(alist))
measure_of_variance = lambda alist,central_tendency: math.sqrt(sum([math.pow((elem - central_tendency),2) for elem in alist])/float(len(alist)))

def p_quicksort(alist):
    if len(alist) <=1:
        return alist
    pivot = alist[random.randint(0,len(alist)-1)]
    less = []
    equal = []
    greater = []
    for i in alist:
        if i < pivot:
            less.append(i)
        elif i == pivot:
            equal.append(i)
        else:
            greater.append(i)
    return p_quicksort(less)+equal+p_quicksort(greater)

print("My sort")
start = time()
quicksort([random.randint(0,100000) for _ in range(1000)],choose_pivot,measure_of_central_tendency,measure_of_variance)
print(time()-start)
print("Other sort")
start = time()
p_quicksort([random.randint(0,100000) for _ in range(1000)])
print(time()-start)
print("tim sort")
start = time()
[random.randint(0,100000) for _ in range(1000)].sort()
print(time()-start)
