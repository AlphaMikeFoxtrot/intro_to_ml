from functools import partial
import sys
import random
import statistics
import math
from collections import Counter

def k_nearest_neighbor(x_s,y_s,k,new_data,distance_function):
    distances = []
    distance_lookup = {}
    for ind,val in enumerate(x_s):
        d = distance_function(val,new_data)
        distances.append(d)
        distance_lookup[d] = ind
    distances.sort()
    return statistics.median([y_s[distance_lookup[elem]] for elem in distances[:k]])

def k_nearest_neighbor_classification(x_s,y_s,k,new_data,distance_function):
    distances = []
    distance_lookup = {}
    for ind,val in enumerate(x_s):
        d = distance_function(val,new_data)
        distances.append(d)
        distance_lookup[d] = ind
    distances.sort()
    classes = [y_s[distance_lookup[elem]] for elem in distances[:k]]
    most_common = Counter(classes).most_common()
    most_likely_classes = [elem[0] for elem in most_common if elem[1] == most_common[0][1]]
    if len(most_likely_classes) == 1:
        return most_likely_classes[0]
    else:
        return random.choice(most_likely_classes)


def nearest_neighbor(x_s,y_s,new_data,distance_function):
    min_distance = sys.float_info.max 
    for ind,val in enumerate(x_s):
        new_distance = distance_function(new_data,val)
        if new_distance < min_distance:
            min_distance = new_distance
            index_of_y = ind
    return y_s[index_of_y]

def train(NN,x_s,y_s):
    return partial(NN,x_s,y_s)

def func(x):
    return x+17

def classify(x):
    if x % 2 == 0:
        return "even"
    else:
        return "odd"
    
def absolute_distance(x,y):
    return abs(x - y)

def cartesian_distance(x,y):
    return math.sqrt(math.pow(x - y,2))
    
def prediction_testing_knn(distance):
    x_s = [i for i in range(1000)]
    y_s = [func(elem) for elem in x_s]
    knn = train(k_nearest_neighbor,x_s,y_s)
    print(knn(4,17,distance))

def prediction_testing_nn(distance):
    x_s = [i for i in range(1000)]
    y_s = [func(elem) for elem in x_s]
    nn = train(nearest_neighbor,x_s,y_s)
    print(nn(17,distance))

def classification_testing_nn(distance):
    x_s = [i for i in range(1000)]
    y_s = [classify(i) for i in x_s]
    nn = train(nearest_neighbor,x_s,y_s)
    print(nn(17,distance))

def classification_testing_knn(distance):
    x_s = [i for i in range(1000)]
    y_s = [classify(i) for i in x_s]
    knn = train(k_nearest_neighbor_classification,x_s,y_s)
    print(knn(5,17,distance))
    
if __name__ == '__main__':
    classification_testing_knn(cartesian_distance)
    prediction_testing_knn(absolute_distance)
