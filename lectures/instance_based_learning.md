#Introduction

Instance based learning are machine learning algorithms that require "no" training.  These algorithms are pretty good for solving specific types of problems.  They usually come in the partition or clustering flavor and are reasonably easy to implement.  Which is why it's always been strange to me that I can't find more implementations of these algorithms on the internet.

Thanks to Google, I was saved today and found a great set of lectures on coursera on machine learning:

https://class.coursera.org/machlearning-001/lecture

This may be the best ML class I've found on the internet, with some caveats - you need to know programming already and you need to know algorithm design.  Aka you need to be able to take an idea, presented at a high level, and implement it.  

As long as you can do that, this course will make your mouth water.  The lecturer gives you everything you need to implement your favorite algorithms.  And I would claim the course is a must watch for anyone serious about machine learning.  

##K-nearest-neighbors

A lot of people like to talk about this algorithm.  It's the first "real" example of a non-standard statistical technique in [Elements of Statistical Learning](http://statweb.stanford.edu/~tibs/ElemStatLearn/) and it's one of the top ten must know algorithms for data science.  And yet, not even it's [wikipedia page](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm) which is called K-nearest neighbors algorithm, has an implementation.  Is this because the algorithm is so sophisticatedly hard?!  Nope.

It's because its desceptively simple to implement.  I guess everyone else just figures they don't need to tell you how to do it, because if you see it, you might not even care.  Which is pretty sad.  Once I finally found the above lectures I mentioned and wrote down the algorithm for myself, I was stunned at it's beauty and simplicity.  And more than that, at it's power - K nearest neighbors assumes very little about it's underlying distribution unlike many classical statistical techniques.  And yet, it can be used for partitioning, regression, prediction, and many other very serious applications.  

Like the coursera class (which really really should check out), I will first show you nearest neighbor, the simpler less talked about cousin of k nearest neighbors.

###Nearest Neighbor

The nearest neighbor algorithm proceeds as follows:

1. "train" your data on your input and outputs
2. when new data comes in, find the input data that is closest to the new input
3. output the output for the closest such result as the result for the new data

(that's it)

Seems like a pretty simple idea, right?!  Well let's see it in practice:

```
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
    return x*x

def distance(x,y):
    return abs(x - y)

x_s = [i for i in range(1000)]
y_s = [func(elem) for elem in x_s]
nn = train(nearest_neighbor,x_s,y_s)
print(nn(17,distance)) #prints 289 (which is the square of 17)
```

Notice the "training" phase is just storing the x and y values in the function.  We could have done this via a class too, but it just felt more natural to do it this way.

Notice also, that we pass in a distance function - feel free to define your own, there is nothing unique about the one I chose.  

Notice also that we are using nearest neighbors to do prediction - returning a number rather than a class.  Let's look at an example now with classes.

We've only added 2 new functions here:
```

def classify(x):
    if x % 2 == 0:
        return "even"
    else:
        return "odd"

def classification_testing_nn():
    x_s = [i for i in range(1000)]
    y_s = [classify(i) for i in x_s]
    nn = train(nearest_neighbor,x_s,y_s)
    print(nn(17,distance))

if __name__ == '__main__':
	classification_testing_nn() #prints odd
```

As expected, nn works pretty damn well on single variable input.  In the real world we are usually modeling something more complex.  And that's where nearest_neighbors tends to fall apart.  But if there is some degree of locality to your data, taking the k-nearest neighbors actually tends to work out pretty!

###K Nearest Neighbors

I'll stick with the 1-dimensional case for the implementation, just to show simple the algorithm is with k-neighbors.  Then we'll show the power of KNNs on many data points (and how Nearest Neighbors falls short).  


Algorithm:

1. "train" your data on your input and outputs
2. when new data comes in, find the k closest input data to the new input
3. store the k-closest results in an array
4. get the corresponding output values for the input, and take the central tendency of all relevant output values.

```
def k_nearest_neighbor(x_s,y_s,k,new_data,distance_function):
    distances = []
    distance_lookup = {}
    for ind,val in enumerate(x_s):
        d = distance_function(val,new_data)
        distances.append(d)
        distance_lookup[d] = ind
    distances.sort()
    return statistics.median([y_s[distance_lookup[elem]] for elem in distances[:k]])
        
def train(NN,x_s,y_s):
    return partial(NN,x_s,y_s)

def func(x):
    return x+17
    
def distance(x,y):
    return abs(x - y)

def prediction_testing_knn():
    x_s = [i for i in range(1000)]
    y_s = [func(elem) for elem in x_s]
    knn = train(k_nearest_neighbor,x_s,y_s)
    print(knn(4,17,distance))

```

The idea here is simple - get the distances for all the training data to the new point.  Get the K smallest distances and then take the central tendency of the k-closest elements.  In this case, we are using the median because it tends to understand the central tendency.  Since K nearest neighbors already tends to over state things, using the median keeps things closer to what you'd want to see.  Of course, this is a little wishy washy, you should try to verify this yourself with lots of experiments!  

Before we move onto complex data, let's see how classification looks in K-NN land:

```
from collections import Counter

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

def classify(x):
    if x % 2 == 0:
        return "even"
    else:
        return "odd"

def classification_testing_knn(distance):
    x_s = [i for i in range(1000)]
    y_s = [classify(i) for i in x_s]
    knn = train(k_nearest_neighbor_classification,x_s,y_s)
    print(knn(5,17,distance))

if __name__ == '__main__':
    classification_testing_knn(cartesian_distance)
```

The `Counter` module takes in a list and does various common facilities on it - the `most_common` function returns a list of tuples of the form (element,frequency of element).  We make use of this to find the most frequently occurring class, if multiple classes are most frequent we randomly choose one of the most frequent classes.  


