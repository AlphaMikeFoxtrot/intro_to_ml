#Clustering

Clustering is the process of taking in lots of data that exists in high dimension or over multiply segments and creating segments or clusters of the data.  The goal is to understand semantically what's happening as the values of your variables vary.  Are there invariants or certain structures in the data that translate into semantic meaning?  This is the central question clustering attempts to answer.

The first algorithm we'll look at here is the faithful old k_means clustering!  

The idea has been around for a while and though it's somewhat of an oldie, it's a goodie:

##Algorithm

Pick random examples as intial means
until convergence:
	- Assign each example to nearest mean
	- new mean = average of examples assigned to it
	- exit condition: new_mean == previous_mean 

The idea here, is about as simple as it gets - where the complexity comes in is doing this over N-dimensional space.  Which is often what we are interested in.

Let's check out how one might be able to do k-means clustering:
```
import random
import statistics
from functools import partial
import math

def distance(num_dimensions,first_point,second_point):
    tmp = []
    for ind,val in enumerate(first_point):
        tmp.append(math.pow((second_point[ind] - val),2))
    return math.sqrt(sum(tmp))

def central_tendency(new_point,current_mean,discount_factor):
    return [current_mean[ind] + (discount_factor*(new_point[ind] - current_mean[ind])) for ind in range(len(current_mean))]

def k_means(data,k=3,epsilon=1):
    #if mean is the same for two successive iterations I'm done
    means = [random.choice(data) for _ in range(k)]
    dist = partial(distance,len(data[0]))
    [data.remove(mean) for mean in means]
    num_data_points_per_mean = {}.fromkeys([i for i in range(len(means))],1)
    while True:
	    for datum in data:
	        smallest_distance = float("inf")
	        mean_to_update = None
	        for ind,mean in enumerate(means):
	            d = dist(datum,mean)
	            if d < smallest_distance:
	                smallest_distance = d
	                mean_to_update = ind
	        num_data_points_per_mean[mean_to_update] += 1
	        prev_val = means[mean_to_update]
	        means[mean_to_update] = central_tendency(datum,means[mean_to_update],1/num_data_points_per_mean[mean_to_update])
	        if all([abs(means[mean_to_update][ind] - prev_val[ind]) < epsilon for i in range(len(prev_val))]):
	            return means
	    

if __name__ == '__main__':
    
    data = [[random.randint(2,150) for x in range(10)] for y in range(1000)]
    print(k_means(data))
```

There are a few points of interest here:

1. we make use of a slightly different version of the mean:

```
def central_tendency(new_point,current_mean,discount_factor):
    return [current_mean[ind] + (discount_factor*(new_point[ind] - current_mean[ind])) for ind in range(len(current_mean))]
```

Notice that this is the same thing as the classical version you are used to - we remove any contribution from the old mean by the new point and multiply by the discount factor - the number of elements already in the mean and then add this to our current mean.

The reason we have a list comprehension is we are creating means for each column in the data, instead of just one mean.  
2. This fact is further reflected by the euclidean distance in N-space:

```
def distance(num_dimensions,first_point,second_point):
    tmp = []
    for ind,val in enumerate(first_point):
        tmp.append(math.pow((second_point[ind] - val),2))
    return math.sqrt(sum(tmp))
```

Notice that we explicitly loop through each of the values here, take the square and then the square root of the total.

