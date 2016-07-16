#Understanding Neural Networks

##Motivating Examples

Before we understand anything about Neural Networks, let's make sure Neural Networks are worth learning:

* [SOMPY - full disclosure, I'm a contributor](https://github.com/sevamoo/SOMPY/tree/master/SOMPY)
    * [Clustering Data](http://nbviewer.jupyter.org/gist/sevamoo/e93699fdb481de1a932b)
    * [Low Level Representation](http://nbviewer.jupyter.org/urls/gist.githubusercontent.com/sevamoo/01753060b059941b4545/raw/1c9597257308c63fe601478f6e6d0d6d63fa80a3/somPy%20Example2)
* [word2vec - word similarity/word algebra](https://radimrehurek.com/gensim/models/word2vec.html)
* [facial recognition](https://cmusatyalab.github.io/openface/)
    * [training a classifier](https://cmusatyalab.github.io/openface/demo-3-classifier/)

##What is it?

A Neural Network is loosely, a set of techniques in machine learning techniques to create mathematical models.  The reason I say a set of techniques, is because I like to think of neural networks as a framework for guessing the right mathematical model.  

So how can we use a mathematical description of a thing?  Well, we can maximize it or minimize it by using calculus - this is known as optimization.  We can predict it's next value, by plugging in the necessary variables.  We can describe the process or thing with precision.  We can also compare it mathematical to other things, by looking at models with similar behavior.  More or less, with a mathematically accurate description of a thing, we know it precisely.

###Interpretting what a neural network does

So far we've defined, loosely what a neural network is, mathematically.  Now we will describe how to interpret the actions a neural network will take on a set of data to produce a result.  

[A topological interpretation](http://colah.github.io/posts/2014-03-NN-Manifolds-Topology/) 

A neural network is a set of non-linear transformations, that make [linearly non-seperable](https://en.wikipedia.org/wiki/Linear_separability) differences, and exagerate them.  Then predictions are made that take a mathematical object, from mathematical space A to mathematical space B.

<img src="https://github.com/EricSchles/intro_to_ml/blob/master/lectures/original_data.png" height=300 width=300 />
<img src="https://github.com/EricSchles/intro_to_ml/blob/master/lectures/arrow.png" height=50 width=100 />
<img src="https://github.com/EricSchles/intro_to_ml/blob/master/lectures/non_linear_projection.png" height=300 width=300 />

As you can see in the above picture the prediction is the classification of data points.  In the original data set drawing a straight line wouldn't be possible, because of the way the sample data was organized.  However by applying non-linear transformations in succession, the neural network is able to make a linear prediction, classifying the two lines correctly.

Let's look now at the tensorflow playground to get a better sense of how this might work in practice:

[tensorflow playground - spiral data](http://playground.tensorflow.org/#activation=tanh&regularization=L2&batchSize=30&dataset=spiral&regDataset=reg-plane&learningRate=0.1&regularizationRate=0.01&noise=20&networkShape=7,3&seed=0.47875&showTestData=false&discretize=false&percTrainData=80&x=true&y=true&xTimesY=false&xSquared=true&ySquared=true&cosX=false&sinX=true&cosY=false&sinY=true&collectStats=false&problem=classification&initZero=false)

[tensorflow playground - alternative network](http://playground.tensorflow.org/#activation=relu&regularization=L2&batchSize=30&dataset=spiral&regDataset=reg-plane&learningRate=0.1&regularizationRate=0.001&noise=0&networkShape=7,6,5,4,3,4&seed=0.66160&showTestData=false&discretize=false&percTrainData=80&x=true&y=true&xTimesY=false&xSquared=true&ySquared=true&cosX=false&sinX=true&cosY=false&sinY=true&collectStats=false&problem=classification&initZero=false)

Using the above example we can begin to test our model.  We'll carry out the following natural experiments:

1. Test the model above with regenerated data
2. Test the model above against all the data points
3. Test the model above with different feature transformations
4. Test the model above with different hyper parameters:
    * L1 v L2
    * Tanh v Linear v ReLU v Sigmoid
    * Classification v Regression
    * Learn Rate
    * Train Rate
    * Noise level
    * Batch Size

Conclusion:

Neural networks are fickle.  Unlike simpler models like linear regression or decision trees, neural networks require lots of care.  Unlike the above models, you can't just clean your data once, decide what variables you want to use, and then run your predictions.  You need to consider a whole bunch of stuff.  But if you're thoughtful enough, then a neural network can deliver superior conclusions.  Even on hard problems.  

Follow up question:

Of course, what if you were just thoughtful about the feature transformations you carried out?  Or the prior probabilities you brought to any given model generation?  Could you, using simple models, do just as well as the neural network?  

The answer is, of course!  But machine time is less valuable than people time.  So, the idea for using anything at this level of sophistication, is about having to think less hard.  Unless there was a sizeable gain in both what these models can do (maximization), and how thoughtful you need to be (minimization), then they wouldn't be used!

#Make your own Neural Network!

##Prerequistes

The goal of what we are trying to do with this talk is build up the right intuition.  The intuition of interpretation that you saw above might make sense already, in which case, you're free to go, because you won't learn anything else new here.  But, if you aren't 100% you took away everything you're supposed, that's what the rest of this talk is about - building up the intuition for what we saw, by understanding the techniques that led to the discovery of neural networks.  This will be the intellectual data, that produced the above model.

To truly understand how neural networks create the output they do, an important first step is understanding how to write your own neural network.  By analyzing and understanding the algorithms that go into this technique we can start to understand what's going on and why.

What we'll need is two mathematical techniques - the dot product from linear algebra and the derivative from calculus.  These two gems are all you are really doing when you apply a neural network to a bunch of data.  And one algorithm - stochastic gradient descent.  All the other stuff we saw above isn't actually part of the neural network, those are general techniques that are used with ANY machine learning algorithm.

##Explanation of the derivative

The derivative (from calculus) is measuring the instantaneous rate of change at a point.  So if the derivative is positive the mathematical function will increase as you plug in bigger values than the current point.  If the derivative is negative the mathematical function will decrease as you plug in bigger values than the current point.  

An example of the derivative is the power rule on x^n:

`f = x^2`
`f' = 2*x^1`

In general:

`f = x^n`
`f' = n*x^(n-1)`

##Explanation of matrices

A matrix is a representation of a mathematical equation where each column is a variable and each row is an equation.  And the row and column combination is a coefficient on that variable. 

So, `5x + 6y` and `7x + 4y` becomes:

```
[[5 6],
 [7 4]]
```

##Explanation of the dot product

The dot product is when you take two rows of a matrix and create a single scalar (aka number).

So `np.dot([5 6],[7 4])` is really doing:

`5*7 + 6*4 = 35 + 24 = 59`

It's important to note that there is a geometric interpretation of the dot product which will be helpful for later!

The dot product: A . B (read A dot B) is the combination of two vectors (since vectors have magnitude and position)

and it's equal to magnitude of `A * magnitiude` of `B * cosine` of theta, read:

`A . B = ||A|| ||B|| * cos(theta)`

So the dot product gives us information about our vectors (aka arrays).  Specifically if `A . B = 0` then our vectors are orthogonal!  That means directionally they are 90 degrees (or a right angle) away from each other.  

Knowing the direction of two things is going to be very helpful for us (which is what the dot product tells us).  It tells us if they are moving closer together or if they are moving further apart!

##Matrix multiplication

Matrix multiplication is the same thing as the dot product, except you do it over each row of the first matrix and each column of the second.  This gives you a new matrix.  Let's look at some examples:

```
A = [[1 2]
	 [3 4]]
B = [[1 0]
     [0 1]]
```

Then `A * B` is

`[1*1 + 0*3 0*1 + 2*1]`
`[3*1 + 0*4 3*0 + 4*1]`

Which is 

```[[1 2]
    [3 4]]```

Wait a second! Doesn't this mean that we just ended up where we started!?  That's because B in this case is the identity matrix.  It's the matrix version of the number 1.  In general we can think of B as the transformation matrix, which applies transformations to the matrix A to translate it magnitudinally and directionally in space.  

So, we can think of matrix multiplication as a transformation to our first matrix.  

This notion of transformation is going to be very, very important.  

##Putting it together

So we can measure whether things are going to go up in the very near future and whether things are going down in the very near future (using the derivative).  And we can change things (using transformation matrices via matrix multiplication).  Using these two techniques, I claim we can guess an initial mathematical model and then check to see how good our guess was (with the derivative).  Then we can transform our model using matrix multiplication.  And then guess again, until our model get's good enough.  

What I've just described is a very, very high level description of gradient descent.

##Neural networks, the prequel - [Newton's Method](http://www.ugrad.math.ubc.ca/coursedoc/math100/notes/approx/newton.html)

One of the main ways neural networks are trained is with an algorithm called stochastic gradient descent.  It's worth noting at this point, that's certainly not the only way! 

At the heart of neural networks is a technique called stochastic gradient descent.  Neural networks primarily operate over matrices and so the implementation of SGD can be challenging.  In order to draw the necessary intuition about our model let's look at another technique that makes use of SGD - [newton's method](http://www.ugrad.math.ubc.ca/coursedoc/math100/notes/approx/newton.html):

[Newton's method](http://www.ugrad.math.ubc.ca/coursedoc/math100/notes/approx/newton.html) is so simple to implement we're going to look at it first and then understand why it works:

```
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
```

So what's the goal?  To find all the roots of the function.  

To do this we use a really sophisticated method called "guess and check".  However, the difference between dumb guess and check and our guess and check, is we guess intelligently, over time.  Our first guess is usually pretty dumb - that's the stochastic (or random) part of the algorithm.  The rest of it is just gradient descent.

So here's our algorithm in all it's gory detail:

1. Look at the distance between our guess `x0`, applied to our function and the number zero - remember we are looking for the zeroes of the function, so when `f(x0) = 0` we are do.

`delta = dx(f,x0)`

2. We don't necessarily need our `delta` to be zero, because this is computer science after all, not math.  But we want to make sure we are sufficiently close to zero.  So:

`while delta > e:`

which says, while our `delta` is greater than some constant hyper parameter `e`, keep on guessing.

3. We update our guess `x0` intelligently at this point:

`x0 = x0 - (f(x0)/deriv(f,x0))`

We look at what we are currently guessing and then we adjust it by the difference between the current value of `f(x0)` and the derivative of `f` at `x0`.  Recall that the derivative is the instantaneous rate of change at the point `x0`.  So we can think of this ratio as the size of `f(x0)` proportional to `f(x0)` varies.  

So let's say that f(x0) is relatively small, but the next value, some epsilon distance greater than x0, say x0+epsilon, is significantly greater - that would mean that we'd have a large positive derivative.  And therefore we'd get a very small contribution to the term `(f(x0)/deriv(f,x0))`.  Meaning our new `x0` would be very close to our old `x0`.

Applying this logic further we see the following pattern:

derivative assumed to be positive:

f(x0) is positive (larger than 1), deriv(f,x0) is big -> small negative contribution to change in x0

f(x0) is positive (larger than 1), deriv(f,x0) is small (less than 1) -> large negative contribution to change in x0

f(x0) is negative (less than -1), deriv(f,x0) is big -> small positive contribution to change in x0

f(x0) is negative (less than -1), deriv(f,x0) is small (less than 1) -> large positive contribution to change in x0

We can also flip the sign of the derivative and look at the associated results, they are equivalent to these.  So without loss of generality, we see the key insight:

Newton's method heads towards solution - finding the zeroes of the function by looking at the dynamics of how the function changes for different values of our guess.  And based on these dynamics and function information, chooses a new value for our guess.  

4. We update our `delta` by looking at the difference between `f(x0)` compare to the value zero:

`delta = dx(f,x0)` 

Again, if delta is sufficiently close to zero, we stop.

After thoughts:

This is more or less the process of approximation that stochastic gradient descent takes.  However, there are a number of methods for doing the process of stochastic gradient descent.  [Wikipedia's SGD page](https://en.wikipedia.org/wiki/Stochastic_gradient_descent) illustrates some of these methods.

##Our first example neural network

This first example comes from [Andrew Trask](http://iamtrask.github.io/2015/07/12/basic-python-network/) - not only is he awesome for having an evil super villian name, he's a super nice guy!  And very good at explaining things.  

This code is ripped off from the above blog post and highly worth reading:


```
X = np.array([ [0,0,1],[0,1,1],[1,0,1],[1,1,1] ])
y = np.array([[0,1,1,0]]).T
syn0 = 2*np.random.random((3,4)) - 1
syn1 = 2*np.random.random((4,1)) - 1
for j in xrange(60000):
    l1 = 1/(1+np.exp(-(np.dot(X,syn0))))
    l2 = 1/(1+np.exp(-(np.dot(l1,syn1))))
    l2_delta = (y - l2)*(l2*(1-l2))
    l1_delta = l2_delta.dot(syn1.T) * (l1 * (1-l1))
    syn1 += l1.T.dot(l2_delta)
    syn0 += X.T.dot(l1_delta)
```

So what does this code do?

It set's up 2 matrices, an input matrix and an output matrix.  Affectionately called X and y.  

```
X = np.array([ [0,0,1],[0,1,1],[1,0,1],[1,1,1] ])
y = np.array([[0,1,1,0]]).T
```

Then synapses are created - these our initial guess matrices.  

```
syn0 = 2*np.random.random((3,4)) - 1
syn1 = 2*np.random.random((4,1)) - 1
```
Then we get our layers - l1 and l2 - here we are using a sigmoid function to and the dot product to chain our matrices together.  

```
l1 = 1/(1+np.exp(-(np.dot(X,syn0))))
l2 = 1/(1+np.exp(-(np.dot(l1,syn1))))
```

Then we get the deltas - by applying the derivative of the sigmoid function to the last layer against our output matrix.  And we also apply our delta from our last layer to our first layer, readjusting the weights by the amount we were off.

```
l2_delta = (y - l2)*(l2*(1-l2))
l1_delta = l2_delta.dot(syn1.T) * (l1 * (1-l1))
```

And then we update our guess matrices by how much we were off in the last step.

```
syn1 += l1.T.dot(l2_delta)
syn0 += X.T.dot(l1_delta)
```

Okay, so that's it - you can all go home.  Bye!

Just kidding `:P`

##Neural networks are a framework

Remember at the beginning of the talk when I said neural networks are a framework?  That's because they aren't all implemented using the above techniques.  The derivative, matrix multiplication and the sigmoid function are just parameters that we plugged in to do one thing - guess and check.  So if we generalize this out and cut away those specifics, what are we left with?  

##The steps of a neural network

Initialization - Initialize a random set of weights for our guess.  These weights are called our synapses, because there is an element of randomness.  Synapses in the brain fire randomly.  It's a poetic, but overly intuitive description of what's going on.

Forward propagation - Create our layers layers from our synapses and our input data.

Back propagation - Check our guess against our output data and then send the error back through to our synapses.

That's it.  There are lots of techniques and optimizations to neural networks, but this is the basic idea in a nutshell.  

So where do we go from here?  

Well, obvious our code doesn't scale at all.  So let's make it better, step by step:

First, it's important to note we'll make everything into functions.

So let's define our initialization function:

```
syn0 = 2*np.random.random((3,4)) - 1
syn1 = 2*np.random.random((4,1)) - 1
```

Becomes:

```
def create_connection(num_rows,num_cols):
    return 2*np.random.random((num_rows,num_cols)) -1
```

AND

```
def create_nn(input_data,output_data,num_hidden_layers):
    nn = [{"name":"input data","connection":input_data}]
    #input layer
    input_syn = {"name":"input layer"}
    input_syn["connection"] = create_connection(len(input_data[0]),len(input_data))
    nn.append(input_syn)
    #hidden layers
    for i in xrange(num_hidden_layers):
        syn = {"name":i}
        syn["connection"] = create_connection(len(input_data),len(input_data))
        nn.append(syn)
    #output_layer
    syn = {"name":"output layer"}
    syn["connection"] = create_connection(len(output_data),len(output_data[0]))
    nn.append(syn)
    nn.append({"name":"output data","connection":output_data})
    return nn
```

This is the first our first big change - parameterizing the creation of our synapses.  The synapses become the model that we apply to new data after training.  By parameterizing their depth and breathe we can have a generalized model that can be used to approximate a wide range of models.  

Note that the parameterization primarily comes in at the hidden synapses.  The input and output synpases are more or less fixed.  

The next step will be applying the synapses to create layers and do forward propagation:

```
l1 = 1/(1+np.exp(-(np.dot(X,syn0))))
l2 = 1/(1+np.exp(-(np.dot(l1,syn1))))
```

Becomes:

```
#The sigmoid function and it's derivative
def nonlin(x,deriv=False):
    if(deriv==True):
	return x*(1-x)
    return 1/(1+np.exp(-x))
```
 
 AND

```
def forward_propagate(synapses):
    layers = [synapses[0]["connection"]]
    for ind,synapse in enumerate(synapses[:-1]):
        if ind == 0: continue
        layers.append(
            nonlin(np.dot(layers[ind-1],synapse["connection"]))
        )
    return layers
```

Here we simply loop through our synapses applying the sigmoid function and doting the matrices like we did before.  Pretty straight forward right?  Neural networks are very easy to apply - partially because of the choice of simple functions with obvious derivatives.  Also, linear algebra is well tred territory for computer scientists, so you get a lot of that for free.  (Not that it's hard to implement)

Now for the hard part - "teaching" our model - with back propagation:

```
l2_delta = (y - l2)*(l2*(1-l2))
l1_delta = l2_delta.dot(syn1.T) * (l1 * (1-l1))
```

AND

```
syn1 += l1.T.dot(l2_delta)
syn0 += X.T.dot(l1_delta)
```

Becomes:

```
def back_propagate(layers,synapses):
    errors = [synapses[-1]["connection"] - layers[-1]]
    synapses_index = -1
    layers_index = -1
    errors_index = 0
    deltas_index = 0
    deltas = []
    while len(layers) - abs(layers_index) > 0:
        deltas.append(errors[errors_index]*nonlin(layers[layers_index],deriv=True))
        synapses_index -= 1
        layers_index -= 1
        errors.append(deltas[deltas_index].dot(synapses[synapses_index]["connection"].T))
        errors_index += 1
        deltas_index += 1
    synapses_index = -2
    layers_index = -2
    deltas_index = 0
    while len(layers) - abs(layers_index) >= 0:
        synapses[synapses_index]["connection"] += layers[layers_index].T.dot(deltas[deltas_index])
        synapses_index -= 1
        layers_index -= 1
        deltas_index += 1
    return synapses,errors[0]
```

This is the hardest part - mostly because we have to apply the deltas in reverse, starting at the back.  And since we are essentially starting at the bottom of a potentially very wide and deep tree (stored as an array) we have to do some weird things with indexing.  I leave it as an exercise to verify this works (it does but it's hard to explain).  

And now we are done!

So,

```
X = np.array([ [0,0,1],[0,1,1],[1,0,1],[1,1,1] ])
y = np.array([[0,1,1,0]]).T
syn0 = 2*np.random.random((3,4)) - 1
syn1 = 2*np.random.random((4,1)) - 1
for j in xrange(60000):
    l1 = 1/(1+np.exp(-(np.dot(X,syn0))))
    l2 = 1/(1+np.exp(-(np.dot(l1,syn1))))
    l2_delta = (y - l2)*(l2*(1-l2))
    l1_delta = l2_delta.dot(syn1.T) * (l1 * (1-l1))
    syn1 += l1.T.dot(l2_delta)
    syn0 += X.T.dot(l1_delta)
```

Becomes:

```
def run_once(num_hidden_nodes,X,y):
    np.random.seed(1)
    errors = []
    nn = create_nn(X,y,num_hidden_nodes)
    for j in xrange(70000):
        layers = forward_propagate(nn)
        nn,error = back_propagate(layers,nn)
        if j%10000 == 0:
            errors.append(np.mean(np.abs(error)))
    return errors

X = np.array([[1,1],
              [1,0],
              [0,1],
              [0,0]])
            
y = np.array([[0],
              [1],
              [1],
              [1]])
run_once(3,X,y)
```

And we are done!  So let's see it go!  To get the source code, check out:

[back_prop.py](https://github.com/EricSchles/intro_to_ml/blob/master/code/neural_networks/back_prop.py)

To run it you'll need numpy and python2.7 (I know I know I should be using 3).  

Then download the code, save it to a file, open a terminal, go to the directory where you saved the code and you'll just do:

`python back_prop.py`

And that should be it.

##Recurrent Neural Networks

##Convolutional Neural networks


##Neural network applications

A neural stack - teaching a neural network when to push and pop from a stack:

https://github.com/EricSchles/intro_to_ml/blob/master/code/neural_networks/neural_stack.py

Flappy bird:

Mac OSX:

Python3 - It's hard to install pygame on python2 (lame I know)
tensorflow: https://www.tensorflow.org/versions/r0.7/get_started/os_setup.html#pip-installation
opencv: https://jjyap.wordpress.com/2014/05/24/installing-opencv-2-4-9-on-mac-osx-with-python-support/ - Python2
opencv: http://stackoverflow.com/questions/33222965/installing-opencv-3-for-python-3-on-a-mac-using-homebrew-and-pyenv - Python3
pygame: http://pygame.org/wiki/macintosh

https://porter.io/github.com/yenchenlin1994/DeepLearningFlappyBird

Somehow this breaks numpy but then you can just install it again from pip and then everything is fine for some reason?

Neural artwork:

https://github.com/EricSchles/image-analogies

you'll need pillow (not necssarily installed)

do:

`python3 image_analogy.py --vgg-weights vgg16_weights.h5 images/arch-mask.jpg images/arch.jpg images/arch-newmask.jpg out/arch`

Facial Recognition:

https://github.com/EricSchles/neuralnet/blob/master/facial_recognition.py

Picture annotation:

http://cs.stanford.edu/people/karpathy/deepimagesent/

##And Now For Something Completely Different

##References:

http://colah.github.io/
http://www.wildml.com/
http://iamtrask.github.io/