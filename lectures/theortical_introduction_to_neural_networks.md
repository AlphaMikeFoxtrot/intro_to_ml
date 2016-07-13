#Neural Networks

##What is it?

A Neural Network is loosely, a set of techniques in machine learning techniques to create mathematical models.  The reason I say a set of techniques, is because I like to think of neural networks as a framework for guessing the right mathematical model.  

So how can we use a mathematical description of a thing?  Well, we can maximize it or minimize it by using calculus - this is known as optimization.  We can predict it's next value, by plugging in the necessary variables.  We can describe the process or thing with precision.  We can also compare it mathematical to other things, by looking at models with similar behavior.  More or less, with a mathematically accurate description of a thing, we know it precisely.

###Interpretting what a neural network does

So far we've defined, loosely what a neural network is, mathematically.  Now we will describe how to interpret the actions a neural network will take on a set of data to produce a result.  

[A topological interpretation](http://colah.github.io/posts/2014-03-NN-Manifolds-Topology/) 

A neural network is a set of non-linear transformations, that make linearly non-seperable differences, and exagerate them.  Then predictions are made that take a mathematical object, from mathematical space A to mathematical space B.

![](original_data.png = 100x20)
![](non_linear_projection.png = 100x20)

##Prerequistes

Most people just launch into the intuition of a neural network.  They try to tell you why they chose the layout they did.  But I'm here to tell you that the intuition is wrong and lies.  They came up with it after, which will be evident when you compare "neural networks" and how they actually work to other things.  There is a strong need in computer science to run very far away from mathematics.  Well, not today.  Because trust me, it will actually make things a lot easier.

What we'll need is two techniques - the dot product from linear algebra and the derivative from calculus.  These two gems are all you are really doing when you apply a neural network to a bunch of data.  

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

So we can measure whether things are going to go up in the very near future and whether things are going down in the very near future (using the derivative).  And we can change things (using transformation matrices via matrix multiplication).  Using these two techniques, I claim we can guess an initial mathematical model and then check to see how good our guess was (with the derivative).  Then we can transform our model using matrix multiplication.  And then guess again, until our model get's good enough.  Once our model is good enough, we can use it do all the fun things I mentioned at the beginning.  

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

As you may have guessed there are lots and lots of kinds of neural networks out there.  This is because neural networks are a frame work - a way of doing things.  They have some things in common and some things are different.  The general idea behind neural networks is we treat our model as a data structure and apply some stuff (typically mathematical stuff) to the data to get a way of doing that task.  The general features of a neural network are:

forward propagation or something like it
back propagation or something like it.
Some sense of structure.

We'll see our first example of a different kind of structure here with recurrent neural networks.  

So what is a recurrent neural network?  It's a neural network that makes use of recurrence to continually apply and update the weights to the hidden layers of the network.  So rather than just taking in the last set of weights, the hidden layers in a recurrent neural network remember all of (or some of) the previous weights of the that were applied to it.  Meaning the order that you train your data on the network affect how the model will learn.  

Therefore there is a sense of time in recurrent neural networks and an assumption that your data is linked, from result to result.  This type of network is very good for working over text, since the order matters.  But it will also be good for any type of data where there is a sense of order.  Like for instance weather patterns have (some) sense of cause and effect.  

Again, I'm going to flat out steal from [Andrew Trask](https://iamtrask.github.io//2015/11/15/anyone-can-code-lstm/) because he's very good at writing minimal intuitive code:

```
import copy, numpy as np
np.random.seed(0)

# compute sigmoid nonlinearity
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

# convert output of sigmoid function to its derivative
def sigmoid_output_to_derivative(output):
    return output*(1-output)


# training dataset generation
int2binary = {}
binary_dim = 8

largest_number = pow(2,binary_dim)
binary = np.unpackbits(
    np.array([range(largest_number)],dtype=np.uint8).T,axis=1)
for i in range(largest_number):
    int2binary[i] = binary[i]


# input variables
alpha = 0.1
input_dim = 2
hidden_dim = 16
output_dim = 1


# initialize neural network weights
synapse_0 = 2*np.random.random((input_dim,hidden_dim)) - 1
synapse_1 = 2*np.random.random((hidden_dim,output_dim)) - 1
synapse_h = 2*np.random.random((hidden_dim,hidden_dim)) - 1

synapse_0_update = np.zeros_like(synapse_0)
synapse_1_update = np.zeros_like(synapse_1)
synapse_h_update = np.zeros_like(synapse_h)

# training logic
for j in range(10000):
    
    # generate a simple addition problem (a + b = c)
    a_int = np.frandom.randint(largest_number/2) # int version
    a = int2binary[a_int] # binary encoding

    b_int = np.random.randint(largest_number/2) # int version
    b = int2binary[b_int] # binary encoding

    # true answer
    c_int = a_int + b_int
    c = int2binary[c_int]
    
    # where we'll store our best guess (binary encoded)
    d = np.zeros_like(c)

    overallError = 0
    
    layer_2_deltas = list()
    layer_1_values = list()
    layer_1_values.append(np.zeros(hidden_dim))
    
    # moving along the positions in the binary encoding
    for position in range(binary_dim):
        
        # generate input and output
        X = np.array([[a[binary_dim - position - 1],b[binary_dim - position - 1]]])
        y = np.array([[c[binary_dim - position - 1]]]).T

        # hidden layer (input ~+ prev_hidden)
        layer_1 = sigmoid(np.dot(X,synapse_0) + np.dot(layer_1_values[-1],synapse_h))

        # output layer (new binary representation)
        layer_2 = sigmoid(np.dot(layer_1,synapse_1))

        # did we miss?... if so, by how much?
        layer_2_error = y - layer_2
        layer_2_deltas.append((layer_2_error)*sigmoid_output_to_derivative(layer_2))
        overallError += np.abs(layer_2_error[0])
    
        # decode estimate so we can print it out
        d[binary_dim - position - 1] = np.round(layer_2[0][0])
        
        # store hidden layer so we can use it in the next timestep
        layer_1_values.append(copy.deepcopy(layer_1))
    
    future_layer_1_delta = np.zeros(hidden_dim)
    
    for position in range(binary_dim):
        
        X = np.array([[a[position],b[position]]])
        layer_1 = layer_1_values[-position-1]
        prev_layer_1 = layer_1_values[-position-2]
        
        # error at output layer
        layer_2_delta = layer_2_deltas[-position-1]
        # error at hidden layer
        layer_1_delta = (future_layer_1_delta.dot(synapse_h.T) + layer_2_delta.dot(synapse_1.T)) * sigmoid_output_to_derivative(layer_1)

        # let's update all our weights so we can try again
        synapse_1_update += np.atleast_2d(layer_1).T.dot(layer_2_delta)
        synapse_h_update += np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
        synapse_0_update += X.T.dot(layer_1_delta)
        
        future_layer_1_delta = layer_1_delta
    

    synapse_0 += synapse_0_update * alpha
    synapse_1 += synapse_1_update * alpha
    synapse_h += synapse_h_update * alpha    

    synapse_0_update *= 0
    synapse_1_update *= 0
    synapse_h_update *= 0
    
    # print out progress
    if(j % 1000 == 0):
        print "Error:" + str(overallError)
        print "Pred:" + str(d)
        print "True:" + str(c)
        out = 0
        for index,x in enumerate(reversed(d)):
            out += x*pow(2,index)
        print str(a_int) + " + " + str(b_int) + " = " + str(out)
        print "------------"
```

Okay, so this code isn't quiet as minimal.  But it's as minimal as it gets.  And really this isn't sooooo bad.  Let's break it down:

Here he's setting things up, just like he did before.  

```
# initialize neural network weights
synapse_0 = 2*np.random.random((input_dim,hidden_dim)) - 1
synapse_1 = 2*np.random.random((hidden_dim,output_dim)) - 1
synapse_h = 2*np.random.random((hidden_dim,hidden_dim)) - 1

synapse_0_update = np.zeros_like(synapse_0)
synapse_1_update = np.zeros_like(synapse_1)
synapse_h_update = np.zeros_like(synapse_h)
```

Here he's doing forward propagation.  

```
# moving along the positions in the binary encoding
for position in range(binary_dim):
    
    # generate input and output
    X = np.array([[a[binary_dim - position - 1],b[binary_dim - position - 1]]])
    y = np.array([[c[binary_dim - position - 1]]]).T

    # hidden layer (input ~+ prev_hidden)
    layer_1 = sigmoid(np.dot(X,synapse_0) + np.dot(layer_1_values[-1],synapse_h))

    # output layer (new binary representation)
    layer_2 = sigmoid(np.dot(layer_1,synapse_1))
```



This is the new stuff - here we are setting up the recurrence:

```
    # did we miss?... if so, by how much?
    layer_2_error = y - layer_2
    layer_2_deltas.append((layer_2_error)*sigmoid_output_to_derivative(layer_2))
    overallError += np.abs(layer_2_error[0])

    # decode estimate so we can print it out
    d[binary_dim - position - 1] = np.round(layer_2[0][0])

    # store hidden layer so we can use it in the next timestep
    layer_1_values.append(copy.deepcopy(layer_1))

future_layer_1_delta = np.zeros(hidden_dim)
```

specifically when we do this:

`layer_1_values.append(copy.deepcopy(layer_1))` AND `future_layer_1_delta = np.zeros(hidden_dim)`

Notice that we have an array of `layer_1_values` where we store all the previous layer one's.  This is the saving part of recurrence.

Next let's look at back propagation.  Unfortunately, as complex as back propagation was before, it's even more so.  So much so it has a new name, back propagation through time:

```
for position in range(binary_dim):
        
        X = np.array([[a[position],b[position]]])
        layer_1 = layer_1_values[-position-1]
        prev_layer_1 = layer_1_values[-position-2]
        
        # error at output layer
        layer_2_delta = layer_2_deltas[-position-1]
        # error at hidden layer
        layer_1_delta = (future_layer_1_delta.dot(synapse_h.T) + layer_2_delta.dot(synapse_1.T)) * sigmoid_output_to_derivative(layer_1)

        # let's update all our weights so we can try again
        synapse_1_update += np.atleast_2d(layer_1).T.dot(layer_2_delta)
        synapse_h_update += np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
        synapse_0_update += X.T.dot(layer_1_delta)
        
        future_layer_1_delta = layer_1_delta
    

    synapse_0 += synapse_0_update * alpha
    synapse_1 += synapse_1_update * alpha
    synapse_h += synapse_h_update * alpha    

    synapse_0_update *= 0
    synapse_1_update *= 0
    synapse_h_update *= 0
```

Notice the big differences here - we need the previous layer one with `prev_layer_1` and also we have synapse updates for each synapse:


```
synapse_1_update += np.atleast_2d(layer_1).T.dot(layer_2_delta)
synapse_h_update += np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
synapse_0_update += X.T.dot(layer_1_delta)

future_layer_1_delta = layer_1_delta
```

This the the back propagation through time part - because this is happening in a loop and because we look at the previous layer ones.  

So let's pause for a minute - if we wanted to generalize this out, we'd need to do this for each layer right?  And keep lists of lists for each synapse that we are back propagating through time.  Ugh.  

It turns out writing this isn't particularly harder, but a lot more tedious and a lot of the intuition is lost.  So, I decided that it'd be better not to.  I tried (and am still trying) but it's tough from a time constraint perspective.  

So I leave it as an exercise to generalize out this code, like we did with the previous code.  

##Convolutional Neural networks

A convolution is a sliding window function applied to a matrix. Convolutional neural networks work differently than recurrent neural networks or vanilla neural networks.  Rather than learning over the entire space, convolutional neural networks work over a subset of the input data to learn local patterns.  

The canonical use-case for convolutional neural networks is an image with pixels.  The pixels are taken in and pieces of the picture are learned, to find locally represented features - much like the human eye recognizes distinct features to identify what humans would refer to as physical objects, like cars, planes, beds, cats, and pizza.  

Here's a great example of a simple convolutional neural network:

https://github.com/andersbll/nnet/blob/master/nnet/convnet/conv.pyx

We won't bother even trying to implement one our selves, but here's a strategy one could take to implement one yourself.  Take in the total size of your matrix, split it up into smaller matrices, then apply neural networks to each piece.  Then combine results to produce one score.  

So like, recurrent neural networks assumed a sense of time or an ordering, convolutional neural networks assume a sense of locality.  That things around a point will be similar.  If you have the network trained on say, pictures of a cat, you can have each piece learn the features specific to that part of the cats face, and then identify them.  


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