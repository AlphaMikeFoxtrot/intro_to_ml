import numpy as np

def nonlin(x,deriv=False):
        if(deriv==True):
                return x*(1-x)
	return 1/(1+np.exp(-x))

def inspect_input_node(X):
        num_hidden = len(X)
        num_input = len(X[0])
        return num_input,num_hidden

def initialize_synapse(num_input,num_output):
        return 2*np.random.random((num_input,num_hidden)) - 1

def forward_propagate(last_layer,synapse):
        return nonlin(np.dot(last_layer,synapse))

def forward_propagation_a(input_nodes,synapses,num_layers):
        layers = [X]
        
	# Feed forward through layers 0, 1, and 2
        for ind in xrange(num_layers):
                layers.append(forward_propagate(layers[ind],synapses[ind]))
        return layers

def forward_propagation_b(input_nodes,synapses):
    layers = []
    layers.append(input_nodes)
    for ind,synapse in enumerate(synapses):
        layers.append(
            nonlin(np.dot(layers[ind],synapse))
        )
    return layers


def calculate_total_error(output,layers):
        error = output - layers[2]
        return error

num_hidden_layers = 2
X = np.array([[0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1]])
                
y = np.array([[0],
	      [1],
	      [1],
	      [0]])

np.random.seed(1)

synapses = []
deltas = []
num_input, num_hidden = inspect_input_node(X)
# randomly initialize our weights with mean 0

synapses.append( initialize_synapse(num_input,num_hidden) )

if num_hidden_layers > 1:
        synapses.append(initialize_synapse(num_hidden,num_hidden))
synapses.append( initialize_synapse(num_hidden,1) )

def calc_delta(error,layer):
        return error*nonlin(layer,deriv=True)

def calc_error(delta,synapse):
        return delta.dot(synapse.T)


for j in xrange(60000):

        #layers = forward_propagation_b(X,synapses)
        layers = forward_propagation_a(X,synapses,3)
        # how much did we miss the target value?
        error = calculate_total_error(y,layers)
        if j%10000 == 0:
                print "Error",np.mean(np.abs(error))
        # in what direction is the target value?
        # were we really sure? if so, don't change too much.
        deltas.append(calc_delta(error,layers[2]))
        
        # how much did each l1 value contribute to the l2 error (according to the weights)?
        error = calc_error(deltas[0],synapses[1])
    
        # in what direction is the target l1?
        # were we really sure? if so, don't change too much.
        deltas.append(calc_delta(error,layers[1]))

        synapses[1] += layers[1].T.dot(deltas[0])
        synapses[0] += layers[0].T.dot(deltas[1])
