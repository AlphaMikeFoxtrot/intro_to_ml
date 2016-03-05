import numpy as np

#sigmoid function
def nonlin(x,derivative=False):
    if derivative:
        return x*(1-x)
    return 1/(1+np.exp(-x))

def inspect_input_node(X):
    number_of_hidden_layers = len(X)
    number_of_input_layers = len(X[0])
    return number_of_input_layers,number_of_hidden_layers

def initialize_synapse(number_of_input_nodes,number_of_output_nodes):
    """
    The synapse is a transient matrix that connects input layers to output layers.
    It can be used in a few ways:
    -To connect input layers to hidden layers:
    num_input_layers = 3
    num_hidden_layers = 4
    initialize_synapse(3,4)
    -To connect hidden layers to output layers:
    num_input_layers = 3
    num_output_layers = 1
    initialize_synapse(4,1)
    Note that the number of columns, is the number of  
    """
    return 2*np.random.random((number_of_input_nodes,number_of_output_nodes)) - 1 

def forward_propagation(input_nodes,synapses):
    layers = []
    layers.append(input_nodes)
    for ind,synapse in enumerate(synapses):
        layers.append(
            nonlin(np.dot(layers[ind],synapse))
        )
    return layers

def back_propagation(layers,output,synapses):
    deltas = []
    error = output - layers[-1]
    deltas.append(error*nonlin(layers[-1],derivative=True))
    syn_counter = len(synapses) - 1 
    delta_counter = 0
    while syn_counter > 0:
        error = deltas[delta_counter].dot(synapses[syn_counter].T)
        deltas.insert(0,error * nonlin(layers[syn_counter],derivative=True))
        syn_counter -= 1
        delta_counter += 1
    syn_counter = len(synapses) - 1
    while syn_counter > 0:
        synapses[syn_counter] += layers[syn_counter].T.dot(deltas[syn_counter])
        syn_counter -= 1
    return synapses

def main():
    num_hidden_layers = 2
    input_nodes = np.array(
        [
            [0,0,1,0],
            [0,1,1,0],
            [1,0,1,1],
            [1,1,1,1],
            [1,0,1,0]
        ]
    )

    output = np.array(
        [
            [0],
            [1],
            [1],
            [0],
            [0]
        ]
    )
    
    np.random.seed(1)
    num_input,num_hidden = inspect_input_node(input_nodes)
    synapses = []
    synapses.append(initialize_synapse(num_input,num_hidden))
    
    if num_hidden_layers > 1:
        for i in xrange(num_hidden_layers):
            synapses.append(initialize_synapse(num_hidden,num_hidden))

    synapses.append(initialize_synapse(num_hidden,1))

    for i in xrange(60000):
        layers = forward_propagation(input_nodes,synapses)
        back_propagation(layers,output,synapses)
        if i%10000 == 0:
            print "Error:",np.mean(np.abs(layers[-1]))
        
if __name__ == '__main__':
    main()
