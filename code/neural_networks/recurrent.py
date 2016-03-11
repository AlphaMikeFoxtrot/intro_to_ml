import copy, numpy as np

# compute sigmoid nonlinearity
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

# convert output of sigmoid function to its derivative
def sigmoid_prime(output):
    return output*(1-output)

def create_connection(num_rows,num_cols):
    return 2*np.random.random((num_rows,num_cols)) -1

def integer2binary(binary_dim=8):
    # training dataset generation
    int2binary = {}

    largest_number = pow(2,binary_dim)
    binary = np.unpackbits(
        np.array([range(largest_number)],dtype=np.uint8).T,axis=1)
    for i in range(largest_number):
        int2binary[i] = binary[i]
    return int2binary,largest_number

def create_training_data(operator,size,largest_number,int2binary):
    # training logic
    training_data = []
    for j in range(size):
        tmp = {}
        # generate a simple addition problem (a + b = c)
        tmp["a_int"] = np.random.randint(largest_number/2) # int version
        tmp["a"] = int2binary[tmp["a_int"]] # binary encoding

        tmp["b_int"] = np.random.randint(largest_number/2) # int version
        tmp["b"] = int2binary[tmp["b_int"]] # binary encoding

        # true answer
        tmp["c_int"] = operator(tmp["a_int"], tmp["b_int"])
        tmp["c"] = int2binary[tmp["c_int"]]

        # where we'll store our best guess (binary encoded)
        tmp["d"] = np.zeros_like(tmp["c"])
        training_data.append(tmp)
    return training_data

def create_synapses(input_data,output_data,input_dim,hidden_dim,output_dim,num_hidden_layers):
    nn = [{"name":"input data","connection":input_data}]
    #input layer
    input_syn = {"name":"input layer"}
    input_syn["connection"] = create_connection(input_dim,hidden_dim)
    nn.append(input_syn)
    #hidden layers
    for i in xrange(num_hidden_layers):
        syn = {"name":i}
        syn["connection"] = create_connection(hidden_dim,hidden_dim)
        nn.append(syn)
    #output_layer
    syn = {"name":"output layer"}
    syn["connection"] = create_connection(hidden_dim,output_dim)
    nn.append(syn)
    nn.append({"name":"output data","connection":output_data})
    return nn

def create_matrices(training_data,binary_dim):
    x_s = []
    y_s = []
    for datum in training_data:
        for position in range(binary_dim):
            x_s.append(np.array([[datum["a"][binary_dim - position - 1],datum["b"][binary_dim - position - 1]]]))
            y_s.append( np.array([[datum["c"][binary_dim - position - 1]]]).T)
    return x_s,y_s


def forward_prop(datum,binary_dim):
    for position in range(binary_dim):
        # generate input and output
        X = np.array([[datum["a"][binary_dim - position - 1],datum["b"][binary_dim - position - 1]]])
        y = np.array([[datum["c"][binary_dim - position - 1]]]).T
        # hidden layer (input ~+ prev_hidden)
        layer_1 = sigmoid(np.dot(X,synapse_0) + np.dot(layer_1_values[-1],synapse_h))

        # output layer (new binary representation)
        layer_2 = sigmoid(np.dot(layer_1,synapse_1))

        # did we miss?... if so, by how much?
        layer_2_error = y - layer_2
        layer_2_deltas.append((layer_2_error)*sigmoid_prime(layer_2))
        overallError += np.abs(layer_2_error[0])
    
        # decode estimate so we can print it out
        training_data[index]["d"][binary_dim - position - 1] = np.round(layer_2[0][0])
        
        # store hidden layer so we can use it in the next timestep
        layer_1_values.append(copy.deepcopy(layer_1))
    return X,y,layer_1,layer_1_values, layer_2,layer_2_error,layer_2_deltas, overallError
        
np.random.seed(0)
binary_dim = 8
int2binary,largest_number = integer2binary(binary_dim=binary_dim)
# input variables
alpha = 0.1
input_dim = 2
hidden_dim = 16
output_dim = 1
size = 30000

# initialize neural network weights
#create_synapses(input_dim
synapse_0 = create_connection(input_dim,hidden_dim)
synapse_1 = create_connection(hidden_dim,output_dim)
synapse_h = create_connection(hidden_dim,hidden_dim)

synapse_0_update = np.zeros_like(synapse_0)
synapse_1_update = np.zeros_like(synapse_1)
synapse_h_update = np.zeros_like(synapse_h)

training_data = create_training_data(lambda x,y: x+y,size,largest_number,int2binary)
input_matrices,output_matrices = create_matrices(training_data,binary_dim)
                
for index,input_matrix in enumerate(input_matrices):
    overallError = 0
    X = input_matrix
    y = output_matrices[index]
    layer_2_deltas = list()
    layer_1_values = list()
    layer_1_values.append(np.zeros(hidden_dim))
    
    # moving along the positions in the binary encoding
    for position in range(binary_dim):
        
    
    future_layer_1_delta = np.zeros(hidden_dim)
    
    for position in range(binary_dim):
        
        X = np.array([[training_data[index]["a"][position],training_data[index]["b"][position]]])
        layer_1 = layer_1_values[-position-1]
        prev_layer_1 = layer_1_values[-position-2]
        
        # error at output layer
        layer_2_delta = layer_2_deltas[-position-1]
        # error at hidden layer
        layer_1_delta = (future_layer_1_delta.dot(synapse_h.T) + layer_2_delta.dot(synapse_1.T)) * sigmoid_prime(layer_1)

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
    if(index % 1000 == 0):
        print "Error:" + str(overallError)
        print "Pred:" + str(training_data[index]["d"])
        print "True:" + str(training_data[index]["c"])
        out = 0
        for index,x in enumerate(reversed(training_data[index]["d"])):
            out += x*pow(2,index)
        print str(training_data[index]["a_int"]) + " + " + str(training_data[index]["b_int"]) + " = " + str(out)
        print "------------"

        
