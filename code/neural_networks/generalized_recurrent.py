import copy, numpy as np
import pandas as pd

# compute sigmoid nonlinearity
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

# convert output of sigmoid function to its derivative
def sigmoid_prime(output):
    return output*(1-output)

def create_connection(num_rows,num_cols):
    return 2*np.random.random((num_rows,num_cols)) -1

def find_binary_dim(number):
    count = 1 # we start with offset one to pad dimension
    while number > 1:
        count += 1
        number /= 2
    return count

def integer2binary(largest_number=256):
    int2binary = {}
    binary_dim = find_binary_dim(largest_number)
    binary = np.unpackbits(
        np.array([range(largest_number)],dtype=np.uint8).T,axis=1)
    for i in range(largest_number):
        int2binary[i] = binary[i]
    return int2binary

def create_training_data(operator,size,largest_number,int2binary):
    training_data = pd.DataFrame()
    for j in range(size):
        tmp = {}
        tmp["first_integer"] = np.random.randint(largest_number/2) 
        tmp["first_binary"] = int2binary[tmp["first_integer"]] 

        tmp["second_integer"] = np.random.randint(largest_number/2) 
        tmp["second_binary"] = int2binary[tmp["second_integer"]] 

        # true answer
        tmp["result_integer"] = operator(tmp["first_integer"], tmp["second_integer"])
        tmp["result_binary"] = int2binary[tmp["result_integer"]]

        # where we'll store our best guess (binary encoded)
        tmp["best_guess"] = np.zeros_like(tmp["c"])
        training_data = training_data.append(tmp,ignore_index=True)
    return training_data

def create_synapses(input_data,output_data,input_dim,hidden_dim,output_dim,num_synapses_layers):
    nn = [{"name":"input data","connection":input_data}]
    #input layer
    input_syn = {"name":"input layer"}
    input_syn["connection"] = create_connection(input_dim,hidden_dim)
    nn.append(input_syn)
    syn = {"name":"hidden0"}
    syn["connection"] = create_connection(hidden_dim,hidden_dim)
    nn.append(syn)
    #there is a layer seperating each hidden layer, since each hidden layer will need to do recurrence
    #this is probably wrong - there should be a hidden layer between each "visible" non-recurrent node, not 2 (but that's just a guess)
    for i in xrange(1,num_synapses_layers-1):
        syn = {"name":"first"+str(i)}
        syn["connection"] = create_connection(hidden_dim,hidden_dim)
        nn.append(syn)
        syn = {"name":"hidden"+str(i)}
        syn["connection"] = create_connection(hidden_dim,hidden_dim)
        nn.append(syn)
        syn = {"name":"second"+str(i)}
        syn["connection"] = create_connection(hidden_dim,hidden_dim)
        nn.append(syn)
    #output_layer
    syn = {"name":"hidden"+str(num_synapses_layers)}
    syn["connection"] = create_connection(hidden_dim,hidden_dim)
    nn.append(syn)
    syn = {"name":"output layer"}
    syn["connection"] = create_connection(hidden_dim,output_dim)
    nn.append(syn)
    nn.append({"name":"output data","connection":output_data})
    return nn

def create_deltas():
    pass

def create_update_synapses(nn,num_synapses_layers):
    update_nn = []
    #input layer
    input_syn = {"name":"input layer"}
    input_syn["connection"] = np.zeros_like(nn[1])
    update_nn.append(input_syn)
    syn = {"name":"hidden0"}
    syn["connection"] = np.zeros_like(nn[2])
    nn.append(syn)
    #there is a layer seperating each hidden layer, since each hidden layer will need to do recurrence
    for i in xrange(1,num_synapses_layers-1):
        ind_offset = 0
        syn = {"name":"first"+str(i)}
        syn["connection"] = np.zeros_like(nn[i+ind_offset])
        nn.append(syn)
        ind_offset += 1
        syn = {"name":"hidden"+str(i)}
        syn["connection"] = np.zeros_like(nn[i+ind_offset])
        nn.append(syn)
        ind_offset += 1
        syn = {"name":"second"+str(i)}
        syn["connection"] = np.zeros_like(nn[i+ind_offset])
        nn.append(syn)
    #output_layer
    syn = {"name":"hidden"+str(num_synapses_layers)}
    syn["connection"] = np.zeros_like(nn[num_synapses_layers*3])
    nn.append(syn)
    syn = {"name":"output layer"}
    syn["connection"] = np.zeros_like(nn[num_synapses_layers*3+1])
    nn.append(syn)
    return nn

def create_input_output(datum,binary_dim,position):
    X = np.array([[datum["first_binary"][binary_dim - position - 1],datum["second_binary"][binary_dim - position - 1]]])
    y = np.array([[datum["result_binary"][binary_dim - position - 1]]]).T
    return X,y

