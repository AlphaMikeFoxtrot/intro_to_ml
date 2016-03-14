import numpy as np
import copy
from tools import *

def nonlin(x,deriv=False):
    if(deriv==True):
	return x*(1-x)
    return 1/(1+np.exp(-x))

def create_connection(num_rows,num_cols):
    return 2*np.random.random((num_rows,num_cols)) -1

def create_nn(input_data,output_data,depth_hidden_layers,breathe_hidden_layers):
    nn = [{"name":"input data","connection":input_data}]
    #input layer
    input_syn = {"name":"input layer"}
    input_syn["connection"] = create_connection(len(input_data[0]),breathe_hidden_layers)
    nn.append(input_syn)
    #hidden layers
    for i in xrange(depth_hidden_layers):
        syn = {"name":i}
        syn["connection"] = create_connection(breathe_hidden_layers,breathe_hidden_layers)
        nn.append(syn)
    #output_layer
    syn = {"name":"output layer"}
    syn["connection"] = create_connection(breathe_hidden_layers,len(output_data[0]))
    nn.append(syn)
    nn.append({"name":"output data","connection":output_data})
    return nn

def forward_propagate(synapses):
    layers = [synapses[0]["connection"]]
    for ind,synapse in enumerate(synapses[:-1]):
        if ind == 0: continue
        layers.append(
            nonlin(np.dot(layers[ind-1],synapse["connection"]))
        )
    return layers

def back_propagate(layers,synapses,alpha=0.1):
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
        synapses[synapses_index]["connection"] += alpha * layers[layers_index].T.dot(deltas[deltas_index])
        synapses_index -= 1
        layers_index -= 1
        deltas_index += 1
    return synapses,errors[0]
                        
def tune(depths,breathes,alphas=[0.1]):
    np.random.seed(1)
    X = np.array([[0,0,1],
                  [0,1,1],
                  [1,0,1],
                  [1,1,1]])
                
    y = np.array([[0],
	          [1],
	          [1],
                  [0]])
    depth_breathes = [] 
    for depth in xrange(depths[0],depths[1]):
        for breathe in xrange(breathes[0],breathes[1]):
            #print "With",depth,"hidden layers"
            #print "With",breathe,"nodes per layer"
            for alpha in alphas:
                tmp = {"depth":depth,"breathe":breathe,"alpha":alpha}
                #print "With alpha=",alpha
                tmp["errors"] = train_network(X,y,depth,breathe,alpha=alpha)
                tmp["min_error"] = min(tmp["errors"])
                tmp["ave_error"] = sum(tmp["errors"])/float(len(tmp["errors"]))
                depth_breathes.append(tmp)
            if depth ==0: break #this way we don't iterate through everything when breathe doesn't change
    return depth_breathes

def train_network(X,y,depth,breathe,alpha=0.1,num_iterations=70000):
    errors = []
    nn = create_nn(X,y,depth,breathe)
    for j in xrange(num_iterations):
        layers = forward_propagate(nn)
        nn,error = back_propagate(layers,nn,alpha=alpha)
        if j %1000 == 0:   
            errors.append(np.mean(np.abs(error)))
    return errors

def run_once(num_hidden_nodes):
    np.random.seed(1)
    X = np.array([[1,1],
                  [1,0],
                  [0,1],
                  [0,0]])
                
    y = np.array([[0],
	          [1],
	          [1],
                  [1]])
    errors = []
    nn = create_nn(X,y,num_hidden_nodes)
    for j in xrange(70000):
        layers = forward_propagate(nn)
        nn,error = back_propagate(layers,nn)
        #if j%100 == 0:
        errors.append(np.mean(np.abs(error)))
    return errors



if __name__ == '__main__':
    depth_breathes = tune([0,5],[4,7],alphas=[0.001,0.1,1,5,10,15,20,25,30,35,40])
    depth_breathes = sort_by_key(depth_breathes,"min_error")
    for elem in depth_breathes[:15]:
        print "The network had the following attributes"
        print "Depth",elem["depth"]
        print "Breathe",elem["breathe"]
        print "Alpha",elem["alpha"]
        print "The minimum error for this network was",elem["min_error"]
        print "The average error for this network was",elem["ave_error"]
        
    # for i in xrange(0,7):
    #     errors = run_once(i)
    #     print "The minimum error for the this network was",min(errors)
    #     print "The average error for the this network was",sum(errors)/float(len(errors))
    #     inflection_points,num_inflection_points = find_inflection_points(errors)
    #     print "These were the inflection points for ",i
    #     print "There were",num_inflection_points,"in total"
