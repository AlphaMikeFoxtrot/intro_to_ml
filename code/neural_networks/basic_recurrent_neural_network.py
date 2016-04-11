import numpy as np
import copy

np.random.seed(0)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_prime(x):
    return x*(1-x)

synapse_0 = 2*np.random.random((2,16)) - 1
synapse_1 = 2*np.random.random((16,1)) - 1
synapse_h = 2*np.random.random((16,16)) - 1

synapse_0_update = np.zeroes_like(synapse_0)
synapse_1_update = np.zeroes_like(synapse_1)
synapse_h_update = np.zeroes_like(synapse_h)

X_1 = np.array([1,1])
y_1 = np.array([1])
X_2 = np.array([1,0])
y_2 = np.array([0])
X_3 = np.array([0,1])
y_3 = np.array([0])
X_4 = np.array([0,0])
y_4 = np.array([0])

Xs = [X_1,X_2,X_3,X_4]
Ys = [y_1,y_2,y_3,y_4]

layer_2_deltas = []
layer_1_values = []
layer_1_values.append(np.zeros(16))

overall_error = 0 

for ind,X in enumerate(Xs):
    layer_1 = sigmoid(np.dot(X,synapse_0) + np.dot(layer_1_values[-1],synapse_h))
    layer_2 = sigmoid(np.dot(layer_1,synapse_1))

    layer_2_error = Ys[ind] - layer_2
    layer_2_deltas.append(layer_2_error * sigmoid_prime(layer_2))
    overall_error += np.abs(layer_2_error[0])

    layer_1_values.append(copy.deepcopy(layer_1))

future_layer_1_delta = np.zeros(16)

for ind,X in enumerate(Xs):
    layer_1 = layer_1_values[-ind-1]
    prev_layer_1 = layer_1_values[-ind-2]

    layer_2_delta = layer_2_deltas[-position-1]

    layer_1_delta = (future_layer_1_delta.dot(synapse_h.T) + layer_2_delta.dot(synapse_1.T)) * sigmoid_prime(layer_1)

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

