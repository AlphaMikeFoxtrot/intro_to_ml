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
