import torch
import torch.nn as nn


def build_mlp(input_size, output_size, n_layers, size):
    """
    Builds a multi-layer perceptron in Pytorch based on a user's input

    Args:
        input_size (int): the dimension of inputs to be given to the network
        output_size (int): the dimension of the output
        n_layers (int): the number of layers of the network
        size (int): the size of each hidden layer
    Returns:
        An instance of (a subclass of) nn.Module representing the network.

    TODO:
        Build a feed-forward network (multi-layer perceptron, or mlp) that maps
        input_size-dimensional vectors to output_size-dimensional vectors.
        It should have 'n_layers - 1' hidden layers, each of 'size' units and followed
        by a ReLU nonlinearity. The final layer should be linear (no ReLU).

        Recall a hidden layer is a layer that occurs between the input and output
        layers of the network.

        As part of your implementation please make use of the following Pytorch
        functionalities:
        nn.Linear (https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)
        nn.Sequential (https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html)

    Hint:
        It is possible to create a list of nn.Modules and unpack these into nn.Sequential.
        For example:
            modules = []
            modules.append(nn.Linear(10, 10))
            modules.append(nn.Linear(10, 10))
            model = nn.Sequential(*modules)
    """
    ### START CODE HERE ###
    # we will keep track of all the layers we want to add 
    layers = []                     # Create an empty list to hold the layers of the MLP
    
    # the current dimension of inputs flowing in the network
    in_dim = input_size              # The first layer's input dimension is the input_size

    # Construct hidden layers (n_layers - 1 total)
    for _ in range(n_layers):    # Loop over the number of hidden layers
        layers.append(nn.Linear(in_dim, size))  # Add a fully connected layer 
        layers.append(nn.ReLU())     # Add a ReLU activation after the linear layer
        in_dim = size                # Update in_dim for the next layer (hidden layers output 'size')

    # Map hidden representation to output dimension. Add final output layer
    layers.append(nn.Linear(in_dim, output_size))  
    # This is the last linear layer that maps from the last hidden size -> output size.

    model = nn.Sequential(*layers)   # Wrap/chain the list of layers into a Sequential container
    return model                     # Return the constructed neural network
    ### END CODE HERE ###
