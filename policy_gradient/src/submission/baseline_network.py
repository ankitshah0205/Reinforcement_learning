import numpy as np
import torch
import torch.nn as nn
from utils.network_utils import np2torch
from submission.mlp import build_mlp


class BaselineNetwork(nn.Module):
    """
    Class for implementing Baseline network

    Args:
        env (): OpenAI gym environment
        config (dict): A dictionary containing generated from reading a yaml configuration file

    TODO:
        Create self.network using build_mlp, with observations space dimensional input 
        and 1-dimensional output. Create an Adam optimizer and assign it to
        self.optimizer which will be used later to optimize the network parameters.
        You should make use of some values from config, such as the number of layers,
        the size of the layers, and the learning rate.
    """

    def __init__(self, env, config):
        super().__init__()
        self.config = config
        self.env = env
        self.lr = self.config["hyper_params"]["learning_rate"]
        self.device = torch.device("cpu")
        if self.config["model_training"]["device"] == "gpu":
            if torch.cuda.is_available(): 
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
                self.device = torch.device("mps")
        print(f"Running Baseline model on device {self.device}")
        ### START CODE HERE ###
        
        # First, figure out the dimensionality of a single observation
        var_obs_vector_dim = self.env.observation_space.shape[0]

        # Now, pull model hyperparameters - let us retrieve Network depth and width from config file
        number_of_hidden_layers = self.config["hyper_params"]["n_layers"]
        size_of_hidden_layer = self.config["hyper_params"]["layer_size"]

        # Now, build our MLP baseline model. Build an MLP that outputs a single scalar (value function estimate)
        # Using our helper build_mlp from submission/mlp.py
        # input : obsevation vector
        # hidden layers: user defined count and size
        # output : our baseline prediction
        self.network = build_mlp(
            input_size=var_obs_vector_dim,
            output_size=1,                      # only one output → baseline prediction
            n_layers=number_of_hidden_layers,
            size=size_of_hidden_layer
        )#.to(self.device)

        # Now, let's define Optimizer for updating baseline network parameters
        # This will update the parameters of baseline MLPs from gradients
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), 
            lr=self.lr
        )

        ### END CODE HERE ###

    def forward(self, observations):
        """
        Pytorch forward method used to perform a forward pass of inputs(observations)
        through the network

        Args:
            observations (torch.Tensor): observation of state from the environment
                                        (shape [batch size, dim(observation space)])

        Returns:
            output (torch.Tensor): networks predicted baseline value for a given observation
                                (shape [batch size])

        TODO:
            Run a forward pass through the network and then squeeze the outputs so that it's
            1-dimensional. Put the squeezed outputs in a variable called "output"
            (which will be returned).
        """
        ### START CODE HERE ###
        

        # Let's push the observations in our baseline model
        predicted_values = self.network(observations)

        # Remove singleton extra dimension so the result is shape [batch_size]
        output = predicted_values.squeeze(-1)

        ### END CODE HERE ###
        assert output.ndim == 1
        return output

    def calculate_advantage(self, returns, observations):
        """


        Args:
            returns (np.array): the history of discounted future returns for each step (shape [batch size])
            observations (np.array): observations at each step (shape [batch size, dim(observation space)])

        Returns:
            advantages (np.array): returns - baseline values  (shape [batch size])

        TODO:
            Evaluate the baseline and use the result to compute the advantages.
            Put the advantages in a variable called "advantages" (which will be
            returned).

        Note:
            The arguments and return value are numpy arrays. The np2torch function
            converts numpy arrays to torch tensors. You will have to convert the
            network output back to numpy, which can be done via the numpy() method.
            See Converting torch Tensor to numpy Array section of the following tutorial
            for further details: https://pytorch.org/tutorials/beginner/former_torchies/tensor_tutorial.html
            Before converting to numpy, take into consideration the current device of the tensor and whether
            this can be directly converted to a numpy array. Further details can be found here:
            https://pytorch.org/docs/stable/generated/torch.Tensor.cpu.html
        """
        observations = np2torch(observations, device=self.device)
        ### START CODE HERE ###
        
        # Use the network to predict baseline values (we are only evluating - no gradient needed for advantage calc)
        # we are not training
        with torch.no_grad():
            # Retrieve baseline predictions based on states
            # convert back to numpy
            baseline_estimates = self.forward(observations).cpu().numpy()            
        
        # Compute advantages as difference between returns and baseline predictions
        # Advantage = actual return - baseline prediction
        # Positive advantage = state-action did better than expected
        # Negative advantage = state-action did worse than expected
        advantages = returns - baseline_estimates
        ### END CODE HERE ###
        return advantages

    def update_baseline(self, returns, observations):
        """
        Performs back propagation to update the weights of the baseline network according to MSE loss

        Args:
            returns (np.array): the history of discounted future returns for each step (shape [batch size])
            observations (np.array): observations at each step (shape [batch size, dim(observation space)])

        TODO:
            Compute the loss (MSE), backpropagate, and step self.optimizer.
            If you want to use mini-batch SGD, we have provided a helper function
            called batch_iterator (implemented in utils/network_utils.py).
        """
        returns = np2torch(returns, device=self.device)
        observations = np2torch(observations, device=self.device)
        ### START CODE HERE ###

        # Reset old/optimizer gradients so they do not accumulation
        self.optimizer.zero_grad()

        # Forward pass- compute baseline predictions for given observations
        predicted_returns = self.forward(observations)

        # Define MSE[mean squared error] loss between predicted values and actual returns
        # This way are ensuring predictions to be close to actual discounted returns
        mse_criterion = nn.MSELoss()
        loss_amount = mse_criterion(predicted_returns, returns)

        # Compute gradients of loss for network parameters - Perform backpropagation
        loss_amount.backward()

        # Update model parameters using the optimizer in the direction that reduces loss
        self.optimizer.step()


        ### END CODE HERE ###
