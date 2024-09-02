import torch
import torch.nn as nn

class relu(nn.Module):

    def __init__(self, debug = True):
        super().__init__()
        self.debug = debug

    def forward(self, X):
        self.X = X
        return torch.maximum(X, torch.tensor(0))

    def backward(self, grad_output):
        '''
        let z be the input to relu
        and y be the output of relu (i.e. y = relu(z) =: f(z))

        then we need dL/dz = dL/dy * dy/dz, where dL/dy is grad_output and dy/dz = f'(z)

        dy/dz is 0 whereever z <= 0 and 1 otherwise

        hence dL/dz = dL/dy * dy/dz = 0 whereever z <= 0 and grad_output otherwise
        '''
        grad_input = grad_output.clone()

        grad_input[self.X <= 0] = 0

        return grad_input
    
