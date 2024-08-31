import torch
import torch.nn as nn

class linear_layer(nn.Module):

    def __init__(self, input_dim, output_dim, debug = True, clip_gradient = False):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(input_dim, output_dim) * (1. / input_dim**0.5))
        self.bias = nn.Parameter(torch.zeros(1, output_dim))

        self.debug = debug
        self.clip_gradient = clip_gradient

    def forward(self, X):
        '''
        X: has dimensions num_samples x input_dim
        self.weights has dimension input_dim x output_dim
        self.bias has dimension output_dim x 1
        '''
        self.X = X
        return torch.matmul(X, self.weights) + self.bias

    def backward(self, grad_output):
        '''
        let z:= XW + b

        then the input is grad_ouput = dL/dz (which has dimension of z = num_samples x output_dim)
        Hence:
            dL/dX = dL/dz * dz/dX = grad_output * W^T (since W has dimension input_dim x output_dim, and dL/dX should have dimension num_samples x input_dim)
            dL/dW = dL/dz * dz/dW = x^T * grad_output (since X has dimension num_samples x input_dim, and dL/dW should have dimensions input_dim x output_dim)
            dL/db = dL/dz * dz/db = vec{1} * grad_output, where vec{1} has dimension 1 x num_samples
        '''

        # dL/dX
        grad_input = torch.matmul(grad_output, self.weights.T)

        #dL/dW
        grad_weights = torch.matmul(self.X.T, grad_output)
        self.weights.grad = torch.nan_to_num(grad_weights)

        #dL/db
        grad_bias = torch.sum(grad_output, dim = 0, keepdim=True)
        self.bias.grad = torch.nan_to_num(grad_bias)

        if self.clip_gradient:
            torch.clamp_(self.weights.grad, -1, 1)
            torch.clamp_(self.bias.grad, -1, 1)

        return grad_input

