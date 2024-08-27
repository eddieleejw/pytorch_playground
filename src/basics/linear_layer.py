import numpy as np
import torch
import torch.nn as nn

class linear_layer():

    def __init__(self, input_dim, output_dim, lr = 0.001):
        self.weights = torch.randn(input_dim, output_dim)
        self.bias = torch.randn(1, output_dim)
        self.weights_grad = None
        self.bias_grad = None
        self.lr = 0.001

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
        self.weights_grad = grad_weights

        #dL/db
        grad_bias = torch.sum(grad_output, dim = 0, keepdim=True)
        self.bias_grad = grad_bias

        return grad_input

    def step(self):
        self.weights -= self.lr * self.weights_grad
        self.bias -= self.lr * self.bias_grad



if __name__ == "__main__":

    layers = []
    ll1 = linear_layer(5, 8)
    ll2 = linear_layer(8, 10)
    ll3 = linear_layer(10, 3)
    layers.append(ll1)
    layers.append(ll2)
    layers.append(ll3)

    # input
    X = torch.randn(3, 5)

    # stores losses, which is just sum(output)
    losses = []


    for _ in range(100):
        # forward pass
        z = X
        for layer in layers:
            z = layer.forward(z)
        
        losses.append(sum(sum(z)))

        # backward pass
        grad_output = torch.ones_like(z)
        for layer in layers[::-1]:
            grad_output = layer.backward(grad_output)
        
        # step optimizer
        for layer in layers:
            layer.step()


    print(losses)