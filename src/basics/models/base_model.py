'''
Base model that can be initialised with a list of layers/activations

Calling forward on the base model will do a forward pass through all layers

Calling backward will do a backward pass through all layers

Calling step will step each layer
'''


import torch
from module.module import module

class base_model(module):

    def __init__(self, layers, debug = True):
        super().__init__()
        self.layers = layers
        self.debug = debug

    def forward(self, X):
        z = X
        for layer in self.layers:
            z = layer.forward(z)
        return z


    def backward(self, grad_output):
        for layer in self.layers[::-1]:
            grad_output = layer.backward(grad_output)


    def step(self):
        for layer in self.layers:
            layer.step()
    
    def to(self, device):
        for layer in self.layers:
            layer.to(device)