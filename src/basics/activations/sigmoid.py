from module.module import module
import torch

class sigmoid(module):

    def __init__(self):
        super().__init__()
    
    def forward(self, X):
        y = 1/(1 + torch.exp(-X))
        self.y = y

        assert y.shape == X.shape

        return y

    def backward(self, grad_output):
        '''
        given the sigmoid function f(x) = 1/(1 + e^-x)

        df/dx = f(x) (1 - f(x))

        hence dL/dx = dL/df * df/dx = grad_output * f(x) (1 - f(x))

        since the sigmoid function is an elemetwise operation, the gradient is also done element wise

        also we let y := f(x)
        '''
        y = self.y
        dx = y * (1 - y) # df/dx i.e. dy/dx
        grad_input = grad_output * dx
        return grad_input


        
        

