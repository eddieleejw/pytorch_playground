# TODO
import torch
from module.module import module

class softmax(module):
    '''
    applies softmax to incoming data

    since data is given as rows of features, the softmax is done along the row
    '''

    def __init__(self):
        super().__init__()

    def foward(self, X):
        '''
        X has shape n x input_dim

        output should also have shape n x input_dim
        '''
        exped = torch.exp(X)
        out = exped / torch.sum( exped, dim = 1, keepdim=True) # for broadcasting reasons, we want to keep the sum having shape (n,1), so it can broadcase with exped which has shape (n, input_dim)
        self.out

        return out

    def backward(self, grad_output):
        # TODO
        '''
        the final output dL/dX is going to have shape n x input_dim, because that's the shape that X has

        but the calculation to get there is a little difficult

        this is because we need to compute the Jacobian matrix for EACH ROW in self.out

        '''
        pass
