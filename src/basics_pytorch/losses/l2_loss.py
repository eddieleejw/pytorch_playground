'''
L2 loss
'''
import torch
import torch.nn as nn


class l2_loss(nn.Module):

    def __init__(self, debug = True):
        super().__init__()
        self.debug = debug
    
    def forward(self, pred, y):
        assert pred.shape == y.shape, "pred and y shape is different in l2 loss"
        assert y.shape[1] == 1
        assert y.dim() == 2

        self.pred = pred
        self.y = y

        z = pred - y
        z = torch.square(z)
        z = torch.sum(z)
        z = z / pred.shape[0]

        return z.item()
    
    def backward(self):
        '''
        y_hat be the predicted value, and y be the true value

        the loss L(y_hat, y) = \sum_i (y_hat_i - y_i)^2, where y_i is the i-th entry of y (and similarly for y_hat)

        then backward compute dL/dy_hat which is a vector with same dimension as y_hat

        The i-th entry of dL/dy_hat is given by \frac{\partial L}{\partial y_hat_i} = 2(y_hat_i - y_i)
        '''

        grad_input = 2*(self.pred - self.y)

        return grad_input
