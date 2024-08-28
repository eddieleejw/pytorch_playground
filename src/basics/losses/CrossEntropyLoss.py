import torch
from module.module import module
from layers.softmax import softmax

class CrossEntropyLoss(module):
    '''
    Cross entropy loss f(yhat, y) = \sum_{i in [C]} - y_i log(yhat_i)
        where C is the number of output classes
        y is the target, and y_i = 1 if the target label is i, and 0 otherwise
        yhat is the model prediction, and yhat_i is the SCORE of label i (not normalised to be a probability)
        y_i and yhat_i are the i-th entries

    Note: CrossEntropyLoss should first do a softmax on the logits to normalise score to probabilities
    '''


    def __init__(self):
        super.__init__()
        self.softmax = softmax()

    def forward(self, scores, y):
        '''
        scores is the unnormalised scores, and has shape n x C, for each row, scores_i is the score of class C for that datapoint

        yhat and y have shape n x C
            where n is the number of samples and C is the number of classes
        
        output should have shape n x 1
        '''

        # normalise scores to probabilities
        yhat = self.softmax.forward(scores)

        self.scores = scores
        self.yhat = yhat
        self.y = y

        # output has shape n x C
        output = - y * torch.log(yhat)

        # sum each row together so that there is only one column left
        output = torch.sum(output, dim = 1)
        
        return output

    def backward(self, grad_output):
        '''
        we want to compute dL/dyhat
            which should be a matrix of shape n x C
            and each row is [dL/dyhat_1   dL/dyhat_2   ...   dL/dyhat_C]

        and dL/dyhat_i = - y_i / yhat_i
        '''

        grad_input = - self.y / self.yhat

        grad_input = self.softmax.backward(grad_input)
        
        
        return grad_input

