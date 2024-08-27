'''
implement the base "module" class which has methods such as

forward
backward
to
step
'''


class module():

    def __init__(self):
        self.debug = False
        self.parameters = [] # a list of tensors

    def forward(self):
        raise NotImplementedError
    
    def backward(self):
        pass

    def step(self):
        pass

    def to(self, device):
        pass

    def log(self, msg):
        if self.debug:
            print(msg)
    