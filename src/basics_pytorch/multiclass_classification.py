'''
Demo for multiclass classification, using cross entropy loss
'''

from .layers.linear_layer import linear_layer
from .layers.softmax import softmax
from .activations.relu import relu
from .activations.sigmoid import sigmoid
from .losses.CrossEntropyLoss import CrossEntropyLoss
from matplotlib import pyplot as plt
import torch
from tqdm import tqdm
import torch.nn as nn

class model(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            linear_layer(5, 5),
            relu(),
            linear_layer(5, 5),
            relu(),
            linear_layer(5, 4)
        )

    def forward(self, X):
        return self.layers(X)




def create_data(n, input_dim, device):
    X = torch.randn(n, input_dim).to(device)
    # y = torch.sum(X, dim = 1, keepdim=True).to(device)
    X_sum = torch.sum(X, dim = 1, keepdim=False)
    y = torch.zeros(n, 4) # 4 is the number of classes
    
    for i, row in enumerate(X_sum):
        if row < 0:
            y[i][0] = 1
        elif row < 2:
            y[i][1] = 1
        elif row < 4:
            y[i][2] = 1
        else:
            y[i][3] = 1

    y = y.to(device)

    return X, y 


def plot_loss(losses):
    plt.plot([i for i in range(len(losses))], losses)
    plt.show()


if __name__ == "__main__":

    # 0. set up device
    device = torch.device("cuda:0")

    # 1. create model
    model = model()
    model.to(device)

    # 2. create data
    X_train, y_train = create_data(500, 5, device)
    X_test, y_test = create_data(5, 5, device)

    # 3. loss and optimizer
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.005, momentum=.9)

    # 3. train
    EPOCHS = 10000
    losses = []
    for _ in tqdm(range(EPOCHS)):
        optimizer.zero_grad()

        output = model(X_train)

        loss = loss_fn(output, y_train)

        loss.backward()

        optimizer.step()

        losses.append(loss.item())


    plot_loss(losses)

    print(y_test)
    print(model(X_test))