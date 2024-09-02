import torch
import torch.nn as nn
from .layers.linear_layer import linear_layer
from .layers.softmax import softmax
from tqdm import tqdm
from matplotlib import pyplot as plt


def create_data(n, input_dim, device):
    X = torch.randn(n, input_dim).to(device)
    y = torch.sum(X, dim = 1, keepdim=True).to(device)

    return X, y 


def plot_loss(losses):
    plt.plot([i for i in range(len(losses))], losses)
    plt.show()




class model(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            linear_layer(5, 5),
            softmax(),
            linear_layer(5, 5),
            softmax(),
            linear_layer(5, 1)
        )

    def forward(self, X):
        return self.layer(X)
    


if __name__ == "__main__":

    torch.manual_seed(42)

    device = torch.device('cuda:0')

    model = model()
    model.to(device)

    X, y = create_data(n = 50, input_dim = 5, device = device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters())

    losses = []

    for _ in tqdm(range(5000)):
        output = model(X)

        optimizer.zero_grad()

        loss = loss_fn(output, y)

        loss.backward()

        optimizer.step()

        losses.append(loss.item())
    

    plot_loss(losses)

    X_test, y_test = create_data(n = 5, input_dim = 5, device = device)

    print(y_test)
    print(model(X_test))

