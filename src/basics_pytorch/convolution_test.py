import torch
import torch.nn as nn
from layers.convolution2D import convolution2d
from activations.relu import relu
from layers.linear_layer import linear_layer
from matplotlib import pyplot as plt
from tqdm import tqdm


class model(nn.Module):

    def __init__(self):
        super().__init__()
        self.convolution_layers = nn.Sequential(
            convolution2d(3, 5, (3,3)),
            relu(),
            convolution2d(5, 5, (3,3)),
            relu(),
            convolution2d(5, 1, (3,3)),
            relu()
        )
        self.linear_layer = linear_layer(14*14, 1)

    def forward(self, X):
        out = self.convolution_layers(X)
        return self.linear_layer(out.view(X.shape[0], -1))



def create_data(n, in_channels, input_dim, device):
    X = torch.randn(n, in_channels, input_dim, input_dim).to(device)
    y = torch.sum(X, dim = (1,2,3), keepdim=False).to(device)
    y = y.unsqueeze(1)
    y = y / (input_dim**2)

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
    X_train, y_train = create_data(500, 3, 20, device)
    X_test, y_test = create_data(5, 3, 20, device)

    # 3. loss and optimizer
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

    # 3. train
    EPOCHS = 2000
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