'''
Demo for multiclass classification, using cross entropy loss
'''

from layers.linear_layer import linear_layer
from layers.softmax import softmax
from activations.relu import relu
from activations.sigmoid import sigmoid
from models.base_model import base_model
from losses.CrossEntropyLoss import CrossEntropyLoss
from matplotlib import pyplot as plt
import torch
import tqdm

def create_model(layers, device):
    model = base_model(layers)
    model.to(device)

    return model


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


def training_loop(X, y, model, loss, epochs, debug = False):

    losses = []

    if debug:
        weights_tracking = []

    for i in tqdm.tqdm(range(epochs)):

        output = model.forward(X)

        losses.append(loss.forward(output, y))

        grad_output = loss.backward()

        model.backward(grad_output)

        model.step()

        if debug:
            weights_tracking.append(torch.max(model.layers[0].weights).item())
            
    if debug:
        plot_loss(weights_tracking)
        pass

    return losses


def plot_loss(losses):
    plt.plot([i for i in range(len(losses))], losses)
    plt.show()


if __name__ == "__main__":

    DEBUG = False

    # 0. speciy device
    device = torch.device('cuda:0')

    # 1. define model
    layers = [
        linear_layer(5, 5, lr = 0.00001, clip_gradient=False),
        relu(),
        linear_layer(5, 5, lr = 0.00001, clip_gradient=False),
        relu(),
        linear_layer(5, 4, lr = 0.00001, clip_gradient=False)
    ]

    model = create_model(
        layers = layers,
        device = device
    )

    # 2. define loss

    loss = CrossEntropyLoss()

    # 3. create/load data

    X, y = create_data(1000, 5, device=device)

    # 4. training loop

    losses = training_loop(X, y, model, loss, epochs = 10000, debug = DEBUG)
    
    # 5. plot
    plot_loss(losses)


    # 6. test model

    X_test, y_test = create_data(5, 5, device)
    output_test = model.forward(X_test)

    print(y_test)
    print(softmax().forward(output_test))