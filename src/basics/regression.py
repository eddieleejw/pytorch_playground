from layers.linear_layer import linear_layer
from activations.relu import relu
from models.base_model import base_model
from losses.l2_loss import l2_loss
from matplotlib import pyplot as plt
import torch
import tqdm

def create_model(layers, device):
    model = base_model(layers)
    model.to(device)

    return model


def create_data(n, input_dim, device):
    X = torch.randn(n, input_dim).to(device)
    y = torch.sum(X, dim = 1, keepdim=True).to(device)

    return X, y 


def training_loop(X, y, model, loss, epochs):

    losses = []

    for i in tqdm.tqdm(range(5000)):

        output = model.forward(X)

        losses.append(loss.forward(output, y).cpu())

        grad_output = loss.backward()

        model.backward(grad_output)

        model.step()
    
    return losses


def plot_loss(losses):
    plt.plot([i for i in range(len(losses))], losses)
    plt.show()


if __name__ == "__main__":

    # 0. speciy device
    device = torch.device('cuda:0')

    # 1. define model
    layers = [
        linear_layer(5, 5, lr = 0.00001),
        relu(),
        linear_layer(5, 5, lr = 0.00001),
        relu(),
        linear_layer(5, 1, lr = 0.00001)
    ]

    model = create_model(
        layers = layers,
        device = device
    )

    # 2. define loss

    loss = l2_loss()

    # 3. create/load data

    X, y = create_data(1000, 5, device=device)

    # 4. training loop

    losses = training_loop(X, y, model, loss, epochs = 1000)
    
    # 5. plot
    plot_loss(losses)


    # 6. test model

    X_test, y_test = create_data(5, 5, device)
    output_test = model.forward(X_test)

    print(y_test)
    print(output_test)