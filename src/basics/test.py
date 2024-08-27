from layers.linear_layer import linear_layer
from activations.relu import relu
from models.base_model import base_model
from losses.l2_loss import l2_loss
from matplotlib import pyplot as plt
import torch
import tqdm

if __name__ == "__main__":

    device = torch.device('cuda:0')

    layers = [
        linear_layer(5, 100, lr = 0.00001),
        relu(),
        linear_layer(100, 100, lr = 0.00001),
        relu(),
        linear_layer(100, 1, lr = 0.00001)
    ]

    model = base_model(layers)

    model.to(device)

    loss = l2_loss()
    losses = []

    X = torch.randn(1000,5).to(device)
    y = torch.sum(X, dim = 1, keepdim=True).to(device)

    for i in tqdm.tqdm(range(5000)):

        output = model.forward(X)

        losses.append(loss.forward(output, y).cpu())

        grad_output = loss.backward()

        model.backward(grad_output)

        model.step()

    
    plt.plot([i for i in range(len(losses))], losses)
    plt.show()


    # test model

    X_test = torch.randn(5, 5).to(device)
    y_test = torch.sum(X_test, dim = 1, keepdim=True).to(device)

    output_test = model.forward(X_test)

    print(y_test)
    print(output_test)