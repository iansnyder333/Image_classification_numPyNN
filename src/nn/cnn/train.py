import torch
from torch.nn import CrossEntropyLoss
from torch.autograd import Variable
from torch.optim import Adam
from constants import *
from model import CNN
from prepdata import *
from tqdm import tqdm
from matplotlib import pylab as plt
import numpy as np

model = CNN()

print(sum(p.numel() for p in model.parameters()) / 1e6, "M parameters")
criterion = CrossEntropyLoss()  # Softmax is internally computed.
optimizer = Adam(params=model.parameters(), lr=LR)

print("Training the Deep Learning network ...")
train_cost = []
train_accu = []


total_batch = len(mnist_train) // BATCH_SIZE
# total_batch = len(fasion_mnist_train) // BATCH_SIZE

print("Size of the training dataset is {}".format(mnist_train.data.size()))
print("Size of the testing dataset".format(mnist_test.data.size()))
print("Batch size is : {}".format(BATCH_SIZE))
print("Total number of batches is : {0:2.0f}".format(total_batch))
print("\nTotal number of epochs is : {0:2.0f}".format(EPOCHS))


for epoch in range(EPOCHS):
    avg_cost = 0
    for i, (batch_X, batch_Y) in tqdm(enumerate(data_loader)):
        X = Variable(batch_X)  # image is already size of (28x28), no reshape
        Y = Variable(batch_Y)  # label is not one-hot encoded

        optimizer.zero_grad()  # <= initialization of the gradients

        # forward propagation
        hypothesis = model(X)
        cost = criterion(hypothesis, Y)  # <= compute the loss function

        # Backward propagation
        cost.backward()  # <= compute the gradient of the loss/cost function
        optimizer.step()  # <= Update the gradients
        # Print some performance to monitor the training
        prediction = hypothesis.data.max(dim=1)[1]
        train_accu.append(((prediction.data == Y.data).float().mean()).item())
        train_cost.append(cost.item())
        if i % 200 == 0:
            print(
                "Epoch= {},\t batch = {},\t cost = {:2.4f},\t accuracy = {}".format(
                    epoch + 1, i, train_cost[-1], train_accu[-1]
                )
            )

        avg_cost += cost.data / total_batch

    print("[Epoch: {:>4}], averaged cost = {:>.9}".format(epoch + 1, avg_cost.item()))

print("Learning Finished!")

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_cost": train_cost,
        "train_accu": train_accu,
    },
    "Image_classification_numPyNN/src/models/fashion_cnn_model.pth",
)


def evaluate():
    # Test model and check accuracy
    model.load_state_dict(
        torch.load("Image_classification_numPyNN/src/models/fashion_cnn_model.pth")[
            "model_state_dict"
        ]
    )
    model.eval()  # set the model to evaluation mode (dropout=False)

    X_test = Variable(
        fasion_mnist_test.data.view(len(fasion_mnist_test), 1, 28, 28).float()
    )
    Y_test = Variable(fasion_mnist_test.targets)

    prediction = model(X_test)

    # Compute accuracy
    correct_prediction = torch.max(prediction.data, dim=1)[1] == Y_test.data
    accuracy = correct_prediction.float().mean().item()
    print("\nAccuracy: {:2.2f} %".format(accuracy * 100))


def plot_train():
    plt.figure(figsize=(20, 10))
    plt.subplot(121), plt.plot(np.arange(len(train_cost)), train_cost), plt.ylim(
        [0, 10]
    )
    plt.subplot(122), plt.plot(
        np.arange(len(train_accu)), 100 * torch.as_tensor(train_accu).numpy()
    ), plt.ylim([0, 100])


def show_pred():
    plt.figure(figsize=(15, 15), facecolor="white")
    for i in torch.arange(0, 12):
        val, idx = torch.max(prediction, dim=1)
        plt.subplot(4, 4, int(i) + 1)
        plt.imshow(X_test[i][0])
        plt.title("This image contains: {0:>2} ".format(idx[i].item()))
        plt.xticks([]), plt.yticks([])
        plt.plt.subplots_adjust()
