import numpy as np
import pandas as pd
from tqdm import tqdm

from model import numpy_FNN
from optimizer import SimpleGradientDescent
from utilities import DataTransformer

model = numpy_FNN(784, 10, 10)

model.load_model_state("Image_classification_numPyNN/src/models/fnn10.json")

optimizer = SimpleGradientDescent(model, 0.1)


def train_fasion(iterations: int):
    X, Y = DataTransformer.load_mnist("Image_classification_numPyNN/data/fasion")
    X = X.T
    Y = Y.T
    checkpoint = iterations // 2

    for i in tqdm(range(iterations)):
        output, memory = model.forward(X)
        gradients = model.backward(Y, memory)
        optimizer.step(gradients)
        if i % checkpoint == 0:
            predictions = np.argmax(output, 0)
            print(np.sum(predictions == Y) / Y.size)
    model.save_model_state(name="fnn10_fasion")


def train_mnist(iterations: int):
    datatrain = pd.read_csv("Image_classification_numPyNN/data/mnist_train.csv")
    X, Y = DataTransformer.split_prep_data(datatrain)
    checkpoint = iterations // 2

    for i in tqdm(range(iterations)):
        output, memory = model.forward(X)
        gradients = model.backward(Y, memory)
        optimizer.step(gradients)
        if i % checkpoint == 0:
            predictions = np.argmax(output, 0)
            print(np.sum(predictions == Y) / Y.size)
    model.save_model_state(name="fnn10")


train_mnist(100)
