import numpy as np
import pandas as pd
import json
from json import JSONEncoder
import io

from model import numpy_nn
from optimizer import SimpleGradientDescent


model = numpy_nn(784, 1, 10)
model.load_model_state("Image_classification_numPyNN/src/models/model0.json")
datatrain = pd.read_csv("Image_classification_numPyNN/data/mnist_train.csv")

data = np.array(datatrain)
m, n = data.shape
optimizer = SimpleGradientDescent(model, 0.01)
# shuffle before splitting into dev and training sets
print(m, n)

np.random.shuffle(data)
data_train = data.T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.0
_, m_train = X_train.shape
X = X_train
Y = Y_train
checkpoint = 10000 // 5

for i in range(10000):
    output, memory = model.forward(X)
    gradients = model.backward(Y, memory)
    optimizer.step(gradients)
    if i % checkpoint == 0:
        print("Iteration: ", i)
        predictions = np.argmax(output, 0)
        print(np.sum(predictions == Y) / Y.size)
model.save_model_state()
