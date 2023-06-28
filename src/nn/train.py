import numpy as np
import pandas as pd


from model import numpy_nn
from optimizer import SimpleGradientDescent
from utilities import DataTransformer

model = numpy_nn(784, 1, 10)
model.load_model_state("Image_classification_numPyNN/src/models/model0.json")
datatrain = pd.read_csv("Image_classification_numPyNN/data/mnist_train.csv")

X, Y = DataTransformer.split_prep_data(datatrain)
optimizer = SimpleGradientDescent(model, 0.01)


checkpoint = 10000 // 5

for i in range(10):
    output, memory = model.forward(X)
    gradients = model.backward(Y, memory)
    optimizer.step(gradients)
    if i % checkpoint == 0:
        print("Iteration: ", i)
        predictions = np.argmax(output, 0)
        print(np.sum(predictions == Y) / Y.size)
model.save_model_state()
