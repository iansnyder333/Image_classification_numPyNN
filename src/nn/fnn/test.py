import numpy as np
import pandas as pd


from model import numpy_FNN

from utilities import DataTransformer

model = numpy_FNN(784, 492, 10)
model.load_model_state("model1.json")
datatrain = pd.read_csv("Image_classification_numPyNN/data/mnist_test.csv")

X, Y = DataTransformer.split_prep_data(datatrain)


output, memory = model.forward(X)


predictions = np.argmax(output, 0)
print(np.sum(predictions == Y) / Y.size)
