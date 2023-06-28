import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from model import numpy_nn
from utilities import metrics


def get_model_prediction(A2):
    return np.argmax(A2, 0)


def make_model_prediction(X):
    output, _ = model.forward(X)
    prediction = get_model_prediction(output)
    return prediction


def get_model_accuracy(predictions, labels):
    return np.sum(predictions == labels) / labels.size


def display_image(current_image):
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation="nearest")
    plt.show()


def test_model_prediction(idx, X, Y, display=False):
    current_image = X[:, idx, None]
    prediction = make_model_prediction(current_image)
    label = Y[idx]
    print(f"Predicted Value: {prediction} \n")
    print(f"Correct Value: {label} \n")
    if display:
        display_image(current_image)
    return


class ClassificationReport:
    def __init__(self, predictions, labels, num_classes):
        self.predictions = predictions
        self.labels = labels
        self.num_classes = num_classes

    def generate_report(self):
        data = {
            "label": [i for i in range(self.num_classes)],
            "accuracy": metrics.accuracy(
                self.predictions, self.labels, self.num_classes
            ),
            "precision": metrics.precision(
                self.predictions, self.labels, self.num_classes
            ),
            "recall": metrics.recall(self.predictions, self.labels, self.num_classes),
            "f1": metrics.f1_score(self.predictions, self.labels, self.num_classes),
        }
        report = pd.DataFrame.from_dict(data)
        return report


model = numpy_nn(784, 1, 10)
model.load_model_state("Image_classification_numPyNN/src/models/model0.json")
datatrain = pd.read_csv("Image_classification_numPyNN/data/mnist_test.csv")

data = np.array(datatrain)
m, n = data.shape

print(m, n)
data_train = data.T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.0
_, m_train = X_train.shape
X = X_train
Y = Y_train


output, memory = model.forward(X)


predictions = np.argmax(output, 0)
rep = ClassificationReport(predictions, Y, 10)
repor = rep.generate_report()
print(repor)
# f1 = metrics.f1_score(predictions, Y, 10)
# print(np.mean(f1))
