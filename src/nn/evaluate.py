import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from model import numpy_nn
from utilities import metrics, DataTransformer


class ClassificationReport:
    def __init__(self, predictions, labels, num_classes):
        self.predictions = predictions
        self.labels = labels
        self.num_classes = num_classes
        self.full_report = self.generate_report()
        self.small_report = self.full_report.mean()

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

X, Y = DataTransformer.split_prep_data(datatrain)


output, memory = model.forward(X)


predictions = np.argmax(output, 0)
rep = ClassificationReport(predictions, Y, 10)

print(rep.small_report)
# f1 = metrics.f1_score(predictions, Y, 10)
# print(np.mean(f1))
