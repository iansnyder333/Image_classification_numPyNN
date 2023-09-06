import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import json
from json import JSONEncoder
import os
import gzip


class metrics:
    def accuracy_for_class(predictions, labels, class_id):
        return np.sum((predictions == class_id) & (labels == class_id)) / np.sum(
            labels == class_id
        )

    def accuracy(predictions, labels, num_classes):
        acc = [
            metrics.accuracy_for_class(predictions, labels, class_id)
            for class_id in range(num_classes)
        ]
        return acc

    def rmse(predictions, labels):
        return np.sqrt(((predictions - labels) ** 2).mean())

    def true_positives(predictions, labels, class_id):
        return np.sum((predictions == class_id) & (labels == class_id))

    def false_positives(predictions, labels, class_id):
        return np.sum((predictions == class_id) & (labels != class_id))

    def false_negatives(predictions, labels, class_id):
        return np.sum((predictions != class_id) & (labels == class_id))

    def precision_for_class(predictions, labels, class_id):
        TP = metrics.true_positives(predictions, labels, class_id)
        FP = metrics.false_positives(predictions, labels, class_id)
        return TP / (TP + FP) if TP + FP > 0 else 0

    def recall_for_class(predictions, labels, class_id):
        TP = metrics.true_positives(predictions, labels, class_id)
        FN = metrics.false_negatives(predictions, labels, class_id)
        return TP / (TP + FN) if TP + FN > 0 else 0

    def precision(predictions, labels, num_classes):
        precisions = [
            metrics.precision_for_class(predictions, labels, class_id)
            for class_id in range(num_classes)
        ]
        return precisions

    def recall(predictions, labels, num_classes):
        recalls = [
            metrics.recall_for_class(predictions, labels, class_id)
            for class_id in range(num_classes)
        ]
        return recalls

    def f1_score(predictions, labels, num_classes):
        f1 = []
        for class_id in range(num_classes):
            p = metrics.precision_for_class(predictions, labels, class_id)
            r = metrics.recall_for_class(predictions, labels, class_id)
            if p != 0 and r != 0:
                cur_f1 = 2 * (p * r) / (p + r)
            else:
                cur_f1 = 0
            f1.append(cur_f1)
        return f1


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


class DataTransformer:
    def split_prep_data(data):
        data = np.array(data)
        np.random.shuffle(data)
        data_train = data.T
        Y_train = data_train[0]
        X_train = data_train[1:]
        X_train = X_train / np.max(X_train)
        X = X_train
        Y = Y_train
        return X, Y

    def display_data_image(data):
        current_image = data
        current_image = current_image.reshape((28, 28)) * 255
        plt.gray()
        plt.imshow(current_image, interpolation="nearest")
        plt.show()

    def get_model_prediction(A2):
        return np.argmax(A2, 0)

    def make_model_prediction(model, X):
        output, _ = model.forward(X)
        prediction = DataTransformer.get_model_prediction(output)
        return prediction

    def test_model_prediction(model, idx, X, Y, display=False):
        current_image = X[:, idx, None]
        prediction = DataTransformer.make_model_prediction(model, current_image)
        label = Y[idx]
        print(f"Predicted Value: {prediction} \n")
        print(f"Correct Value: {label} \n")
        if display:
            DataTransformer.display_data_image(current_image)
        return

    # THIS FUNCTION WAS TAKEN FROM https://github.com/zalandoresearch/fashion-mnist/tree/master !!!!!!
    def load_mnist(path, kind="train"):
        """Load MNIST data from `path`"""
        labels_path = os.path.join(path, "%s-labels-idx1-ubyte.gz" % kind)
        images_path = os.path.join(path, "%s-images-idx3-ubyte.gz" % kind)

        with gzip.open(labels_path, "rb") as lbpath:
            labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

        with gzip.open(images_path, "rb") as imgpath:
            images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(
                len(labels), 784
            )

        return images, labels
