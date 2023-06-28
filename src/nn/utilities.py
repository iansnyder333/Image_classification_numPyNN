import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


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
