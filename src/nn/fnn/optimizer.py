import numpy as np
import pandas as pd


class SimpleGradientDescent:
    def __init__(self, model, lr):
        self.model = model
        self.params = model.parameters()
        self.alpha = lr

    def step(self, gradients):
        p = {}
        for idx in range(self.model.layers):
            layer_idx = idx + 1
            w = self.params["W" + str(layer_idx)]
            b = self.params["b" + str(layer_idx)]
            dw = gradients["dW" + str(layer_idx)]
            db = gradients["db" + str(layer_idx)]
            new_w = w - self.alpha * dw
            new_b = b - self.alpha * db
            p["W" + str(layer_idx)] = new_w
            p["b" + str(layer_idx)] = new_b

        self.model.update(p)
        self.params = self.model.parameters()
