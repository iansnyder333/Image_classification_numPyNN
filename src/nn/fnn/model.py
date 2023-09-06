import numpy as np
import pandas as pd
import json
from json import JSONEncoder
import io
import os
from utilities import NumpyArrayEncoder


class numpy_FNN:
    def __init__(
        self, input_size: int, hidden_size: int, output_size: int, hidden_layers=1
    ) -> None:
        self.params = self._init_layers(
            input_size, hidden_size, output_size, hidden_layers
        )
        self.layers = len(self.params) // 2

    def _init_layers(
        self, input_size: int, hidden_size: int, output_size: int, hidden_layers: int
    ) -> dict:
        param_values = {}
        for idx in range(hidden_layers):
            layer_idx = idx + 1
            param_values["W" + str(layer_idx)] = (
                np.random.rand(hidden_size, input_size) - 0.5
            )
            param_values["b" + str(layer_idx)] = (
                np.random.rand(hidden_size, hidden_layers) - 0.5
            )
        param_values["W" + str(hidden_layers + 1)] = (
            np.random.rand(output_size, hidden_size) - 0.5
        )
        param_values["b" + str(hidden_layers + 1)] = (
            np.random.rand(output_size, hidden_layers) - 0.5
        )
        return param_values

    def forward(self, X: np.ndarray) -> tuple:
        memory = {}
        cur_A = X
        memory["A0"] = cur_A
        for idx in range(self.layers):
            layer_idx = idx + 1
            prev_A = cur_A
            layer_W = self.params["W" + str(layer_idx)]
            layer_b = self.params["b" + str(layer_idx)]
            cur_Z = layer_W @ prev_A + layer_b
            cur_A = (
                self.ReLU(cur_Z) if layer_idx != self.layers else self.softmax(cur_Z)
            )

            memory["Z" + str(layer_idx)] = cur_Z
            memory["A" + str(layer_idx)] = cur_A
        return cur_A, memory

    def backward(self, Y: np.ndarray, memory: dict) -> dict:
        gradients = {}
        encoded_Y = self.one_hot(Y)
        cur_A = memory["A" + str(self.layers)]
        cur_DZ = cur_A - encoded_Y
        for idx in range(self.layers, 1, -1):
            prev_DZ = cur_DZ
            prev_layer = idx - 1
            prev_A = memory["A" + str(prev_layer)]
            cur_DW = 1 / Y.size * prev_DZ.dot(prev_A.T)
            cur_Db = 1 / Y.size * np.sum(prev_DZ)
            cur_DZ = self.params["W" + str(idx)].T.dot(prev_DZ) * self.ReLU_deriv(
                memory["Z" + str(prev_layer)]
            )
            gradients["db" + str(idx)] = cur_Db
            gradients["dW" + str(idx)] = cur_DW
        gradients["db1"] = 1 / Y.size * np.sum(cur_DZ)
        gradients["dW1"] = 1 / Y.size * cur_DZ.dot(memory["A0"].T)

        return gradients

    def load_model_state(self, path: str) -> bool:
        try:
            with open(path) as data_file:
                data_loaded = json.load(data_file)
            loaded_params = {}
            for key, value in data_loaded["model_params"].items():
                loaded_params[key] = np.asarray(value)
            self.params = loaded_params
            self.layers = len(self.params) // 2
        except:
            print("Model failed to load")
            return False

    def save_model_state(
        self, name="model0", path="Image_classification_numPyNN/src/models"
    ) -> None:
        data = {"model_params": self.parameters()}
        try:
            to_unicode = unicode
        except NameError:
            to_unicode = str
        filename = "".join([name, ".json"])
        filepath = os.path.join(path, filename)
        with io.open(filepath, "w", encoding="utf8") as outfile:
            str_ = json.dumps(
                data,
                cls=NumpyArrayEncoder,
                indent=4,
                sort_keys=True,
                separators=(",", ": "),
                ensure_ascii=False,
            )
            outfile.write(to_unicode(str_))

    def parameters(self) -> dict:
        return self.params

    def update(self, new_params: dict) -> dict:
        self.params = new_params

    def ReLU(self, Z: np.ndarray):
        # Returns raw value if above zero, else zero
        return np.maximum(Z, 0)

    def ReLU_deriv(self, Z: np.ndarray):
        return Z > 0

    def softmax(self, Z: np.ndarray):
        # Z_max = np.max(Z)
        # A = np.exp(Z - Z_max) / np.sum(np.exp(Z - Z_max))
        A = np.exp(Z) / sum(np.exp(Z))
        return A

    def one_hot(self, Y: np.ndarray):
        one_hot_Y = np.zeros((Y.size, Y.max() + 1))  # Initializing the one-hot matrix
        one_hot_Y[np.arange(Y.size), Y] = 1  # Setting the appropriate indices to 1
        one_hot_Y = one_hot_Y.T  # Transposing the matrix to match the expected shape
        return one_hot_Y
