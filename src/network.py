"""A small dense neural network with explicit backpropagation and Adam."""
import numpy as np

ARCHITECTURE = (47, 16, 8, 18)
SETTINGS = {"learning_rate": 0.003, "l2": 0.01, "max_epochs": 200,
            "patience": 20, "seed": 42, "validation_months": 12}


class DenseMLP:
    def __init__(self, architecture=ARCHITECTURE, seed=42, l2=0.01):
        rng = np.random.default_rng(seed)
        self.l2 = l2
        self.parameters = {}
        for layer, (inputs, outputs) in enumerate(zip(architecture[:-1], architecture[1:]), 1):
            self.parameters["W" + str(layer)] = rng.normal(
                0, np.sqrt(2 / inputs), size=(inputs, outputs))
            self.parameters["b" + str(layer)] = np.zeros(outputs)
        self.parameters["W3"] *= .05

    def forward(self, x):
        p = self.parameters
        z1 = x @ p["W1"] + p["b1"]
        a1 = np.maximum(z1, 0)
        z2 = a1 @ p["W2"] + p["b2"]
        a2 = np.maximum(z2, 0)
        prediction = a2 @ p["W3"] + p["b3"]
        return prediction, (x, z1, a1, z2, a2)

    def loss_and_gradients(self, x, y, mask):
        prediction, (x, z1, a1, z2, a2) = self.forward(x)
        counts = np.maximum(mask.sum(axis=0), 1)
        weight = mask / counts[None, :] / y.shape[1]
        error = prediction - y
        loss = np.sum(weight * error ** 2)
        loss += self.l2 / 2 * sum(
            np.sum(value ** 2) for key, value in self.parameters.items() if key.startswith("W"))
        d3 = 2 * error * weight
        gradients = {}
        gradients["W3"] = a2.T @ d3 + self.l2 * self.parameters["W3"]
        gradients["b3"] = d3.sum(axis=0)
        d2 = (d3 @ self.parameters["W3"].T) * (z2 > 0)
        gradients["W2"] = a1.T @ d2 + self.l2 * self.parameters["W2"]
        gradients["b2"] = d2.sum(axis=0)
        d1 = (d2 @ self.parameters["W2"].T) * (z1 > 0)
        gradients["W1"] = x.T @ d1 + self.l2 * self.parameters["W1"]
        gradients["b1"] = d1.sum(axis=0)
        return float(loss), gradients

    def fit(self, x, y, mask, epochs=200, learning_rate=.003, validation=None, patience=20):
        first = {key: np.zeros_like(value) for key, value in self.parameters.items()}
        second = {key: np.zeros_like(value) for key, value in self.parameters.items()}
        best_loss, best_epoch, stale = np.inf, 1, 0
        curve = []
        for epoch in range(1, epochs + 1):
            loss, gradients = self.loss_and_gradients(x, y, mask)
            if not np.isfinite(loss):
                raise ValueError("Neural training produced a non-finite loss")
            for name, gradient in gradients.items():
                first[name] = .9 * first[name] + .1 * gradient
                second[name] = .999 * second[name] + .001 * gradient ** 2
                corrected_m = first[name] / (1 - .9 ** epoch)
                corrected_v = second[name] / (1 - .999 ** epoch)
                self.parameters[name] -= learning_rate * corrected_m / (np.sqrt(corrected_v) + 1e-8)
            item = {"epoch": epoch, "train_loss_with_l2": loss}
            if validation is not None:
                vx, vy, vm = validation
                prediction = self.forward(vx)[0]
                counts = np.maximum(vm.sum(axis=0), 1)
                value = float(np.sum(((prediction - vy) ** 2) * vm / counts[None, :]) / vy.shape[1])
                item["validation_loss"] = value
                if value < best_loss - 1e-8:
                    best_loss, best_epoch, stale = value, epoch, 0
                else:
                    stale += 1
            curve.append(item)
            if validation is not None and stale >= patience:
                break
        return best_epoch if validation is not None else epochs, curve
