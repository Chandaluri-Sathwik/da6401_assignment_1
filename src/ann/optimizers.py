"""
Optimization algorithms for training neural networks.
Supports: SGD, Momentum, NAG, RMSProp, Adam, Nadam.
"""

import numpy as np


class Optimizer:
    def __init__(
        self,
        name = "sgd",
        learning_rate = 1e-3,
        weight_decay = 0.0,
        beta1 = 0.9,
        beta2 = 0.999,
        epsilon = 1e-8,
    ) -> None:
        self.name = name.lower()
        self.lr = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.epsilon = float(epsilon)

        self.t = 0  

        self.m_w = {}
        self.m_b = {}
        self.v_w = {}
        self.v_b = {}
        self.s_w = {}
        self.s_b = {}

        supported = {"sgd", "momentum", "nag", "rmsprop", "adam", "nadam"}
        if self.name not in supported:
            raise ValueError(f"Unsupported optimizer '{self.name}'. Supported: {sorted(supported)}")

    def _init_state(self, layer_idx: int, layer) -> None:
        if layer_idx in self.m_w:
            return
        self.m_w[layer_idx] = np.zeros_like(layer.W)
        self.m_b[layer_idx] = np.zeros_like(layer.b)
        self.v_w[layer_idx] = np.zeros_like(layer.W)
        self.v_b[layer_idx] = np.zeros_like(layer.b)
        self.s_w[layer_idx] = np.zeros_like(layer.W)
        self.s_b[layer_idx] = np.zeros_like(layer.b)

    def step(self, layers) -> None:
        """
        Update all layers in-place.
        Expects each layer to expose: W, b, grad_W, grad_b
        """
        self.t += 1

        for i, layer in enumerate(layers):
            if layer.grad_W is None or layer.grad_b is None:
                continue

            self._init_state(i, layer)

            g_w = layer.grad_W.copy()
            g_b = layer.grad_b.copy()

            # L2 weight decay
            if self.weight_decay > 0.0:
                g_w = g_w + self.weight_decay * layer.W

            if self.name == "sgd":
                self._sgd(layer, g_w, g_b)
            elif self.name == "momentum":
                self._momentum(layer, i, g_w, g_b)
            elif self.name == "nag":
                self._nag(layer, i, g_w, g_b)
            elif self.name == "rmsprop":
                self._rmsprop(layer, i, g_w, g_b)
            elif self.name == "adam":
                self._adam(layer, i, g_w, g_b)
            elif self.name == "nadam":
                self._nadam(layer, i, g_w, g_b)

    def _sgd(self, layer, g_w, g_b) -> None:
        layer.W -= self.lr * g_w
        layer.b -= self.lr * g_b

    def _momentum(self, layer, i: int, g_w, g_b) -> None:
        self.v_w[i] = self.beta1 * self.v_w[i] + (1.0 - self.beta1) * g_w
        self.v_b[i] = self.beta1 * self.v_b[i] + (1.0 - self.beta1) * g_b
        layer.W -= self.lr * self.v_w[i]
        layer.b -= self.lr * self.v_b[i]

    def _nag(self, layer, i: int, g_w, g_b) -> None:
        v_w_prev = self.v_w[i].copy()
        v_b_prev = self.v_b[i].copy()

        self.v_w[i] = self.beta1 * self.v_w[i] + (1.0 - self.beta1) * g_w
        self.v_b[i] = self.beta1 * self.v_b[i] + (1.0 - self.beta1) * g_b

        layer.W -= self.lr * ((1.0 + self.beta1) * self.v_w[i] - self.beta1 * v_w_prev)
        layer.b -= self.lr * ((1.0 + self.beta1) * self.v_b[i] - self.beta1 * v_b_prev)

    def _rmsprop(self, layer, i: int, g_w, g_b) -> None:
        self.s_w[i] = self.beta2 * self.s_w[i] + (1.0 - self.beta2) * (g_w ** 2)
        self.s_b[i] = self.beta2 * self.s_b[i] + (1.0 - self.beta2) * (g_b ** 2)

        layer.W -= self.lr * g_w / (np.sqrt(self.s_w[i]) + self.epsilon)
        layer.b -= self.lr * g_b / (np.sqrt(self.s_b[i]) + self.epsilon)

    def _adam(self, layer, i: int, g_w, g_b) -> None:
        self.m_w[i] = self.beta1 * self.m_w[i] + (1.0 - self.beta1) * g_w
        self.m_b[i] = self.beta1 * self.m_b[i] + (1.0 - self.beta1) * g_b
        self.s_w[i] = self.beta2 * self.s_w[i] + (1.0 - self.beta2) * (g_w ** 2)
        self.s_b[i] = self.beta2 * self.s_b[i] + (1.0 - self.beta2) * (g_b ** 2)

        m_w_hat = self.m_w[i] / (1.0 - self.beta1 ** self.t)
        m_b_hat = self.m_b[i] / (1.0 - self.beta1 ** self.t)
        s_w_hat = self.s_w[i] / (1.0 - self.beta2 ** self.t)
        s_b_hat = self.s_b[i] / (1.0 - self.beta2 ** self.t)

        layer.W -= self.lr * m_w_hat / (np.sqrt(s_w_hat) + self.epsilon)
        layer.b -= self.lr * m_b_hat / (np.sqrt(s_b_hat) + self.epsilon)

    def _nadam(self, layer, i: int, g_w, g_b) -> None:
        self.m_w[i] = self.beta1 * self.m_w[i] + (1.0 - self.beta1) * g_w
        self.m_b[i] = self.beta1 * self.m_b[i] + (1.0 - self.beta1) * g_b
        self.s_w[i] = self.beta2 * self.s_w[i] + (1.0 - self.beta2) * (g_w ** 2)
        self.s_b[i] = self.beta2 * self.s_b[i] + (1.0 - self.beta2) * (g_b ** 2)

        m_w_hat = self.m_w[i] / (1.0 - self.beta1 ** self.t)
        m_b_hat = self.m_b[i] / (1.0 - self.beta1 ** self.t)
        s_w_hat = self.s_w[i] / (1.0 - self.beta2 ** self.t)
        s_b_hat = self.s_b[i] / (1.0 - self.beta2 ** self.t)

        m_w_nesterov = self.beta1 * m_w_hat + ((1.0 - self.beta1) * g_w) / (1.0 - self.beta1 ** self.t)
        m_b_nesterov = self.beta1 * m_b_hat + ((1.0 - self.beta1) * g_b) / (1.0 - self.beta1 ** self.t)

        layer.W -= self.lr * m_w_nesterov / (np.sqrt(s_w_hat) + self.epsilon)
        layer.b -= self.lr * m_b_nesterov / (np.sqrt(s_b_hat) + self.epsilon)
