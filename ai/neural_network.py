"""
游戏内神经网络与深度学习系统 - 核心模块

纯 Python 标准库实现：
- 激活函数：ReLU / Sigmoid（含导数）
- 损失函数：均方误差 MSE / 交叉熵 CrossEntropy（含梯度）
- 前向传播：矩阵乘法 + 激活函数 + 批量归一化
- 反向传播：链式法则求梯度 + 数值梯度检查
- 梯度下降：学习率 + 动量
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math
import random
import time


@dataclass
class Config:
    layer_sizes: List[int] = field(default_factory=lambda: [2, 4, 1])
    activation: str = "relu"
    output_activation: str = "sigmoid"
    loss: str = "mse"
    learning_rate: float = 0.1
    momentum: float = 0.9
    batch_norm: bool = True
    seed: int = 42


def _is_matrix(x):
    return isinstance(x, list) and len(x) > 0 and isinstance(x[0], list)


def _to_matrix(x):
    if _is_matrix(x):
        return [list(row) for row in x]
    if isinstance(x, list):
        return [list(x)]
    return [[float(x)]]


def _matmul(a, b):
    rows, inner, cols = len(a), len(b), len(b[0])
    out = [[0.0] * cols for _ in range(rows)]
    for i in range(rows):
        ai = a[i]
        oi = out[i]
        for k in range(inner):
            aik = ai[k]
            bk = b[k]
            for j in range(cols):
                oi[j] += aik * bk[j]
    return out


def _transpose(a):
    return [list(col) for col in zip(*a)]


def _zeros_like(a):
    return [[0.0] * len(row) for row in a]


def _relu(x):
    return x if x > 0.0 else 0.0


def _relu_derivative(x):
    return 1.0 if x > 0.0 else 0.0


def _sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def _sigmoid_derivative(x):
    s = _sigmoid(x)
    return s * (1.0 - s)


def _linear(x):
    return x


def _linear_derivative(x):
    return 1.0


class NeuralNetwork:
    EPS = 1e-8

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._state = {}
        self._history = []

        cfg = self.config
        self.layer_sizes = list(cfg.get("layer_sizes", [2, 4, 1]))
        self.activation = cfg.get("activation", "relu")
        self.output_activation = cfg.get("output_activation", "sigmoid")
        self.loss = cfg.get("loss", "mse")
        self.learning_rate = float(cfg.get("learning_rate", 0.1))
        self.momentum = float(cfg.get("momentum", 0.9))
        self.use_batch_norm = bool(cfg.get("batch_norm", True))
        seed = cfg.get("seed", 42)

        rng = random.Random(seed)
        self.params: Dict[str, List[List[float]]] = {}
        self.num_layers = len(self.layer_sizes) - 1
        for l in range(1, self.num_layers + 1):
            n_in, n_out = self.layer_sizes[l - 1], self.layer_sizes[l]
            limit = math.sqrt(6.0 / (n_in + n_out))
            self.params[f"W{l}"] = [
                [rng.uniform(-limit, limit) for _ in range(n_out)]
                for _ in range(n_in)
            ]
            self.params[f"b{l}"] = [[0.0] * n_out]
            if l < self.num_layers and self.use_batch_norm:
                self.params[f"gamma{l}"] = [[1.0] * n_out]
                self.params[f"beta{l}"] = [[0.0] * n_out]

        self._velocities = {k: _zeros_like(v) for k, v in self.params.items()}
        self._cache: List[Dict] = []

    # ------------------------------------------------------------------
    # 激活函数
    # ------------------------------------------------------------------
    _ACTIVATIONS = {
        "relu": (_relu, _relu_derivative),
        "sigmoid": (_sigmoid, _sigmoid_derivative),
        "linear": (_linear, _linear_derivative),
        "identity": (_linear, _linear_derivative),
    }

    def activation_functions(self, name: str, x, derivative: bool = False):
        """按名称应用激活函数（或其导数），支持标量 / 向量 / 矩阵。"""
        if name not in self._ACTIVATIONS:
            raise ValueError(f"未知激活函数: {name}")
        fn = self._ACTIVATIONS[name][1 if derivative else 0]
        if isinstance(x, (int, float)):
            return fn(float(x))
        if _is_matrix(x):
            return [[fn(v) for v in row] for row in x]
        return [fn(v) for v in x]

    def _apply_activation(self, name, matrix, derivative=False):
        fn = self._ACTIVATIONS[name][1 if derivative else 0]
        return [[fn(v) for v in row] for row in matrix]

    # ------------------------------------------------------------------
    # 损失函数
    # ------------------------------------------------------------------
    def loss_functions(self, name: str, y_pred, y_true, derivative: bool = False):
        """计算损失值（derivative=False）或对预测的梯度（derivative=True）。"""
        pred, true = _to_matrix(y_pred), _to_matrix(y_true)
        m, n = len(pred), len(pred[0])
        if name == "mse":
            if derivative:
                scale = 2.0 / (m * n)
                return [[(p - t) * scale for p, t in zip(pr, tr)]
                        for pr, tr in zip(pred, true)]
            return sum((p - t) ** 2 for pr, tr in zip(pred, true)
                       for p, t in zip(pr, tr)) / (m * n)
        if name == "cross_entropy":
            eps = self.EPS
            if derivative:
                scale = 1.0 / (m * n)
                return [[(-t / max(p, eps)
                          + (1.0 - t) / max(1.0 - p, eps)) * scale
                         for p, t in zip(pr, tr)]
                        for pr, tr in zip(pred, true)]
            total = 0.0
            for pr, tr in zip(pred, true):
                for p, t in zip(pr, tr):
                    p = min(max(p, eps), 1.0 - eps)
                    total -= t * math.log(p) + (1.0 - t) * math.log(1.0 - p)
            return total / (m * n)
        raise ValueError(f"未知损失函数: {name}")

    # ------------------------------------------------------------------
    # 前向传播（矩阵乘法 + 激活函数 + 批量归一化）
    # ------------------------------------------------------------------
    def forward_propagation(self, X, training: bool = True):
        a = _to_matrix(X)
        self._cache = []
        for l in range(1, self.num_layers + 1):
            cache = {"a_prev": a}
            z = _matmul(a, self.params[f"W{l}"])
            b = self.params[f"b{l}"][0]
            z = [[v + b[j] for j, v in enumerate(row)] for row in z]
            cache["z"] = z

            is_hidden = l < self.num_layers
            if is_hidden and self.use_batch_norm:
                z_out = self._batch_norm_forward(z, l, cache, training)
            else:
                z_out = z
            cache["z_hat"] = z_out

            act = self.activation if is_hidden else self.output_activation
            a = self._apply_activation(act, z_out)
            cache["a"] = a
            self._cache.append(cache)

        self._state["forward_cache"] = self._cache
        return a

    def _batch_norm_forward(self, z, layer, cache, training):
        m, n = len(z), len(z[0])
        mean = [sum(row[j] for row in z) / m for j in range(n)]
        var = [sum((row[j] - mean[j]) ** 2 for row in z) / m for j in range(n)]
        inv_std = [1.0 / math.sqrt(v + self.EPS) for v in var]
        z_norm = [[(row[j] - mean[j]) * inv_std[j] for j in range(n)]
                  for row in z]
        gamma = self.params[f"gamma{layer}"][0]
        beta = self.params[f"beta{layer}"][0]
        z_hat = [[gamma[j] * zn[j] + beta[j] for j in range(n)]
                 for zn in z_norm]
        cache["bn"] = {"z": z, "mean": mean, "var": var,
                       "inv_std": inv_std, "z_norm": z_norm}
        return z_hat

    # ------------------------------------------------------------------
    # 反向传播（链式法则）
    # ------------------------------------------------------------------
    def backpropagation(self, X, y) -> Dict[str, List[List[float]]]:
        y_pred = self.forward_propagation(X, training=True)
        m = len(y_pred)
        da = self.loss_functions(self.loss, y_pred, y, derivative=True)

        grads: Dict[str, List[List[float]]] = {}
        for l in range(self.num_layers, 0, -1):
            cache = self._cache[l - 1]
            is_hidden = l < self.num_layers
            act = self.activation if is_hidden else self.output_activation
            act_grad = self._apply_activation(act, cache["z_hat"],
                                              derivative=True)
            dz = [[da[i][j] * act_grad[i][j] for j in range(len(da[0]))]
                  for i in range(m)]

            if is_hidden and self.use_batch_norm:
                dz, dgamma, dbeta = self._batch_norm_backward(dz, l,
                                                              cache["bn"])
                grads[f"gamma{l}"] = dgamma
                grads[f"beta{l}"] = dbeta

            a_prev_t = _transpose(cache["a_prev"])
            dW = _matmul(a_prev_t, dz)
            grads[f"W{l}"] = dW
            grads[f"b{l}"] = [[sum(row[j] for row in dz)
                               for j in range(len(dz[0]))]]
            da = _matmul(dz, _transpose(self.params[f"W{l}"]))
        return grads

    def _batch_norm_backward(self, dz_hat, layer, bn):
        z, mean, var = bn["z"], bn["mean"], bn["var"]
        inv_std, z_norm = bn["inv_std"], bn["z_norm"]
        gamma = self.params[f"gamma{layer}"][0]
        m, n = len(dz_hat), len(dz_hat[0])

        dgamma = [[sum(dz_hat[i][j] * z_norm[i][j] for i in range(m))
                   for j in range(n)]]
        dbeta = [[sum(dz_hat[i][j] for i in range(m)) for j in range(n)]]

        dz_norm = [[dz_hat[i][j] * gamma[j] for j in range(n)]
                   for i in range(m)]
        dvar = [
            sum(dz_norm[i][j] * (z[i][j] - mean[j])
                * -0.5 * (var[j] + self.EPS) ** -1.5
                for i in range(m))
            for j in range(n)
        ]
        dmean = [
            sum(dz_norm[i][j] * -inv_std[j] for i in range(m))
            + dvar[j] * sum(-2.0 * (z[i][j] - mean[j])
                            for i in range(m)) / m
            for j in range(n)
        ]
        dz = [
            [
                dz_norm[i][j] * inv_std[j]
                + dvar[j] * 2.0 * (z[i][j] - mean[j]) / m
                + dmean[j] / m
                for j in range(n)
            ]
            for i in range(m)
        ]
        return dz, dgamma, dbeta

    # ------------------------------------------------------------------
    # 梯度检查（数值梯度 vs 解析梯度）
    # ------------------------------------------------------------------
    def gradient_check(self, X, y, epsilon: float = 1e-5,
                       tolerance: float = 1e-4) -> Dict:
        analytic = self.backpropagation(X, y)
        max_rel_error = 0.0
        for key, param in self.params.items():
            grad = analytic[key]
            for i in range(len(param)):
                for j in range(len(param[0])):
                    orig = param[i][j]
                    param[i][j] = orig + epsilon
                    loss_plus = self.loss_functions(
                        self.loss, self.forward_propagation(X), y)
                    param[i][j] = orig - epsilon
                    loss_minus = self.loss_functions(
                        self.loss, self.forward_propagation(X), y)
                    param[i][j] = orig
                    numerical = (loss_plus - loss_minus) / (2.0 * epsilon)
                    denom = max(abs(numerical) + abs(grad[i][j]), 1e-8)
                    max_rel_error = max(max_rel_error,
                                        abs(numerical - grad[i][j]) / denom)
        return {"max_relative_error": max_rel_error,
                "passed": max_rel_error < tolerance}

    # ------------------------------------------------------------------
    # 梯度下降（学习率 + 动量）
    # ------------------------------------------------------------------
    def gradient_descent(self, X, y, epochs: int = 1) -> List[float]:
        losses = []
        for _ in range(epochs):
            grads = self.backpropagation(X, y)
            for key, param in self.params.items():
                grad, vel = grads[key], self._velocities[key]
                for i in range(len(param)):
                    for j in range(len(param[0])):
                        vel[i][j] = (self.momentum * vel[i][j]
                                     - self.learning_rate * grad[i][j])
                        param[i][j] += vel[i][j]
            loss = self.loss_functions(
                self.loss, self.forward_propagation(X, training=False), y)
            losses.append(loss)
            self._history.append({"loss": loss, "time": time.time()})
        return losses

    def train(self, X, y, epochs: int = 100) -> List[float]:
        return self.gradient_descent(X, y, epochs=epochs)

    # ------------------------------------------------------------------
    # 游戏循环钩子
    # ------------------------------------------------------------------
    def update(self, dt: float):
        pass

    def reset(self):
        self._state = {}
        self._history = []
        self._cache = []
        self._velocities = {k: _zeros_like(v) for k, v in self.params.items()}
