"""
游戏内神经网络与深度学习系统 - 验收测试
运行方式：python -m pytest tests/test_neural_network.py -q
共 15 个测试用例
"""
import pytest
import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.neural_network import NeuralNetwork


def make_network(**overrides):
    config = {
        "layer_sizes": [2, 4, 1],
        "activation": "relu",
        "output_activation": "sigmoid",
        "loss": "mse",
        "learning_rate": 0.1,
        "momentum": 0.9,
        "batch_norm": True,
        "seed": 42,
    }
    config.update(overrides)
    return NeuralNetwork(config)


class TestNeuralNetwork:
    def setup_method(self):
        self.system = NeuralNetwork()

    # 1. 激活函数 ReLU
    def test_case_01(self):
        nn = make_network()
        assert nn.activation_functions("relu", -3.0) == 0.0
        assert nn.activation_functions("relu", 2.5) == 2.5
        assert nn.activation_functions("relu", 0.0) == 0.0
        assert nn.activation_functions("relu", -1.0, derivative=True) == 0.0
        assert nn.activation_functions("relu", 4.0, derivative=True) == 1.0

    # 2. 激活函数 Sigmoid
    def test_case_02(self):
        nn = make_network()
        assert nn.activation_functions("sigmoid", 0.0) == pytest.approx(0.5)
        big = nn.activation_functions("sigmoid", 100.0)
        assert big == pytest.approx(1.0, abs=1e-6)
        small = nn.activation_functions("sigmoid", -100.0)
        assert small == pytest.approx(0.0, abs=1e-6)
        assert nn.activation_functions("sigmoid", 0.0,
                                       derivative=True) == pytest.approx(0.25)

    # 3. 损失函数 均方误差 MSE
    def test_case_03(self):
        nn = make_network()
        y_pred = [[1.0, 2.0], [3.0, 4.0]]
        y_true = [[1.0, 0.0], [3.0, 0.0]]
        assert nn.loss_functions("mse", y_pred, y_true) == pytest.approx(5.0)
        grad = nn.loss_functions("mse", y_pred, y_true, derivative=True)
        assert grad[0][0] == pytest.approx(0.0)
        assert grad[0][1] == pytest.approx(1.0)
        assert grad[1][1] == pytest.approx(2.0)

    # 4. 损失函数 交叉熵 CrossEntropy
    def test_case_04(self):
        nn = make_network()
        y_pred = [[0.9], [0.1]]
        y_true = [[1.0], [0.0]]
        expected = -(math.log(0.9) + math.log(0.9)) / 2
        assert nn.loss_functions("cross_entropy", y_pred,
                                 y_true) == pytest.approx(expected, rel=1e-5)
        grad = nn.loss_functions("cross_entropy", [[0.75]], [[1.0]],
                                 derivative=True)
        assert grad[0][0] == pytest.approx(-1.0 / 0.75, rel=1e-5)
        grad = nn.loss_functions("cross_entropy", [[0.75]], [[0.0]],
                                 derivative=True)
        assert grad[0][0] == pytest.approx(1.0 / 0.25, rel=1e-5)

    # 5. 前向传播输出形状与取值范围
    def test_case_05(self):
        nn = make_network()
        X = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        out = nn.forward_propagation(X)
        assert len(out) == 3
        assert all(len(row) == 1 for row in out)
        assert all(0.0 < row[0] < 1.0 for row in out)

    # 6. 前向传播确实应用了激活函数（不是单纯矩阵乘法）
    def test_case_06(self):
        nn = make_network(activation="relu")
        nn.forward_propagation([[0.1, 0.2], [0.3, 0.4]])
        hidden = nn._cache[0]["a"]
        assert all(v >= 0.0 for row in hidden for v in row)
        z = nn._cache[0]["z"]
        assert z != hidden  # 经过 BN+ReLU，输出与原始线性结果不同

    # 7. 批量归一化：归一化后的隐层特征均值约 0、方差约 1
    def test_case_07(self):
        nn = make_network(batch_norm=True)
        X = [[0.1 * i + 0.05, 0.2 * i - 0.3] for i in range(1, 21)]
        nn.forward_propagation(X)
        z_norm = nn._cache[0]["bn"]["z_norm"]
        m = len(z_norm)
        means = [sum(row[j] for row in z_norm) / m for j in range(4)]
        vars_ = [sum((row[j] - means[j]) ** 2 for row in z_norm) / m
                 for j in range(4)]
        assert all(abs(mu) < 1e-6 for mu in means)
        assert all(abs(v - 1.0) < 1e-6 for v in vars_)

    # 8. 反向传播返回与权重同形的梯度
    def test_case_08(self):
        nn = make_network()
        X = [[0.1, 0.2], [0.3, 0.4]]
        y = [[1.0], [0.0]]
        grads = nn.backpropagation(X, y)
        for key, param in nn.params.items():
            assert key in grads
            assert len(grads[key]) == len(param)
            assert len(grads[key][0]) == len(param[0])

    # 9. 梯度检查：解析梯度与数值梯度一致（链式法则正确）
    def test_case_09(self):
        nn = make_network(loss="cross_entropy")
        X = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.1]]
        y = [[1.0], [0.0], [1.0]]
        result = nn.gradient_check(X, y)
        assert result["passed"]
        assert result["max_relative_error"] < 1e-4

    # 10. 梯度下降真正更新权重
    def test_case_10(self):
        nn = make_network(loss="cross_entropy", momentum=0.0)
        X = [[0.1, 0.2], [0.3, 0.4]]
        y = [[1.0], [0.0]]
        before = {k: [row[:] for row in v] for k, v in nn.params.items()}
        nn.gradient_descent(X, y, epochs=1)
        changed = any(
            nn.params[k][i][j] != before[k][i][j]
            for k in nn.params for i in range(len(nn.params[k]))
            for j in range(len(nn.params[k][0]))
        )
        assert changed

    # 11. 学习率控制更新步长
    def test_case_11(self):
        X = [[0.1, 0.2], [0.3, 0.4]]
        y = [[1.0], [0.0]]
        nns = [make_network(loss="cross_entropy", momentum=0.0,
                            learning_rate=lr) for lr in (0.01, 0.1)]
        initial = {k: [row[:] for row in v] for k, v in nns[0].params.items()}
        total = []
        for nn in nns:
            nn.gradient_descent(X, y, epochs=1)
            total.append(sum(
                abs(nn.params[k][i][j] - initial[k][i][j])
                for k in nn.params for i in range(len(nn.params[k]))
                for j in range(len(nn.params[k][0]))
            ))
        assert total[1] == pytest.approx(10.0 * total[0], rel=1e-8)

    # 12. 动量：第二步包含累积速度，更新量大于纯梯度步长
    def test_case_12(self):
        nn = make_network(loss="cross_entropy", momentum=0.9,
                          learning_rate=0.1)
        X = [[0.1, 0.2], [0.3, 0.4]]
        y = [[1.0], [0.0]]
        start = {k: [row[:] for row in v] for k, v in nn.params.items()}

        grads1 = nn.backpropagation(X, y)
        nn.gradient_descent(X, y, epochs=1)
        after1 = {k: [row[:] for row in v] for k, v in nn.params.items()}
        step1 = sum(abs(after1[k][i][j] - start[k][i][j])
                    for k in nn.params for i in range(len(nn.params[k]))
                    for j in range(len(nn.params[k][0])))

        grads2 = nn.backpropagation(X, y)
        nn.gradient_descent(X, y, epochs=1)
        expected_step2 = sum(
            abs(0.9 * 0.1 * grads1[k][i][j]
                + 0.1 * grads2[k][i][j])
            for k in nn.params for i in range(len(nn.params[k]))
            for j in range(len(nn.params[k][0]))
        )
        actual_step2 = sum(
            abs(nn.params[k][i][j] - after1[k][i][j])
            for k in nn.params for i in range(len(nn.params[k]))
            for j in range(len(nn.params[k][0]))
        )
        assert actual_step2 == pytest.approx(expected_step2, rel=1e-8)
        assert actual_step2 > step1

    # 13. 训练持续降低损失
    def test_case_13(self):
        nn = make_network(activation="relu", output_activation="linear",
                          loss="mse", learning_rate=0.05, momentum=0.0,
                          batch_norm=False)
        X = [[i / 10.0, (i % 3) / 3.0] for i in range(20)]
        y = [[2.0 * row[0] - row[1] + 0.5] for row in X]
        losses = nn.train(X, y, epochs=200)
        assert losses[-1] < losses[0]
        assert losses[-1] < 0.01

    # 14. 网络真正学会 OR 逻辑（不再是随机输出）
    def test_case_14(self):
        nn = make_network(loss="cross_entropy", learning_rate=0.5,
                          momentum=0.0, batch_norm=False)
        X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
        y = [[0.0], [1.0], [1.0], [1.0]]
        nn.train(X, y, epochs=500)
        out = [row[0] for row in nn.forward_propagation(X, training=False)]
        pred = [1 if p > 0.5 else 0 for p in out]
        assert pred == [0, 1, 1, 1]
        assert out[0] < 0.1
        assert min(out[1:]) > 0.9

    # 15. reset 清空训练历史与内部状态
    def test_case_15(self):
        nn = make_network(loss="cross_entropy")
        X = [[0.1, 0.2], [0.3, 0.4]]
        y = [[1.0], [0.0]]
        nn.gradient_descent(X, y, epochs=3)
        assert len(nn._history) == 3
        nn.reset()
        assert nn._history == []
        assert nn._state == {}
        assert nn._cache == []
        assert all(v == 0.0 for vel in nn._velocities.values()
                   for row in vel for v in row)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
