import math
import paddle
from Op import Op
from matplotlib import pyplot as plt
import paddle

paddle.seed(10)

# 线性算子


def create_toy_data(func, interval, sample_num, noise=0.0, add_outlier=False, outlier_ratio=0.001):
    X = paddle.rand(shape=[sample_num]) * \
        (interval[1]-interval[0]) + interval[0]
    y = func(X)
    epsilon = paddle.normal(0, noise, paddle.to_tensor(y.shape[0]))
    y = y + epsilon
    if add_outlier:
        outlier_num = int(len(y)*outlier_ratio)
        if outlier_num != 0:
            outlier_idx = paddle.randint(len(y), shage=[outlier_num])
            y[outlier_idx] = y[outlier_idx] * 5
    return X, y


def liner_func(x, w=1.2, b=0.5):
    y = w * x + b
    return y


X_train, y_train = create_toy_data(
    func=liner_func, interval=(-10, 10), sample_num=100, noise=2, add_outlier=False)

X_test, y_test = create_toy_data(
    func=liner_func, interval=(-10, 10), sample_num=50, noise=2, add_outlier=False)

X_underlying = paddle.linspace(-10, 10, 100)
y_underlying = liner_func(X_underlying)

# plt.scatter(X_train, y_train, facecolor="none",
#            edgecolor="b", s=50, label="train data")
# plt.scatter(X_test, y_test, facecolor="none",
#            edgecolor="r", s=50, label="test data")
# plt.plot(X_underlying, y_underlying, c="g", label="underlying distribution")
# plt.legend()
# plt.show()


class LinearT(Op):
    def __init__(self, input_size):
        """
        输入：
           - input_size:模型要处理的数据特征向量长度
        """
        self.input_size = input_size
        # 模型参数
        self.params = {}
        self.params['w'] = paddle.randn(
            shape=[self.input_size, 1], dtype='float32')
        self.params['b'] = paddle.zeros(shape=[1], dtype='float32')

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        N, D = X.shape

        # if self.dim == 0:
        #    return paddle.full(shape=[N, 1], fill_value=self.params['b'])

        assert D == self.input_size
        y_pred = paddle.matmul(X, self.params['w'])+self.params['b']
        return y_pred


input_size = 3
N = 2
X = paddle.randn(shape=[N, input_size], dtype='float32')
model = LinearT(input_size)
y_pred = model(X)

print('y_pred', y_pred)


def mean_squared_error(y_true, y_pred):
    assert y_true.shape[0] == y_true.shape[0]
    error = paddle.mean(paddle.square(y_true-y_pred))
    return error


y_true = paddle.to_tensor([[-0.2], [4.9]], dtype='float32')
print('y_true.shape', y_true.shape)

y_pred = paddle.to_tensor([[1.3], [2.5]], dtype='float32')
print('y_pred.shape', y_pred.shape)

error = mean_squared_error(y_true, y_pred)

print('error', error)


def optimizer_lsm(model, X, y, reg_lambda=0):
    N, D = X.shape

    x_bar_tran = paddle.mean(X, axis=0).T
    y_bar = paddle.mean(y)

    x_sub = paddle.subtract(X, x_bar_tran)

    if paddle.all(x_sub == 0):
        model.params['b'] = y_bar
        model.params['w'] = paddle.zeros(shape=[D])
        return model

    tmp = paddle.inverse(paddle.matmul(x_sub.T, x_sub) +
                         reg_lambda*paddle.eye(D))
    w = paddle.matmul(paddle.matmul(tmp, x_sub.T), (y-y_bar))
    b = y_bar-paddle.matmul(x_bar_tran, w)

    model.params['b'] = b
    model.params['w'] = paddle.squeeze(w, axis=-1)

    return model


input_size = 1
model = LinearT(input_size)
model = optimizer_lsm(model, X_train.reshape(
    [-1, 1]), y_train.reshape([-1, 1]))

print('w_pred', model.params['w'].item(), 'b_pred', model.params['b'].item())

y_train_pred = model(X_train.reshape([-1, 1])).squeeze()
train_error = mean_squared_error(y_train, y_train_pred).item()

print('train_error', train_error)

y_test_pred = model(X_test.reshape([-1, 1])).squeeze()
test_error = mean_squared_error(y_test, y_test_pred).item()

print('test_error', test_error)


def sin(x):
    y = paddle.sin(2 * math.pi * x)
    return y


X_train, y_train = create_toy_data(
    func=sin, interval=(0, 1), sample_num=15, noise=0.1)
X_test, y_test = create_toy_data(
    func=sin, interval=(0, 1), sample_num=10, noise=0.1)

X_underlying = paddle.linspace(0, 1, 100)
y_underlying = sin(X_underlying)

plt.rcParams['figure.figsize'] = (8, 6)
plt.scatter(X_train, y_train, facecolor="none",
            edgecolors="b", s=50, label="train data")
plt.plot(X_underlying, y_underlying, c="g", label=r"$\sin(2\pi x)$")
plt.legend()
plt.show()
