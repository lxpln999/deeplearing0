from matplotlib import pyplot as plt
import paddle


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

plt.scatter(X_train, y_train, facecolor="none",
            edgecolor="b", s=50, label="train data")
plt.scatter(X_test, y_test, facecolor="none",
            edgecolor="r", s=50, label="test data")
plt.plot(X_underlying, y_underlying, c="g", label="underlying distribution")
plt.legend()
plt.show()
