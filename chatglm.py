import paddle

print(paddle.__version__)

ndim_4_tensor = paddle.ones([2, 3, 4, 5],place=paddle.CPUPlace())

print("number of dimensions: ", ndim_4_tensor.ndim)

# shapge = number of each dimension
print("shape of tensor: ", ndim_4_tensor.shape)

print("elements number along axis 2 of tensor: ", ndim_4_tensor.shape[2])

print(ndim_4_tensor.place)