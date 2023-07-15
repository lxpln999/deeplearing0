import paddle

ndim_1_tensor = paddle.to_tensor([2.0, 3.0, 4.0])
print(ndim_1_tensor)

ndim_2_tensor = paddle.to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
print(ndim_2_tensor)

ndim_3_tensor = paddle.to_tensor([[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
                                  [[11, 12, 13, 14, 15], [16, 17, 18, 19, 20]]])
print(ndim_3_tensor)
print('shape of ndim_3_tensor is', ndim_3_tensor.shape[2])

m, n = 2, 3
zeros_tensor = paddle.zeros([m, n])
ones_tensor = paddle.ones([m, n])
full_tensor = paddle.full([m, n], 10)

print(zeros_tensor)
print(ones_tensor)
print(full_tensor)

arange_tensor = paddle.arange(start=1, end=5, step=1)
linspace_tensor = paddle.linspace(start=1, stop=5, num=5)

print('arange_tensor', arange_tensor)
print('linspace_tensor', linspace_tensor)

ndis_4_tensor = paddle.ones([2, 3, 4, 5])
print('number of dimensions', ndis_4_tensor.ndim)
print('shape of tenson', ndis_4_tensor.shape)
print('elements number along axis 0 of tensor', ndis_4_tensor.shape[0])
print('elements along the last axis of tensor', ndis_4_tensor.shape[-1])
print('number of elements in tensor', ndis_4_tensor.size)

ndim_3_tensor = paddle.to_tensor([[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], [[11, 12, 13, 14, 15], [
                                 16, 17, 18, 19, 20]], [[21, 22, 23, 24, 25], [26, 27, 28, 29, 30]]])
print('ndim_3_tensor shage', ndim_3_tensor.shape)
reshape_tensor = paddle.reshape(ndim_3_tensor, [3, 2, 5])
print('after reshape', reshape_tensor)

new_reshape_tensor = ndim_3_tensor.reshape([-1])
print('new_reshape', new_reshape_tensor)
print('new_reshape_tensor', new_reshape_tensor.shape)

new_reshape_tensor2 = ndim_3_tensor.reshape([0, 5, 2])
print('new_reshape_tensor2', new_reshape_tensor2)
print('new_reshape_tensor2 shape ', new_reshape_tensor2.shape)

ones_tensor = paddle.ones([5, 10])
print(ones_tensor.shape)
print(ones_tensor)

new_one_tensor = paddle.unsqueeze(ones_tensor, axis=0)
print(new_one_tensor.shape)
print(new_one_tensor)

new_one_tensor2 = paddle.unsqueeze(ones_tensor, axis=[1, 2])
print(new_one_tensor2.shape)
print(new_one_tensor2)

print(new_one_tensor.dtype)

print('tensor dtype from integers', paddle.to_tensor(1).dtype)
print('tensor dtype from floats', paddle.to_tensor(1.0).dtype)

ndim_2_tensor = paddle.to_tensor([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]])
print('ndim_2_tensor', ndim_2_tensor)
print('first row', ndim_2_tensor[0, :])
print('first column', ndim_2_tensor[:, 0])

x = paddle.to_tensor([[1.1, 2.2], [3.3, 4.4]])
print('x',x)
print('x.max',x.max())

y = paddle.to_tensor([[5.1, 6.2], [7.3, 8.4]])

print(x.isfinite())

print('x.t',x.t())