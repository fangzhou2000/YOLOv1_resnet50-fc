import torch
a = torch.Tensor([
    [
        [[1,2,3],
         [4,5,6],
         [7,8,9]]
    ]
])
print(a.shape)
b = a.numpy()[:,:]
print(b.shape)